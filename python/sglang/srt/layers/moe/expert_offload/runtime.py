from __future__ import annotations

import gc
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.srt.layers.moe.expert_offload.adapters.marlin import MarlinExpertTensors
from sglang.srt.layers.moe.expert_offload.adapters.marlin_record import (
    BoundMarlinRecordPublisher,
    marlin_packing_fingerprint,
    marlin_tensor_segments,
)
from sglang.srt.layers.moe.expert_offload.adapters.marlin_scheduled import (
    ScheduledMarlinExpertOffload,
)
from sglang.srt.layers.moe.expert_offload.cache import PartitionedExpertCache
from sglang.srt.layers.moe.expert_offload.interfaces import (
    Admission,
    ExpertIdentity,
    ExpertRecordLease,
)
from sglang.srt.layers.moe.expert_offload.scheduler import DeadlineExpertScheduler
from sglang.srt.layers.moe.expert_offload.storage.builder import (
    build_marlin_expert_store,
)
from sglang.srt.layers.moe.expert_offload.storage.cuda_host import (
    CudaHostRegistration,
)
from sglang.srt.layers.moe.expert_offload.storage.manifest import ExpertStoreManifest
from sglang.srt.layers.moe.expert_offload.storage.nvme_direct import (
    DirectExpertStorage,
)
from sglang.srt.model_executor.cuda_graph_config import Backend

logger = logging.getLogger(__name__)


@dataclass
class _LayerBinding:
    name: str
    sparse_block: torch.nn.Module
    experts: torch.nn.Module
    source: MarlinExpertTensors | None


class _PublisherRegistry:
    def __init__(self, publishers: list[BoundMarlinRecordPublisher]) -> None:
        self._publishers = publishers

    def publish(self, record: ExpertRecordLease, slot_id: int):
        return self._publishers[record.identity.layer_id].publish(record, slot_id)


class ExpertOffloadCoordinator:
    """Own process-wide storage/scheduling and per-layer Marlin slot views."""

    def __init__(self, storage, scheduler, runtimes) -> None:
        self.storage = storage
        self.scheduler = scheduler
        self.runtimes = runtimes

    def close(self) -> None:
        self.storage.close()


def _marlin_tensors(layer: torch.nn.Module) -> MarlinExpertTensors:
    return MarlinExpertTensors(
        w13_qweight=layer.w13_weight,
        w2_qweight=layer.w2_weight,
        w13_scales=layer.w13_weight_scale,
        w2_scales=layer.w2_weight_scale,
        w13_global_scale=layer.w13_weight_scale_2,
        w2_global_scale=layer.w2_weight_scale_2,
    )


def _collect_layers(model: torch.nn.Module) -> list[_LayerBinding]:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    found = []
    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        gate = getattr(module, "gate", None)
        if not isinstance(experts, FusedMoE) or gate is None:
            continue
        method = getattr(experts, "quant_method", None)
        if not isinstance(method, ModelOptNvFp4FusedMoEMethod):
            continue
        backend = getattr(method, "_moe_runner_backend", None)
        if backend is None or not backend.is_marlin():
            continue
        found.append(
            _LayerBinding(
                name=name,
                sparse_block=module,
                experts=experts,
                source=_marlin_tensors(experts),
            )
        )
    found.sort(key=lambda item: item.experts.moe_runner_config.layer_id)
    layer_ids = [item.experts.moe_runner_config.layer_id for item in found]
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError("expert offload requires unique local MoE layer IDs")
    return found


def _model_fingerprint(model_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(model_path.resolve()).encode())
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "nightlab-manifest.json",
    ):
        path = model_path / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    for path in sorted(model_path.glob("*.safetensors")):
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(path.resolve()).encode())
    return digest.hexdigest()


def _manifest_path(root: Path, tp_rank: int, pp_rank: int) -> Path:
    if root.suffix == ".json":
        return root
    return root / f"target-tp{tp_rank}-pp{pp_rank}" / "manifest.json"


def _seed_ids(path: str | None, layers: int, experts: int, capacity: int):
    if path is None:
        return [list(range(capacity)) for _ in range(layers)]
    payload = json.loads(Path(path).read_text())
    values = payload.get("layers", payload)
    result = []
    for layer in range(layers):
        raw = values[str(layer)] if isinstance(values, dict) else values[layer]
        selected = []
        for expert_id in raw:
            expert_id = int(expert_id)
            if not 0 <= expert_id < experts:
                raise ValueError(f"seed expert is outside layer {layer}: {expert_id}")
            if expert_id not in selected:
                selected.append(expert_id)
            if len(selected) == capacity:
                break
        if len(selected) != capacity:
            raise ValueError(f"seed layer {layer} has fewer than {capacity} experts")
        result.append(selected)
    return result


def _resident_capacity(server_args, layers: int, experts: int, row_bytes: int) -> int:
    if server_args.expert_resident_ratio is not None:
        capacity = round(experts * server_args.expert_resident_ratio)
    else:
        budget_bytes = server_args.expert_resident_gib * (1 << 30)
        capacity = int(budget_bytes // (layers * row_bytes))
    if not 0 < capacity <= experts:
        raise ValueError(
            f"expert resident budget produces {capacity} slots per layer; "
            f"expected [1, {experts}]"
        )
    return capacity


def _slot_tensors(
    source: MarlinExpertTensors, seed_ids: list[int]
) -> MarlinExpertTensors:
    indices = torch.tensor(seed_ids, dtype=torch.long, device=source.w13_qweight.device)

    def select(tensor):
        return None if tensor is None else tensor.index_select(0, indices).contiguous()

    return MarlinExpertTensors(
        w13_qweight=select(source.w13_qweight),
        w2_qweight=select(source.w2_qweight),
        w13_scales=select(source.w13_scales),
        w2_scales=select(source.w2_scales),
        w13_global_scale=select(source.w13_global_scale),
        w2_global_scale=select(source.w2_global_scale),
    )


def _replace_layer_tensors(layer, resident: MarlinExpertTensors) -> None:
    for layer_name, tensor_name in (
        ("w13_weight", "w13_qweight"),
        ("w2_weight", "w2_qweight"),
        ("w13_weight_scale", "w13_scales"),
        ("w2_weight_scale", "w2_scales"),
        ("w13_weight_scale_2", "w13_global_scale"),
        ("w2_weight_scale_2", "w2_global_scale"),
    ):
        tensor = getattr(resident, tensor_name)
        if tensor is not None:
            layer.replace_expert_tensor(layer_name, tensor)


def install_expert_offload(
    *,
    model: torch.nn.Module,
    server_args,
    model_path: str,
    is_draft_worker: bool,
    tp_rank: int,
    pp_rank: int,
) -> ExpertOffloadCoordinator | None:
    if server_args.expert_offload_backend == "none" or is_draft_worker:
        return None
    if server_args.expert_offload_backend != "nvme":
        raise NotImplementedError("live expert offload currently requires backend=nvme")
    if server_args.moe_a2a_backend != "none" or server_args.ep_size != 1:
        raise ValueError("initial Marlin expert offload supports EP1/a2a=none only")
    if not server_args.disable_overlap_schedule:
        raise ValueError("initial expert offload requires --disable-overlap-schedule")
    if (
        server_args.cuda_graph_config.decode.backend != Backend.DISABLED
        or server_args.cuda_graph_config.prefill.backend != Backend.DISABLED
    ):
        raise ValueError("initial expert offload requires CUDA graphs to be disabled")

    bindings = _collect_layers(model)
    if not bindings:
        raise ValueError("no ModelOpt NVFP4 Marlin MoE layers were found")
    first_source = bindings[0].source
    assert first_source is not None
    experts = first_source.w13_qweight.shape[0]
    if any(
        binding.source is None or binding.source.w13_qweight.shape[0] != experts
        for binding in bindings
    ):
        raise ValueError("all offloaded layers must share one logical expert count")
    segments = marlin_tensor_segments(first_source)
    if any(
        binding.source is None or marlin_tensor_segments(binding.source) != segments
        for binding in bindings
    ):
        raise ValueError("all offloaded layers must share one Marlin record layout")
    raw_bytes = max(segment.offset + segment.nbytes for segment in segments)
    capacity = _resident_capacity(server_args, len(bindings), experts, raw_bytes)
    seeds = _seed_ids(server_args.expert_seed_path, len(bindings), experts, capacity)

    manifest_path = _manifest_path(
        Path(server_args.expert_storage_path), tp_rank, pp_rank
    )
    fingerprint = _model_fingerprint(Path(model_path))
    if manifest_path.exists():
        manifest = ExpertStoreManifest.load(manifest_path)
        if (
            manifest.model_fingerprint != fingerprint
            or manifest.packing_fingerprint != marlin_packing_fingerprint(segments)
            or manifest.num_layers != len(bindings)
            or manifest.experts_per_layer != experts
            or manifest.tensor_segments != segments
        ):
            raise ValueError(
                "existing expert cold store is incompatible with the model"
            )
        logger.info("Reusing expert cold store %s", manifest_path)
    else:
        logger.info(
            "Building expert cold store: layers=%d experts=%d path=%s",
            len(bindings),
            experts,
            manifest_path,
        )
        sources = [binding.source for binding in bindings]
        assert all(source is not None for source in sources)
        manifest = build_marlin_expert_store(
            manifest_path,
            sources,
            model_fingerprint=fingerprint,
            direct=True,
        )
        del sources

    cache = PartitionedExpertCache(len(bindings), capacity)
    resident_tensors = []
    expert_maps = []
    publishers = []
    for layer_index, (binding, seed) in enumerate(zip(bindings, seeds, strict=True)):
        # Replace one layer at a time.  Allocating every resident cache before
        # releasing the complete model would transiently require both copies
        # and can exhaust unified memory on single-socket systems.
        source = binding.source
        assert source is not None
        resident = _slot_tensors(source, seed)
        _replace_layer_tensors(binding.experts, resident)
        binding.source = None
        del source
        gc.collect()
        if resident.w13_qweight.device.type == "cuda":
            torch.cuda.empty_cache()
        expert_map = torch.full(
            (experts,), -1, dtype=torch.int32, device=resident.w13_qweight.device
        )
        for expected_slot, expert_id in enumerate(seed):
            admission = cache.admit(ExpertIdentity(layer_index, expert_id))
            if admission.slot_id != expected_slot:
                raise RuntimeError("seed cache slot order is inconsistent")
            cache.publish(admission)
            expert_map[expert_id] = expected_slot
        resident_tensors.append(resident)
        expert_maps.append(expert_map)
        publishers.append(BoundMarlinRecordPublisher(resident, segments))

    storage = DirectExpertStorage(
        manifest_path,
        io_depth=server_args.expert_prefetch_io_depth,
        direct=True,
        verify_checksums=server_args.expert_verify_store_checksums,
        buffer_registrar=CudaHostRegistration,
    )

    def on_evict(identity: ExpertIdentity) -> None:
        expert_maps[identity.layer_id][identity.expert_id] = -1

    def on_resident(admission: Admission) -> None:
        identity = admission.identity
        expert_maps[identity.layer_id][identity.expert_id] = admission.slot_id

    scheduler = DeadlineExpertScheduler(
        cache=cache,
        storage=storage,
        publisher=_PublisherRegistry(publishers),
        io_depth=server_args.expert_prefetch_io_depth,
        initial_service_ns=int(server_args.expert_prefetch_initial_latency_ms * 1e6),
        on_evict=on_evict,
        on_resident=on_resident,
        max_pending=max(server_args.expert_prefetch_candidates * 8, 64),
    )

    horizon = server_args.expert_prefetch_layer_horizon
    runtimes = []
    deadline_ns = int(server_args.expert_prefetch_layer_ms * horizon * 1e6)
    for layer_index, binding in enumerate(bindings):
        target = layer_index + horizon
        predictor = (
            bindings[target].sparse_block.gate if target < len(bindings) else None
        )
        runtime = ScheduledMarlinExpertOffload(
            layer_index=layer_index,
            num_experts=experts,
            resident_tensors=resident_tensors[layer_index],
            expert_map=expert_maps[layer_index],
            scheduler=scheduler,
            predictor=predictor,
            predictor_target_layer=target if predictor is not None else None,
            prefetch_candidates=server_args.expert_prefetch_candidates,
            prefetch_deadline_ns=deadline_ns,
        )
        binding.experts.expert_offload = runtime
        runtimes.append(runtime)

    del resident_tensors, publishers, bindings
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(
        "Expert offload installed: layers=%d experts=%d resident_per_layer=%d "
        "resident_ratio=%.4f record_bytes=%d io_depth=%d horizon=%d candidates=%d",
        manifest.num_layers,
        experts,
        capacity,
        capacity / experts,
        manifest.record_bytes,
        server_args.expert_prefetch_io_depth,
        horizon,
        server_args.expert_prefetch_candidates,
    )
    return ExpertOffloadCoordinator(storage, scheduler, runtimes)
