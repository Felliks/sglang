from __future__ import annotations

import logging
import threading
import time

import torch
from sglang.srt.layers.moe.expert_offload.adapters.marlin import (
    MarlinExpertTensors,
)

logger = logging.getLogger(__name__)
from sglang.srt.layers.moe.expert_offload.interfaces import SlotLease
from sglang.srt.layers.moe.expert_offload.scheduler import (
    DeadlineExpertScheduler,
    PrefetchCandidate,
)


class PreparedScheduledMarlinExperts:
    def __init__(
        self,
        adapter: ScheduledMarlinExpertOffload,
        leases: list[SlotLease],
    ) -> None:
        self.tensors = adapter.resident_tensors
        self.expert_map = adapter.expert_map
        self.global_num_experts = adapter.num_experts
        self._adapter = adapter
        self._leases = leases
        self._completed = False

    def record_completion(self) -> None:
        if self._completed:
            raise RuntimeError("Marlin expert submission was already completed")
        self._adapter.record_kernel_completion(self._leases)
        self._completed = True


class ScheduledMarlinExpertOffload:
    """One layer's Marlin view backed by a shared deadline scheduler."""

    def __init__(
        self,
        *,
        layer_index: int,
        num_experts: int,
        resident_tensors: MarlinExpertTensors,
        expert_map: torch.Tensor,
        scheduler: DeadlineExpertScheduler,
        predictor=None,
        predictor_target_layer: int | None = None,
        prefetch_candidates: int = 0,
        prefetch_deadline_ns: int = 0,
    ) -> None:
        self.layer_index = layer_index
        self.num_experts = num_experts
        self.resident_tensors = resident_tensors
        self.expert_map = expert_map
        self._scheduler = scheduler
        self._predictor = predictor
        self._predictor_target_layer = predictor_target_layer
        self._prefetch_candidates = prefetch_candidates
        self._prefetch_deadline_ns = prefetch_deadline_ns
        self._pending: list[tuple[torch.cuda.Event, list[SlotLease]]] = []
        self._cuda_stream: int | None = None
        self._owner_thread = threading.get_ident()
        self._prepare_calls = 0

    def _check_context(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("scheduled Marlin offload changed model thread")
        device = self.resident_tensors.w13_qweight.device
        if device.type != "cuda":
            return
        stream_id = torch.cuda.current_stream(device).cuda_stream
        if self._cuda_stream is None:
            self._cuda_stream = stream_id
        elif self._cuda_stream != stream_id:
            raise RuntimeError(
                "scheduled Marlin offload currently requires one CUDA stream"
            )

    def _reap_kernel_leases(self) -> None:
        remaining = []
        for event, leases in self._pending:
            if event.query():
                self._scheduler.release(leases)
            else:
                remaining.append((event, leases))
        self._pending = remaining

    def _prefetch(self, hidden_states: torch.Tensor) -> None:
        if (
            self._predictor is None
            or self._predictor_target_layer is None
            or self._prefetch_candidates <= 0
            or hidden_states.numel() == 0
        ):
            return
        predicted = self._predictor(hidden_states)
        if isinstance(predicted, tuple):
            predicted = predicted[0]
        aggregate = predicted.float().amax(dim=0)
        count = min(self._prefetch_candidates, aggregate.numel())
        values, ids = torch.topk(aggregate, count)
        scores = torch.softmax(values, dim=0)
        ids_cpu = ids.detach().cpu().tolist()
        scores_cpu = scores.detach().cpu().tolist()
        deadline_ns = time.monotonic_ns() + self._prefetch_deadline_ns
        self._scheduler.prefetch(
            PrefetchCandidate(
                identity=self._identity(self._predictor_target_layer, expert_id),
                score=score,
                deadline_ns=deadline_ns,
            )
            for expert_id, score in zip(ids_cpu, scores_cpu, strict=True)
        )

    @staticmethod
    def _identity(layer_index: int, expert_id: int):
        from sglang.srt.layers.moe.expert_offload.interfaces import ExpertIdentity

        return ExpertIdentity(layer_index, expert_id)

    def prepare(
        self,
        topk_ids: torch.Tensor,
        *,
        hidden_states: torch.Tensor | None = None,
    ) -> PreparedScheduledMarlinExperts:
        self._check_context()
        self._prepare_calls += 1
        self._reap_kernel_leases()
        self._scheduler.poll()
        if hidden_states is not None:
            self._prefetch(hidden_states)
        expert_ids = sorted(
            {
                int(expert_id)
                for expert_id in topk_ids.detach().reshape(-1).cpu().tolist()
                if 0 <= int(expert_id) < self.num_experts
            }
        )
        leases = self._scheduler.demand(
            self._identity(self.layer_index, expert_id) for expert_id in expert_ids
        )
        if self.layer_index == 0 and self._prepare_calls % 64 == 0:
            logger.info(
                "Expert offload runtime metrics: %s",
                self._scheduler.snapshot_metrics(),
            )
        return PreparedScheduledMarlinExperts(self, leases)

    def record_kernel_completion(self, leases: list[SlotLease]) -> None:
        device = self.resident_tensors.w13_qweight.device
        if device.type == "cuda":
            event = torch.cuda.Event(blocking=False)
            event.record(torch.cuda.current_stream(device))
            self._pending.append((event, leases))
        else:
            self._scheduler.release(leases)
