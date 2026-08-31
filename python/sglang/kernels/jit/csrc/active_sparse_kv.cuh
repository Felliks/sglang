#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdint.h>
#include <stdexcept>

namespace sglang {

template <int BLOCK_SIZE, int BLOCK_TOKENS, int RECORD_BYTES>
__global__ void unpack_qsa_records_kernel(
    const char* __restrict__ staging,
    char* __restrict__ destination_k,
    char* __restrict__ destination_v,
    const int32_t* __restrict__ destination_blocks,
    int32_t num_records,
    int64_t staging_stride,
    int64_t destination_token_stride) {
  constexpr int HALF_RECORD_BYTES = RECORD_BYTES / 2;
  constexpr int VECTOR_BYTES = 16;
  constexpr int VECTORS_PER_HALF = HALF_RECORD_BYTES / VECTOR_BYTES;
  const int record = blockIdx.x;
  if (record >= num_records) {
    return;
  }
  const int32_t destination_block = destination_blocks[record];
  if (destination_block < 0) {
    return;
  }
  const char* source = staging + static_cast<int64_t>(record) * staging_stride;
  char* output_k = destination_k +
      static_cast<int64_t>(destination_block) * BLOCK_TOKENS * destination_token_stride;
  char* output_v = destination_v +
      static_cast<int64_t>(destination_block) * BLOCK_TOKENS * destination_token_stride;
  const uint4* source_k = reinterpret_cast<const uint4*>(source);
  const uint4* source_v = reinterpret_cast<const uint4*>(source + HALF_RECORD_BYTES);
  uint4* output_k_vec = reinterpret_cast<uint4*>(output_k);
  uint4* output_v_vec = reinterpret_cast<uint4*>(output_v);
  for (int vector = threadIdx.x; vector < VECTORS_PER_HALF; vector += BLOCK_SIZE) {
    output_k_vec[vector] = source_k[vector];
    output_v_vec[vector] = source_v[vector];
  }
}

template <int BLOCK_SIZE, int BLOCK_TOKENS, int RECORD_BYTES>
__global__ void pack_qsa_records_kernel(
    const char* __restrict__ source_k,
    const char* __restrict__ source_v,
    const int32_t* __restrict__ source_blocks,
    char* __restrict__ staging,
    int32_t num_records,
    int64_t source_token_stride,
    int64_t staging_stride) {
  constexpr int HALF_RECORD_BYTES = RECORD_BYTES / 2;
  constexpr int VECTOR_BYTES = 16;
  constexpr int VECTORS_PER_HALF = HALF_RECORD_BYTES / VECTOR_BYTES;
  const int record = blockIdx.x;
  if (record >= num_records) {
    return;
  }
  const int32_t source_block = source_blocks[record];
  if (source_block < 0) {
    return;
  }
  const char* input_k = source_k +
      static_cast<int64_t>(source_block) * BLOCK_TOKENS * source_token_stride;
  const char* input_v = source_v +
      static_cast<int64_t>(source_block) * BLOCK_TOKENS * source_token_stride;
  char* destination = staging + static_cast<int64_t>(record) * staging_stride;
  const uint4* input_k_vec = reinterpret_cast<const uint4*>(input_k);
  const uint4* input_v_vec = reinterpret_cast<const uint4*>(input_v);
  uint4* destination_k = reinterpret_cast<uint4*>(destination);
  uint4* destination_v = reinterpret_cast<uint4*>(destination + HALF_RECORD_BYTES);
  for (int vector = threadIdx.x; vector < VECTORS_PER_HALF; vector += BLOCK_SIZE) {
    destination_k[vector] = input_k_vec[vector];
    destination_v[vector] = input_v_vec[vector];
  }
}

template <int BLOCK_SIZE, int BLOCK_TOKENS, int RECORD_BYTES>
void unpack_qsa_records(tvm::ffi::TensorView staging,
                        tvm::ffi::TensorView destination_k,
                        tvm::ffi::TensorView destination_v,
                        tvm::ffi::TensorView destination_blocks,
                        int64_t num_records) {
  using namespace host;
  static_assert(RECORD_BYTES % 32 == 0, "K and V halves must be 16-byte aligned");
  if (num_records < 0 || num_records > staging.shape()[0] ||
      num_records > destination_blocks.shape()[0]) {
    throw std::runtime_error("unpack_qsa_records: invalid record count");
  }
  if (destination_k.ndim() != 3 || destination_v.ndim() != 3 ||
      staging.ndim() != 2 || destination_blocks.ndim() != 1) {
    throw std::runtime_error("unpack_qsa_records: invalid tensor rank");
  }
  if (staging.shape()[1] != RECORD_BYTES ||
      destination_k.strides()[0] != destination_v.strides()[0] ||
      destination_k.strides()[0] * destination_k.dtype().bits / 8 * BLOCK_TOKENS != RECORD_BYTES / 2) {
    throw std::runtime_error("unpack_qsa_records: geometry mismatch");
  }
  if (num_records == 0) {
    return;
  }
  const auto device = LaunchKernel::resolve_device(destination_k.device());
  if (destination_blocks.dtype().code != kDLInt ||
      destination_blocks.dtype().bits != 32) {
    throw std::runtime_error("unpack_qsa_records: destination blocks must be int32");
  }
  LaunchKernel(num_records, BLOCK_SIZE, device)(
      unpack_qsa_records_kernel<BLOCK_SIZE, BLOCK_TOKENS, RECORD_BYTES>,
      static_cast<const char*>(staging.data_ptr()),
      static_cast<char*>(destination_k.data_ptr()),
      static_cast<char*>(destination_v.data_ptr()),
      static_cast<const int32_t*>(destination_blocks.data_ptr()),
      static_cast<int32_t>(num_records),
      staging.strides()[0],
      destination_k.strides()[0] * destination_k.dtype().bits / 8);
}

template <int BLOCK_SIZE, int BLOCK_TOKENS, int RECORD_BYTES>
void pack_qsa_records(tvm::ffi::TensorView source_k,
                      tvm::ffi::TensorView source_v,
                      tvm::ffi::TensorView source_blocks,
                      tvm::ffi::TensorView staging,
                      int64_t num_records) {
  using namespace host;
  static_assert(RECORD_BYTES % 32 == 0, "K and V halves must be 16-byte aligned");
  if (num_records < 0 || num_records > staging.shape()[0] ||
      num_records > source_blocks.shape()[0]) {
    throw std::runtime_error("pack_qsa_records: invalid record count");
  }
  if (source_k.ndim() != 3 || source_v.ndim() != 3 || staging.ndim() != 2 ||
      source_blocks.ndim() != 1) {
    throw std::runtime_error("pack_qsa_records: invalid tensor rank");
  }
  if (staging.shape()[1] != RECORD_BYTES ||
      source_k.strides()[0] != source_v.strides()[0] ||
      source_k.strides()[0] * source_k.dtype().bits / 8 * BLOCK_TOKENS != RECORD_BYTES / 2) {
    throw std::runtime_error("pack_qsa_records: geometry mismatch");
  }
  if (num_records == 0) {
    return;
  }
  const auto device = LaunchKernel::resolve_device(source_k.device());
  if (source_blocks.dtype().code != kDLInt || source_blocks.dtype().bits != 32) {
    throw std::runtime_error("pack_qsa_records: source blocks must be int32");
  }
  LaunchKernel(num_records, BLOCK_SIZE, device)(
      pack_qsa_records_kernel<BLOCK_SIZE, BLOCK_TOKENS, RECORD_BYTES>,
      static_cast<const char*>(source_k.data_ptr()),
      static_cast<const char*>(source_v.data_ptr()),
      static_cast<const int32_t*>(source_blocks.data_ptr()),
      static_cast<char*>(staging.data_ptr()),
      static_cast<int32_t>(num_records),
      source_k.strides()[0] * source_k.dtype().bits / 8,
      staging.strides()[0]);
}

template <int BLOCK_SIZE, int BLOCK_TOKENS, int RECORD_BYTES>
__global__ void resolve_qsa_slots_kernel(
    const int32_t* __restrict__ logical_slots,
    const int32_t* __restrict__ logical_to_hot,
    const int32_t* __restrict__ hot_to_logical,
    int32_t* __restrict__ physical_slots,
    int32_t* __restrict__ selected_blocks,
    int32_t* __restrict__ seen_epochs,
    int32_t* __restrict__ selected_count,
    int32_t* __restrict__ miss_count,
    int32_t epoch,
    int64_t num_slots,
    int64_t logical_blocks,
    int64_t hot_blocks) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
       index < num_slots;
       index += static_cast<int64_t>(gridDim.x) * BLOCK_SIZE) {
    const int32_t logical_slot = logical_slots[index];
    if (logical_slot < 0) {
      physical_slots[index] = -1;
      continue;
    }
    const int32_t logical_block = logical_slot / BLOCK_TOKENS;
    if (logical_block < 0 || logical_block >= logical_blocks) {
      physical_slots[index] = -1;
      atomicAdd(miss_count, 1);
      continue;
    }

    // One representative per logical block is emitted without sorting the
    // full [rows, top-k] selection matrix.  Epoch tags avoid clearing a large
    // bitmap on every layer/step.
    const int32_t old_epoch = atomicExch(&seen_epochs[logical_block], epoch);
    if (old_epoch != epoch) {
      const int32_t output = atomicAdd(selected_count, 1);
      selected_blocks[output] = logical_block;
    }

    const int32_t hot_block = logical_to_hot[logical_block];
    const bool resident = hot_block >= 0 && hot_block < hot_blocks &&
        hot_to_logical[hot_block] == logical_block;
    physical_slots[index] = resident
        ? hot_block * BLOCK_TOKENS + logical_slot % BLOCK_TOKENS
        : -1;
    if (!resident) {
      atomicAdd(miss_count, 1);
    }
  }
}

template <int BLOCK_SIZE, int BLOCK_TOKENS, int RECORD_BYTES>
void resolve_qsa_slots(tvm::ffi::TensorView logical_slots,
                       tvm::ffi::TensorView logical_to_hot,
                       tvm::ffi::TensorView hot_to_logical,
                       tvm::ffi::TensorView physical_slots,
                       tvm::ffi::TensorView selected_blocks,
                       tvm::ffi::TensorView seen_epochs,
                       tvm::ffi::TensorView selected_count,
                       tvm::ffi::TensorView miss_count,
                       int64_t epoch) {
  using namespace host;
  if (logical_slots.dtype().code != kDLInt || logical_slots.dtype().bits != 32 ||
      logical_to_hot.dtype().code != kDLInt || logical_to_hot.dtype().bits != 32 ||
      hot_to_logical.dtype().code != kDLInt || hot_to_logical.dtype().bits != 32 ||
      physical_slots.dtype().code != kDLInt || physical_slots.dtype().bits != 32) {
    throw std::runtime_error("resolve_qsa_slots: slot tensors must be int32");
  }
  if (logical_to_hot.ndim() != 1 || hot_to_logical.ndim() != 1 ||
      selected_blocks.ndim() != 1 || seen_epochs.ndim() != 1 ||
      selected_count.ndim() != 1 || miss_count.ndim() != 1) {
    throw std::runtime_error("resolve_qsa_slots: invalid tensor rank");
  }
  int64_t num_slots = 1;
  int64_t physical_num_slots = 1;
  for (int dim = 0; dim < logical_slots.ndim(); ++dim) {
    num_slots *= logical_slots.shape()[dim];
  }
  for (int dim = 0; dim < physical_slots.ndim(); ++dim) {
    physical_num_slots *= physical_slots.shape()[dim];
  }
  if (num_slots == 0) {
    return;
  }
  if (physical_num_slots != num_slots ||
      selected_blocks.shape()[0] < logical_to_hot.shape()[0] ||
      seen_epochs.shape()[0] != logical_to_hot.shape()[0] ||
      selected_count.shape()[0] != 1 || miss_count.shape()[0] != 1) {
    throw std::runtime_error("resolve_qsa_slots: geometry mismatch");
  }
  const auto device = LaunchKernel::resolve_device(logical_slots.device());
  const int64_t blocks = (num_slots + BLOCK_SIZE - 1) / BLOCK_SIZE;
  LaunchKernel(blocks, BLOCK_SIZE, device)(
      resolve_qsa_slots_kernel<BLOCK_SIZE, BLOCK_TOKENS, RECORD_BYTES>,
      static_cast<const int32_t*>(logical_slots.data_ptr()),
      static_cast<const int32_t*>(logical_to_hot.data_ptr()),
      static_cast<const int32_t*>(hot_to_logical.data_ptr()),
      static_cast<int32_t*>(physical_slots.data_ptr()),
      static_cast<int32_t*>(selected_blocks.data_ptr()),
      static_cast<int32_t*>(seen_epochs.data_ptr()),
      static_cast<int32_t*>(selected_count.data_ptr()),
      static_cast<int32_t*>(miss_count.data_ptr()),
      static_cast<int32_t>(epoch),
      num_slots,
      logical_to_hot.shape()[0],
      hot_to_logical.shape()[0]);
}

}  // namespace sglang
