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

}  // namespace sglang
