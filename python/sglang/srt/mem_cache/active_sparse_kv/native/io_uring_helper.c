#define _GNU_SOURCE

// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/uio.h>
#include <unistd.h>

enum sglang_active_kv_operation {
    SGLANG_ACTIVE_KV_READ = 0,
    SGLANG_ACTIVE_KV_WRITE = 1,
};

struct sglang_active_kv_uring {
    struct io_uring ring;
    int fd;
    unsigned depth;
    size_t buffer_bytes;
    void **buffers;
};

uint32_t sglang_active_kv_uring_abi_version(void) { return 1; }

int sglang_active_kv_uring_create(const char *path, void *const *buffers,
                                  size_t buffer_bytes, unsigned depth,
                                  void **handle_out) {
    if (!path || !buffers || !buffer_bytes || !depth || !handle_out) {
        return -EINVAL;
    }
    *handle_out = NULL;
    struct sglang_active_kv_uring *handle = calloc(1, sizeof(*handle));
    struct iovec *iovecs = calloc(depth, sizeof(*iovecs));
    if (handle) {
        handle->buffers = calloc(depth, sizeof(*handle->buffers));
    }
    if (!handle || !iovecs || !handle->buffers) {
        free(iovecs);
        if (handle) {
            free(handle->buffers);
        }
        free(handle);
        return -ENOMEM;
    }
    handle->fd = -1;
    handle->depth = depth;
    handle->buffer_bytes = buffer_bytes;
    struct io_uring_params params = {0};
    params.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_COOP_TASKRUN;
    int result = io_uring_queue_init_params(depth * 2, &handle->ring, &params);
    if (result < 0) {
        free(handle->buffers);
        free(iovecs);
        free(handle);
        return result;
    }
    for (unsigned index = 0; index < depth; ++index) {
        handle->buffers[index] = buffers[index];
        iovecs[index].iov_base = buffers[index];
        iovecs[index].iov_len = buffer_bytes;
    }
    result = io_uring_register_buffers(&handle->ring, iovecs, depth);
    free(iovecs);
    if (result < 0) {
        io_uring_queue_exit(&handle->ring);
        free(handle->buffers);
        free(handle);
        return result;
    }
    handle->fd = open(path, O_RDWR | O_DIRECT | O_CLOEXEC);
    if (handle->fd < 0) {
        result = -errno;
        io_uring_unregister_buffers(&handle->ring);
        io_uring_queue_exit(&handle->ring);
        free(handle->buffers);
        free(handle);
        return result;
    }
    result = io_uring_register_files(&handle->ring, &handle->fd, 1);
    if (result < 0) {
        close(handle->fd);
        io_uring_unregister_buffers(&handle->ring);
        io_uring_queue_exit(&handle->ring);
        free(handle->buffers);
        free(handle);
        return result;
    }
    *handle_out = handle;
    return 0;
}

int sglang_active_kv_uring_submit_batch(
    void *opaque, int operation, const unsigned *buffer_indices,
    const int64_t *offsets, unsigned count) {
    struct sglang_active_kv_uring *handle = opaque;
    if (!handle || !buffer_indices || !offsets || !count ||
        count > handle->depth ||
        (operation != SGLANG_ACTIVE_KV_READ &&
         operation != SGLANG_ACTIVE_KV_WRITE)) {
        return -EINVAL;
    }
    for (unsigned item = 0; item < count; ++item) {
        const unsigned buffer_index = buffer_indices[item];
        if (buffer_index >= handle->depth || offsets[item] < 0) {
            return -EINVAL;
        }
        struct io_uring_sqe *sqe = io_uring_get_sqe(&handle->ring);
        if (!sqe) {
            return -EAGAIN;
        }
        if (operation == SGLANG_ACTIVE_KV_READ) {
            io_uring_prep_read_fixed(sqe, 0, handle->buffers[buffer_index],
                                     handle->buffer_bytes, offsets[item],
                                     buffer_index);
        } else {
            io_uring_prep_write_fixed(sqe, 0, handle->buffers[buffer_index],
                                      handle->buffer_bytes, offsets[item],
                                      buffer_index);
        }
        sqe->flags |= IOSQE_FIXED_FILE;
        io_uring_sqe_set_data64(sqe, buffer_index);
    }
    const int result = io_uring_submit(&handle->ring);
    return result == (int)count ? 0 : result < 0 ? result : -EIO;
}

int sglang_active_kv_uring_wait_batch(void *opaque, unsigned count,
                                     unsigned *buffer_indices_out,
                                     int *results_out) {
    struct sglang_active_kv_uring *handle = opaque;
    if (!handle || !count || count > handle->depth || !buffer_indices_out ||
        !results_out) {
        return -EINVAL;
    }
    for (unsigned item = 0; item < count; ++item) {
        struct io_uring_cqe *cqe = NULL;
        int result = io_uring_wait_cqe(&handle->ring, &cqe);
        if (result < 0) {
            return result;
        }
        buffer_indices_out[item] =
            (unsigned)io_uring_cqe_get_data64(cqe);
        results_out[item] = cqe->res;
        io_uring_cqe_seen(&handle->ring, cqe);
    }
    return 0;
}

void sglang_active_kv_uring_destroy(void *opaque) {
    struct sglang_active_kv_uring *handle = opaque;
    if (!handle) {
        return;
    }
    io_uring_unregister_files(&handle->ring);
    if (handle->fd >= 0) {
        close(handle->fd);
    }
    io_uring_unregister_buffers(&handle->ring);
    io_uring_queue_exit(&handle->ring);
    free(handle->buffers);
    free(handle);
}
