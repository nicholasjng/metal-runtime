// C-linkable entry point into metal-runtime's dispatch, for callers without
// Python or the GIL (e.g. an XLA FFI custom-call handler). Mirrors
// dispatch.h's Library/ComputePipeline/Buffer/dispatch(), flattened to
// opaque handles and POD structs. No C++ or Objective-C types in signatures.
//
// Ownership: compile once (mr_compile_library + mr_get_pipeline) at
// registration time, hold the handle for the process lifetime, release once
// at teardown. Per-call code (mr_wrap_buffer/mr_dispatch) never releases.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MRLibrary MRLibrary;
typedef struct MRPipeline MRPipeline;
typedef struct MRBuffer MRBuffer;

typedef enum MRStatus {
    MR_OK = 0,
    MR_ERROR_NO_DEVICE = 1,
    MR_ERROR_COMPILE = 2,
    MR_ERROR_FUNCTION_NOT_FOUND = 3,
    MR_ERROR_DISPATCH = 4,
    MR_ERROR_INVALID_ARGUMENT = 5,
    MR_ERROR_UNKNOWN = 6,
} MRStatus;

// No-op on NULL. *out_err_msg is untouched on success everywhere below.
void mr_free_error_message(char* msg);

// Compiles MSL source. Call once at registration time.
MRStatus mr_compile_library(const char* msl_source, size_t msl_source_len, MRLibrary** out_library,
                            char** out_err_msg);
void mr_release_library(MRLibrary* library);

// Builds (and caches) the named kernel's pipeline.
MRStatus mr_get_pipeline(MRLibrary* library, const char* function_name, MRPipeline** out_pipeline,
                         char** out_err_msg);
void mr_release_pipeline(MRPipeline* pipeline);

// Wraps `ptr` for use in mr_dispatch. Zero-copy if `ptr`/`size_bytes` are
// both page-aligned; otherwise falls back to an owned copy, synced back by
// mr_buffer_flush_to. Caller keeps owning `ptr`; this never frees it.
//
// Open item: which branch a real XLA FFI handler hits is unverified (no
// XLA/JAX dependency here to test against). To close it out:
//   1. In the handler, check `ptr % pagesize` and `size_bytes % pagesize`
//      for actual xla::ffi::Buffer<F32> pointers, across a few shapes.
//   2. Aligned: no change needed, zero-copy already fires automatically.
//   3. Not aligned: determine if that's inherent to XLA's allocator (copy
//      path is then permanent; measure its cost) or specific to how that
//      buffer was created (a different allocation strategy may avoid it).
//   4. Record the finding here and in palladium's NEXT.md/ROADMAP.md.
MRStatus mr_wrap_buffer(void* ptr, size_t size_bytes, MRBuffer** out_buffer, char** out_err_msg);

// Copies the buffer's contents back to `ptr`. No-op if the wrap was
// zero-copy. Call after mr_dispatch, before reading `ptr` again.
MRStatus mr_buffer_flush_to(MRBuffer* buffer, char** out_err_msg);

// Releases the handle. Never frees the pointer passed to mr_wrap_buffer.
void mr_release_buffer(MRBuffer* buffer);

// One kernel launch, flattened from dispatch.h's Launch. buffers/
// buffer_offsets and scalars/scalar_sizes are parallel arrays of
// buffer_count/scalar_count length. threadgroup_x == 0 picks a default via
// ComputePipeline::default_threadgroup.
typedef struct MRLaunchDesc {
    MRPipeline* pipeline;
    MRBuffer* const* buffers;
    const size_t* buffer_offsets;
    size_t buffer_count;
    const void* const* scalars;
    const size_t* scalar_sizes;
    size_t scalar_count;
    const size_t* threadgroup_memory;
    size_t threadgroup_memory_count;
    size_t grid_x, grid_y, grid_z;
    size_t threadgroup_x, threadgroup_y, threadgroup_z;
} MRLaunchDesc;

// Encodes and synchronously waits on one launch (dispatch.h's dispatch(),
// not CommandBatch: one Launch per FFI call).
MRStatus mr_dispatch(const MRLaunchDesc* launch, char** out_err_msg);

#ifdef __cplusplus
}  // extern "C"
#endif
