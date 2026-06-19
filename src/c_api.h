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

// Mirrors library.h's MathMode. Metal's default (FAST) permits reassociation,
// which deletes compensated-arithmetic error terms (df32);
// SAFE is required for those.
typedef enum MRMathMode {
    MR_MATH_MODE_SAFE = 0,
    MR_MATH_MODE_RELAXED = 1,
    MR_MATH_MODE_FAST = 2,
} MRMathMode;

// No-op on NULL. *out_err_msg is untouched on success everywhere below.
void mr_free_error_message(char* msg);

// Compiles MSL source. Call once at registration time. Part of the
// library identity: the same source under a different math_mode is a
// different library, matching library.h's own CompileOptions contract.
MRStatus mr_compile_library(const char* msl_source, size_t msl_source_len, MRMathMode math_mode,
                            MRLibrary** out_library, char** out_err_msg);
void mr_release_library(MRLibrary* library);

// Builds (and caches) the named kernel's pipeline.
MRStatus mr_get_pipeline(MRLibrary* library, const char* function_name, MRPipeline** out_pipeline,
                         char** out_err_msg);
void mr_release_pipeline(MRPipeline* pipeline);

// Wraps `ptr` for use in mr_dispatch. Zero-copy for any `ptr`/`size_bytes`,
// not just page-aligned ones: rounds the range out to the enclosing page
// boundaries (newBufferWithBytesNoCopy's hard requirement) and dispatch
// binds at the true data offset within that rounded region automatically.
// Falls back to an owned copy, synced back by mr_buffer_flush_to, only if
// the rounded range would reach outside ptr's actual VM mapping (checked
// via mach_vm_region before ever handing memory to Metal). Caller keeps
// owning `ptr`; this never frees it.
//
// Real xla::ffi::Buffer<F32> pointers were measured page-aligned on neither
// `ptr` nor `size_bytes`, so page-rounding is the path that actually matters,
// not an edge case.
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
