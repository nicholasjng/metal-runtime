// C entry point into metal-runtime for callers without Python or the GIL,
// such as an XLA FFI handler. Opaque handles and POD structs only.
//
// Compile once (mr_compile_library + mr_get_pipeline), hold the handles for
// the process lifetime, release at teardown.
#pragma once

#include <stddef.h>

#if defined(__GNUC__)
#define MR_EXPORT __attribute__((visibility("default")))
#else
#define MR_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MRLibrary MRLibrary;
typedef struct MRPipeline MRPipeline;
typedef struct MRBuffer MRBuffer;
typedef struct MRBatch MRBatch;

typedef enum MRStatus {
    MR_OK = 0,
    MR_ERROR_NO_DEVICE = 1,
    MR_ERROR_COMPILE = 2,
    MR_ERROR_FUNCTION_NOT_FOUND = 3,
    MR_ERROR_DISPATCH = 4,
    MR_ERROR_INVALID_ARGUMENT = 5,
    MR_ERROR_UNKNOWN = 6,
    MR_ERROR_ALLOCATION = 7,
} MRStatus;

// Mirrors library.h's MathMode. FAST (Metal's default) permits reassociation;
// compensated arithmetic needs SAFE.
typedef enum MRMathMode {
    MR_MATH_MODE_SAFE = 0,
    MR_MATH_MODE_RELAXED = 1,
    MR_MATH_MODE_FAST = 2,
} MRMathMode;

// No-op on NULL. Every *out_err_msg below is untouched on success.
MR_EXPORT void mr_free_error_message(char* msg);

// Compiles MSL source.
MR_EXPORT MRStatus mr_compile_library(const char* msl_source, size_t msl_source_len,
                                      MRMathMode math_mode, MRLibrary** out_library,
                                      char** out_err_msg);
MR_EXPORT void mr_release_library(MRLibrary* library);

// Builds (and caches) the named kernel's pipeline.
MR_EXPORT MRStatus mr_get_pipeline(MRLibrary* library, const char* function_name,
                                   MRPipeline** out_pipeline, char** out_err_msg);
MR_EXPORT void mr_release_pipeline(MRPipeline* pipeline);

// Wraps `ptr` for mr_dispatch without copying: the range is rounded out to
// page boundaries (a Metal requirement) and dispatch binds at the true
// offset. Falls back to an owned copy, synced back by mr_buffer_flush_to,
// only if the rounded range would leave ptr's VM mapping. The caller keeps
// owning `ptr`.
MR_EXPORT MRStatus mr_wrap_buffer(void* ptr, size_t size_bytes, MRBuffer** out_buffer,
                                  char** out_err_msg);

// Copies the contents back to `ptr` after a dispatch. No-op for a zero-copy wrap.
MR_EXPORT MRStatus mr_buffer_flush_to(MRBuffer* buffer, char** out_err_msg);

// Releases the handle. Never frees the pointer passed to mr_wrap_buffer.
MR_EXPORT void mr_release_buffer(MRBuffer* buffer);

// One kernel launch. buffers/buffer_offsets and scalars/scalar_sizes are
// parallel arrays. threadgroup_x == 0 picks a default threadgroup.
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

// Encodes one launch and waits for it.
MR_EXPORT MRStatus mr_dispatch(const MRLaunchDesc* launch, char** out_err_msg);

// Encodes and submits one launch without blocking. On MR_OK, *out_batch owns
// the in-flight work: mr_batch_wait blocks on it, mr_release_batch frees the
// handle. Releasing without waiting leaves faults unreported.
MR_EXPORT MRStatus mr_dispatch_async(const MRLaunchDesc* launch, MRBatch** out_batch,
                                     char** out_err_msg);

// Blocks until the batch completes; a faulted command buffer is MR_ERROR_DISPATCH.
MR_EXPORT MRStatus mr_batch_wait(MRBatch* batch, char** out_err_msg);

MR_EXPORT void mr_release_batch(MRBatch* batch);

#ifdef __cplusplus
}  // extern "C"
#endif
