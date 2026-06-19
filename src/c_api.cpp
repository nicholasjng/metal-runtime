#include "c_api.h"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "buffer.h"
#include "dispatch.h"
#include "library.h"
#include "metal.h"
#include "runtime.h"

struct MRLibrary {
    Library library;
};

struct MRPipeline {
    std::shared_ptr<ComputePipeline> pipeline;
};

struct MRBuffer {
    Buffer buffer;
    void* external_ptr = nullptr;  // set even on the copy path, for flush_to
};

namespace {

void set_error(char** out_err_msg, const std::string& message) {
    if (!out_err_msg) return;
    *out_err_msg = (char*)std::malloc(message.size() + 1);
    std::memcpy(*out_err_msg, message.c_str(), message.size() + 1);
}

// One AutoreleaseScope per call, one exception-to-MRStatus mapping.
// MSLFunctionNotFoundError must be caught before MSLCompileError: it
// derives from it.
template <typename Fn>
MRStatus mr_guard(char** out_err_msg, Fn&& fn) {
    AutoreleaseScope scope;
    try {
        fn();
        return MR_OK;
    } catch (const MSLFunctionNotFoundError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_FUNCTION_NOT_FOUND;
    } catch (const MSLCompileError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_COMPILE;
    } catch (const DispatchError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_DISPATCH;
    } catch (const NoDeviceError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_NO_DEVICE;
    } catch (const std::invalid_argument& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_UNKNOWN;
    }
}

}  // namespace

void mr_free_error_message(char* msg) { std::free(msg); }

MRStatus mr_compile_library(const char* msl_source, size_t msl_source_len, MRLibrary** out_library,
                            char** out_err_msg) {
    if (out_library) *out_library = nullptr;
    return mr_guard(out_err_msg, [&] {
        auto* lib =
            new MRLibrary{Library(runtime().device(), std::string(msl_source, msl_source_len))};
        *out_library = lib;
    });
}

void mr_release_library(MRLibrary* library) { delete library; }

MRStatus mr_get_pipeline(MRLibrary* library, const char* function_name, MRPipeline** out_pipeline,
                         char** out_err_msg) {
    if (out_pipeline) *out_pipeline = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!library) throw std::invalid_argument("mr_get_pipeline: library is null");
        auto* pipeline = new MRPipeline{library->library.pipeline_for(function_name)};
        *out_pipeline = pipeline;
    });
}

void mr_release_pipeline(MRPipeline* pipeline) { delete pipeline; }

MRStatus mr_wrap_buffer(void* ptr, size_t size_bytes, MRBuffer** out_buffer, char** out_err_msg) {
    if (out_buffer) *out_buffer = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!ptr && size_bytes > 0) {
            throw std::invalid_argument("mr_wrap_buffer: ptr is null for a nonzero size");
        }
        size_t page_size = (size_t)getpagesize();
        bool page_aligned =
            size_bytes > 0 && (uintptr_t)ptr % page_size == 0 && size_bytes % page_size == 0;
        if (page_aligned) {
            auto* buf = new MRBuffer{Buffer(runtime().device(), ptr, size_bytes), ptr};
            *out_buffer = buf;
        } else {
            Buffer owned(runtime().device(), size_bytes);
            if (size_bytes > 0) std::memcpy(owned.contents(), ptr, size_bytes);
            auto* buf = new MRBuffer{std::move(owned), ptr};
            *out_buffer = buf;
        }
    });
}

MRStatus mr_buffer_flush_to(MRBuffer* buffer, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!buffer) throw std::invalid_argument("mr_buffer_flush_to: buffer is null");
        // Zero-copy: contents() is already external_ptr, nothing to sync.
        if (buffer->buffer.contents() == buffer->external_ptr) return;
        if (buffer->buffer.size() > 0) {
            std::memcpy(buffer->external_ptr, buffer->buffer.contents(), buffer->buffer.size());
        }
    });
}

void mr_release_buffer(MRBuffer* buffer) { delete buffer; }

MRStatus mr_dispatch(const MRLaunchDesc* launch_desc, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!launch_desc) throw std::invalid_argument("mr_dispatch: launch is null");
        if (!launch_desc->pipeline)
            throw std::invalid_argument("mr_dispatch: launch has no pipeline");

        Launch launch;
        launch.pipeline = launch_desc->pipeline->pipeline.get();
        for (size_t i = 0; i < launch_desc->buffer_count; ++i) {
            if (!launch_desc->buffers[i]) {
                throw std::invalid_argument("mr_dispatch: buffers[" + std::to_string(i) +
                                            "] is null");
            }
            launch.buffers.emplace_back(&launch_desc->buffers[i]->buffer,
                                        launch_desc->buffer_offsets[i]);
        }
        for (size_t i = 0; i < launch_desc->scalar_count; ++i) {
            launch.scalars.emplace_back(launch_desc->scalars[i], launch_desc->scalar_sizes[i]);
        }
        launch.threadgroup_memory.assign(
            launch_desc->threadgroup_memory,
            launch_desc->threadgroup_memory + launch_desc->threadgroup_memory_count);
        launch.grid = {launch_desc->grid_x, launch_desc->grid_y, launch_desc->grid_z};

        if (launch_desc->threadgroup_x == 0) {
            launch.threadgroup = launch.pipeline->default_threadgroup(launch.grid);
        } else {
            launch.threadgroup = {launch_desc->threadgroup_x, launch_desc->threadgroup_y,
                                  launch_desc->threadgroup_z};
        }

        dispatch(runtime().queue(), launch);
    });
}
