#include "c_api.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
    size_t external_offset = 0;    // buffer.contents() + external_offset == external_ptr
    bool owns_copy = false;        // true: buffer is a private copy, flush_to must run
    size_t logical_size = 0;       // excludes the enclosing page padding
};

namespace {

void set_error(char** out_err_msg, const std::string& message) {
    if (!out_err_msg) return;
    *out_err_msg = (char*)std::malloc(message.size() + 1);
    if (!*out_err_msg) return;
    std::memcpy(*out_err_msg, message.c_str(), message.size() + 1);
}

// Maps exceptions to MRStatus.
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
    } catch (const AllocationError& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_ALLOCATION;
    } catch (const std::invalid_argument& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        set_error(out_err_msg, e.what());
        return MR_ERROR_UNKNOWN;
    }
}

// True if every byte of [start, start + length) lies in one mapped, r+w VM region.
bool region_is_mapped_rw(void* start, size_t length) {
    if (length == 0) return true;
    mach_vm_address_t queried = (mach_vm_address_t)(uintptr_t)start;
    mach_vm_address_t addr = queried;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name = MACH_PORT_NULL;
    kern_return_t kr =
        mach_vm_region(mach_task_self(), &addr, &region_size, VM_REGION_BASIC_INFO_64,
                       (vm_region_info_t)&info, &info_count, &object_name);
    if (object_name != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object_name);
    if (kr != KERN_SUCCESS) return false;
    // mach_vm_region moves `addr` forward when the query falls in a hole.
    if (addr > queried) return false;
    if (!(info.protection & VM_PROT_READ) || !(info.protection & VM_PROT_WRITE)) return false;
    mach_vm_size_t offset = queried - addr;
    return offset <= region_size && length <= region_size - offset;
}

}  // namespace

void mr_free_error_message(char* msg) { std::free(msg); }

MRStatus mr_compile_library(const char* msl_source, size_t msl_source_len, MRMathMode math_mode,
                            MRLibrary** out_library, char** out_err_msg) {
    if (out_library) *out_library = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!out_library) throw std::invalid_argument("mr_compile_library: out_library is null");
        if (!msl_source) throw std::invalid_argument("mr_compile_library: msl_source is null");
        CompileOptions options;
        switch (math_mode) {
            case MR_MATH_MODE_SAFE:
                options.math_mode = MathMode::Safe;
                break;
            case MR_MATH_MODE_RELAXED:
                options.math_mode = MathMode::Relaxed;
                break;
            case MR_MATH_MODE_FAST:
                options.math_mode = MathMode::Fast;
                break;
            default:
                throw std::invalid_argument("mr_compile_library: unknown math_mode");
        }
        auto* lib = new MRLibrary{
            Library(runtime().device(), std::string(msl_source, msl_source_len), options)};
        *out_library = lib;
    });
}

void mr_release_library(MRLibrary* library) { delete library; }

MRStatus mr_get_pipeline(MRLibrary* library, const char* function_name, MRPipeline** out_pipeline,
                         char** out_err_msg) {
    if (out_pipeline) *out_pipeline = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!out_pipeline) throw std::invalid_argument("mr_get_pipeline: out_pipeline is null");
        if (!library) throw std::invalid_argument("mr_get_pipeline: library is null");
        if (!function_name) throw std::invalid_argument("mr_get_pipeline: function_name is null");
        auto* pipeline = new MRPipeline{library->library.pipeline_for(function_name)};
        *out_pipeline = pipeline;
    });
}

void mr_release_pipeline(MRPipeline* pipeline) { delete pipeline; }

MRStatus mr_wrap_buffer(void* ptr, size_t size_bytes, MRBuffer** out_buffer, char** out_err_msg) {
    if (out_buffer) *out_buffer = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!out_buffer) throw std::invalid_argument("mr_wrap_buffer: out_buffer is null");
        if (!ptr && size_bytes > 0) {
            throw std::invalid_argument("mr_wrap_buffer: ptr is null for a nonzero size");
        }

        if (size_bytes > 0) {
            size_t page_size = (size_t)getpagesize();
            uintptr_t addr = (uintptr_t)ptr;
            uintptr_t page_start = addr & ~(uintptr_t)(page_size - 1);
            size_t offset = addr - page_start;
            constexpr size_t max_size = std::numeric_limits<size_t>::max();
            if (size_bytes > std::numeric_limits<uintptr_t>::max() - addr ||
                size_bytes > max_size - offset ||
                offset + size_bytes > max_size - (page_size - 1)) {
                throw std::invalid_argument("mr_wrap_buffer: memory range overflows");
            }
            size_t wrapped_length = ((offset + size_bytes + page_size - 1) / page_size) * page_size;
            if (wrapped_length > std::numeric_limits<uintptr_t>::max() - page_start) {
                throw std::invalid_argument("mr_wrap_buffer: page-rounded memory range overflows");
            }

            if (region_is_mapped_rw((void*)page_start, wrapped_length)) {
                auto* buf =
                    new MRBuffer{Buffer(runtime().device(), (void*)page_start, wrapped_length), ptr,
                                 offset, false, size_bytes};
                *out_buffer = buf;
                return;
            }
        }

        Buffer owned(runtime().device(), size_bytes);
        if (size_bytes > 0) std::memcpy(owned.contents(), ptr, size_bytes);
        auto* buf = new MRBuffer{std::move(owned), ptr, 0, true, size_bytes};
        *out_buffer = buf;
    });
}

MRStatus mr_buffer_flush_to(MRBuffer* buffer, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!buffer) throw std::invalid_argument("mr_buffer_flush_to: buffer is null");
        if (!buffer->owns_copy) return;
        if (buffer->buffer.size() > 0) {
            std::memcpy(buffer->external_ptr, buffer->buffer.contents(), buffer->buffer.size());
        }
    });
}

void mr_release_buffer(MRBuffer* buffer) { delete buffer; }

namespace {

// `who` names the entry point in error messages.
Launch build_launch(const MRLaunchDesc* launch_desc, const char* who) {
    if (!launch_desc) throw std::invalid_argument(std::string(who) + ": launch is null");
    if (!launch_desc->pipeline)
        throw std::invalid_argument(std::string(who) + ": launch has no pipeline");
    if (launch_desc->buffer_count && (!launch_desc->buffers || !launch_desc->buffer_offsets)) {
        throw std::invalid_argument(std::string(who) +
                                    ": buffers and buffer_offsets are required when buffer_count "
                                    "is nonzero");
    }
    if (launch_desc->scalar_count && (!launch_desc->scalars || !launch_desc->scalar_sizes)) {
        throw std::invalid_argument(std::string(who) +
                                    ": scalars and scalar_sizes are required when scalar_count is "
                                    "nonzero");
    }
    if (launch_desc->threadgroup_memory_count && !launch_desc->threadgroup_memory) {
        throw std::invalid_argument(std::string(who) +
                                    ": threadgroup_memory is required when its count is nonzero");
    }

    Launch launch;
    launch.pipeline = launch_desc->pipeline->pipeline.get();
    for (size_t i = 0; i < launch_desc->buffer_count; ++i) {
        if (!launch_desc->buffers[i]) {
            throw std::invalid_argument(std::string(who) + ": buffers[" + std::to_string(i) +
                                        "] is null");
        }
        const MRBuffer& buffer = *launch_desc->buffers[i];
        size_t offset = launch_desc->buffer_offsets[i];
        if ((offset >= buffer.logical_size && !(offset == 0 && buffer.logical_size == 0)) ||
            offset > std::numeric_limits<size_t>::max() - buffer.external_offset) {
            throw std::invalid_argument(std::string(who) + ": buffers[" + std::to_string(i) +
                                        "] offset " + std::to_string(offset) +
                                        " is out of bounds for a logical buffer of " +
                                        std::to_string(buffer.logical_size) + " bytes");
        }
        launch.buffers.push_back(
            {&launch_desc->buffers[i]->buffer, offset + buffer.external_offset});
    }
    for (size_t i = 0; i < launch_desc->scalar_count; ++i) {
        if (!launch_desc->scalars[i]) {
            throw std::invalid_argument(std::string(who) + ": scalars[" + std::to_string(i) +
                                        "] is null");
        }
        launch.scalars.push_back({launch_desc->scalars[i], launch_desc->scalar_sizes[i]});
    }
    if (launch_desc->threadgroup_memory_count) {
        launch.threadgroup_memory.assign(
            launch_desc->threadgroup_memory,
            launch_desc->threadgroup_memory + launch_desc->threadgroup_memory_count);
    }
    launch.grid = {launch_desc->grid_x, launch_desc->grid_y, launch_desc->grid_z};

    if (launch_desc->threadgroup_x == 0) {
        launch.threadgroup = launch.pipeline->default_threadgroup(launch.grid);
    } else {
        launch.threadgroup = {launch_desc->threadgroup_x, launch_desc->threadgroup_y,
                              launch_desc->threadgroup_z};
    }
    return launch;
}

}  // namespace

MRStatus mr_dispatch(const MRLaunchDesc* launch_desc, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        Launch launch = build_launch(launch_desc, "mr_dispatch");
        dispatch(runtime(), launch);
    });
}

struct MRBatch {
    explicit MRBatch(MetalRuntime& rt) : batch(rt) {}
    CommandBatch batch;
};

MRStatus mr_batch_wait(MRBatch* batch, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!batch) throw std::invalid_argument("mr_batch_wait: batch is null");
        batch->batch.wait();
    });
}

void mr_release_batch(MRBatch* batch) { delete batch; }

MRStatus mr_batch_create(MRBatch** out_batch, char** out_err_msg) {
    if (out_batch) *out_batch = nullptr;
    return mr_guard(out_err_msg, [&] {
        if (!out_batch) throw std::invalid_argument("mr_batch_create: out_batch is null");
        *out_batch = new MRBatch(runtime());
    });
}

MRStatus mr_batch_add(MRBatch* batch, const MRLaunchDesc* launch_desc, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!batch) throw std::invalid_argument("mr_batch_add: batch is null");
        Launch launch = build_launch(launch_desc, "mr_batch_add");
        batch->batch.add(launch);
    });
}

MRStatus mr_batch_commit(MRBatch* batch, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!batch) throw std::invalid_argument("mr_batch_commit: batch is null");
        batch->batch.commit();
    });
}
