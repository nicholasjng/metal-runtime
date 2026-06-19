#include "c_api.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>
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
    size_t external_offset = 0;    // buffer.contents() + external_offset == external_ptr
    bool owns_copy = false;        // true: buffer is a private copy, flush_to must run
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

// True if every byte of [start, start + length) lies in one mapped,
// read+write VM region. Page-rounding a caller-supplied pointer can grow
// the wrapped range past what the caller actually owns; this is what
// keeps that growth from ever reaching into unmapped memory before
// handing the range to Metal.
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
    // mach_vm_region rounds `addr` up to the next mapped region when the
    // queried address itself falls in an unmapped hole; if it moved,
    // `start` was never mapped in the first place.
    if (addr > queried) return false;
    if (!(info.protection & VM_PROT_READ) || !(info.protection & VM_PROT_WRITE)) return false;
    return queried + (mach_vm_size_t)length <= addr + region_size;
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

        if (size_bytes > 0) {
            // Round [ptr, ptr + size_bytes) out to enclosing page boundaries:
            // newBufferWithBytesNoCopy (Buffer's external-ptr constructor)
            // requires both page-exact, which real allocations almost never are
            // on their own. The true data starts `offset` bytes into the wrapped
            // region; mr_dispatch adds that back in via external_offset, so callers
            // never see it. Falls back to an owned copy if the rounded range would
            // reach outside ptr's actual mapping.
            size_t page_size = (size_t)getpagesize();
            uintptr_t addr = (uintptr_t)ptr;
            uintptr_t page_start = addr & ~(uintptr_t)(page_size - 1);
            size_t offset = addr - page_start;
            size_t wrapped_length = ((offset + size_bytes + page_size - 1) / page_size) * page_size;

            if (region_is_mapped_rw((void*)page_start, wrapped_length)) {
                auto* buf =
                    new MRBuffer{Buffer(runtime().device(), (void*)page_start, wrapped_length), ptr,
                                 offset, false};
                *out_buffer = buf;
                return;
            }
        }

        Buffer owned(runtime().device(), size_bytes);
        if (size_bytes > 0) std::memcpy(owned.contents(), ptr, size_bytes);
        auto* buf = new MRBuffer{std::move(owned), ptr, 0, true};
        *out_buffer = buf;
    });
}

MRStatus mr_buffer_flush_to(MRBuffer* buffer, char** out_err_msg) {
    return mr_guard(out_err_msg, [&] {
        if (!buffer) throw std::invalid_argument("mr_buffer_flush_to: buffer is null");
        // Zero-copy (page-exact or page-rounded): nothing to sync back.
        if (!buffer->owns_copy) return;
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
            launch.buffers.emplace_back(
                &launch_desc->buffers[i]->buffer,
                launch_desc->buffer_offsets[i] + launch_desc->buffers[i]->external_offset);
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
