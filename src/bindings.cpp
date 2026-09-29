#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "buffer.h"
#include "dispatch.h"
#include "dtype.h"
#include "library.h"
#include "runtime.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using HostArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// A bare int is the 1D case.
using Extent = std::variant<size_t, std::vector<size_t>>;

class PyBuffer;

// Host extents, or a Buffer of three uint32 threadgroup counts (indirect).
using GridArg = std::variant<size_t, std::vector<size_t>, PyBuffer*>;

// A buffer binding, optionally at a byte offset.
using BufferArg = std::variant<PyBuffer*, std::pair<PyBuffer*, size_t>>;

Dim3 to_dim3(size_t n, const char*) {
    Dim3 out;
    out.x = n;
    return out;
}

Dim3 to_dim3(const std::vector<size_t>& dims, const char* what) {
    if (dims.empty() || dims.size() > 3) {
        throw std::invalid_argument(std::string(what) + " must have 1 to 3 dimensions, got " +
                                    std::to_string(dims.size()));
    }
    Dim3 out;
    out.x = dims[0];
    if (dims.size() > 1) out.y = dims[1];
    if (dims.size() > 2) out.z = dims[2];
    return out;
}

Dim3 to_dim3(const Extent& extent, const char* what) {
    if (const size_t* n = std::get_if<size_t>(&extent)) return to_dim3(*n, what);
    return to_dim3(std::get<std::vector<size_t>>(extent), what);
}

DType from_dlpack(nb::dlpack::dtype dt) {
    if (dt.lanes != 1) {
        throw std::invalid_argument("vector dtypes are not supported; pass a scalar dtype");
    }
    DType out{dt.code, dt.bits};
    if (!dtype_name(out)) {
        if (out.code == DType::Float && out.bits == 64) {
            throw std::invalid_argument(
                "dtype 'float64' is not supported: Metal has no double-precision type. "
                "Convert with .astype(numpy.float32) before uploading.");
        }
        throw std::invalid_argument("unsupported array dtype (DLPack code " +
                                    std::to_string(out.code) + ", " + std::to_string(out.bits) +
                                    " bits); supported: " + supported_dtype_names());
    }
    return out;
}

nb::dlpack::dtype to_dlpack(DType dt) { return nb::dlpack::dtype{dt.code, dt.bits, 1}; }

void check_scalar_dtype(const HostArray& scalar, size_t position) {
    nb::dlpack::dtype dt = scalar.dtype();
    if (dt.code == DType::Float && dt.bits == 64 && dt.lanes == 1) {
        throw std::invalid_argument(
            "scalars[" + std::to_string(position) +
            "] is a float64, numpy's default: Metal has no double-precision type, and a kernel "
            "expecting a 4-byte scalar would silently read half of it. Pass an explicit width, "
            "e.g. numpy.float32(value).");
    }
    try {
        from_dlpack(dt);
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument("scalars[" + std::to_string(position) + "]: " + e.what());
    }
}

size_t checked_element_count(const std::vector<size_t>& shape) {
    size_t count = 1;
    for (size_t dim : shape) {
        if (dim != 0 && count > SIZE_MAX / dim) {
            throw std::invalid_argument("buffer shape is too large: element count overflows");
        }
        count *= dim;
    }
    return count;
}

// An explicit dtype name reinterprets the bytes (`.view` semantics).
DType resolve_dtype(const HostArray& array, const std::optional<std::string>& override_name) {
    if (!override_name) return from_dlpack(array.dtype());

    DType requested = dtype_from_name(*override_name);
    if (array.dtype().lanes != 1) {
        throw std::invalid_argument("vector dtypes are not supported; pass a scalar dtype");
    }
    if (array.itemsize() != requested.itemsize()) {
        throw std::invalid_argument(
            "dtype '" + *override_name + "' is " + std::to_string(requested.itemsize()) +
            " bytes per element, but the array is " + std::to_string(array.itemsize()) +
            "; reinterpreting bytes cannot change element width, so .view() the array to a "
            "matching width first");
    }
    return requested;
}

class PyBuffer {
   public:
    explicit PyBuffer(HostArray array, const std::optional<std::string>& dtype)
        : PyBuffer(std::vector<size_t>(array.shape_ptr(), array.shape_ptr() + array.ndim()),
                   resolve_dtype(array, dtype), /*zero_fill=*/false) {
        std::memcpy(buffer_->contents(), array.data(), array.nbytes());
    }

    static PyBuffer zeros(std::vector<size_t> shape, const std::string& dtype) {
        return PyBuffer(std::move(shape), dtype_from_name(dtype), /*zero_fill=*/true);
    }

    static PyBuffer empty(std::vector<size_t> shape, const std::string& dtype) {
        return PyBuffer(std::move(shape), dtype_from_name(dtype), /*zero_fill=*/false);
    }

    // Shape may differ, byte count may not.
    void copy_from(const HostArray& array, const std::optional<std::string>& dtype) {
        DType incoming = resolve_dtype(array, dtype);
        if (incoming != dtype_) {
            throw std::invalid_argument(std::string("copy_from(): array resolves to dtype '") +
                                        dtype_name(incoming) + "', but this buffer holds '" +
                                        this->dtype() +
                                        "'; pass dtype=... to relabel, or match the array dtype");
        }
        if (array.nbytes() != nbytes()) {
            throw std::invalid_argument("copy_from(): array has " + std::to_string(array.nbytes()) +
                                        " bytes, but this buffer holds " +
                                        std::to_string(nbytes()) +
                                        "; copy_from cannot resize an allocation");
        }
        std::memcpy(buffer_->contents(), array.data(), array.nbytes());
    }

    nb::ndarray<nb::numpy> to_numpy(const std::optional<std::string>& dtype) const {
        DType out = dtype_;
        if (dtype) {
            out = dtype_from_name(*dtype);
            if (out.itemsize() != dtype_.itemsize()) {
                throw std::invalid_argument("dtype '" + *dtype + "' is " +
                                            std::to_string(out.itemsize()) +
                                            " bytes per element, but this buffer holds " +
                                            std::to_string(dtype_.itemsize()) + "-byte elements");
            }
        }
        if (out.code == DType::Bfloat) {
            throw nb::type_error(
                "to_numpy(): NumPy has no native bfloat16 dtype. Read the bytes back with "
                "to_numpy(dtype='uint16'), which ml_dtypes can .view() as bfloat16.");
        }
        return nb::ndarray<nb::numpy>(buffer_->contents(), shape_.size(), shape_.data(),
                                      nb::find(*this), nullptr, to_dlpack(out));
    }

    // A raw capsule skips NumPy's dtype table, so bfloat16 exports too.
    nb::ndarray<> to_dlpack_ndarray() const {
        return nb::ndarray<>(buffer_->contents(), shape_.size(), shape_.data(), nb::find(*this),
                             nullptr, to_dlpack(dtype_));
    }

    Buffer* buffer() { return buffer_.get(); }

    const std::vector<size_t>& shape() const { return shape_; }
    const char* dtype() const { return dtype_name(dtype_); }
    size_t size() const { return size_; }
    size_t nbytes() const { return nbytes_; }

   private:
    PyBuffer(std::vector<size_t> shape, DType dtype, bool zero_fill)
        : shape_(std::move(shape)),
          dtype_(dtype),
          size_(checked_element_count(shape_)),
          nbytes_(checked_byte_count(size_, dtype_)),
          buffer_(std::make_unique<Buffer>(runtime().device(), nbytes_)) {
        if (zero_fill) std::memset(buffer_->contents(), 0, nbytes_);
    }

    static size_t checked_byte_count(size_t count, DType dtype) {
        if (count != 0 && dtype.itemsize() > SIZE_MAX / count) {
            throw std::invalid_argument("buffer shape is too large: byte size overflows");
        }
        return count * dtype.itemsize();
    }

    std::vector<size_t> shape_;
    DType dtype_;
    size_t size_;
    size_t nbytes_;
    std::unique_ptr<Buffer> buffer_;
};

// Python scalars coerce to the declared MSL type; a numpy scalar must match it exactly.
FunctionConstants parse_constants(const nb::dict& constants) {
    FunctionConstants out;
    out.reserve(constants.size());
    for (auto [key, value] : constants) {
        FunctionConstant c;
        c.name = nb::cast<std::string>(key);
        // bool before int: Python bools are ints.
        if (nb::isinstance<nb::bool_>(value)) {
            c.kind = FunctionConstant::Kind::Bool;
            c.bool_value = nb::cast<bool>(value);
        } else if (nb::isinstance<nb::int_>(value)) {
            c.kind = FunctionConstant::Kind::Int;
            long long v;
            if (nb::try_cast<long long>(value, v)) {
                c.int_value = v;
            } else if (nb::try_cast<unsigned long long>(value, c.uint_value)) {
                c.int_is_wide_unsigned = true;
            } else {
                throw std::invalid_argument("function constant '" + c.name +
                                            "' is out of range for a 64-bit integer");
            }
        } else if (nb::isinstance<nb::float_>(value)) {
            c.kind = FunctionConstant::Kind::Float;
            c.float_value = nb::cast<double>(value);
        } else {
            HostArray scalar;
            if (!nb::try_cast<HostArray>(value, scalar) || scalar.size() != 1) {
                throw std::invalid_argument(
                    "function constant '" + c.name +
                    "' must be a bool, int, float, or a single numpy scalar");
            }
            c.kind = FunctionConstant::Kind::Exact;
            c.dtype = from_dlpack(scalar.dtype());
            std::memcpy(c.value.data(), scalar.data(), c.dtype.itemsize());
        }
        out.push_back(std::move(c));
    }
    // Sorted so the cache key is order-independent.
    std::sort(out.begin(), out.end(),
              [](const FunctionConstant& a, const FunctionConstant& b) { return a.name < b.name; });
    return out;
}

class PyKernel {
   public:
    PyKernel(const std::string& msl_source, const std::string& function_name, MathMode math_mode,
             const std::map<std::string, std::string>& defines, const nb::dict& constants)
        : options_{math_mode, defines},
          library_(runtime().library_for(msl_source, options_)),
          pipeline_(library_->pipeline_for(function_name, parse_constants(constants))),
          function_name_(function_name) {
        for (auto [key, value] : constants) constants_[key] = value;
    }

    MathMode math_mode() const { return options_.math_mode; }
    const std::map<std::string, std::string>& defines() const { return options_.defines; }
    const nb::dict& constants() const { return constants_; }

    ComputePipeline& pipeline() { return *pipeline_; }
    const std::string& function_name() const { return function_name_; }

   private:
    CompileOptions options_;            // initialized before library_ reads it
    std::shared_ptr<Library> library_;  // held: the library cache evicts
    std::shared_ptr<ComputePipeline> pipeline_;
    std::string function_name_;
    nb::dict constants_;
};

// A Launch plus the Python objects it points into, pinned because dispatch
// runs with the GIL released.
struct PreparedLaunch {
    Launch launch;
    std::vector<nb::object> keepalive;
    std::vector<HostArray> scalars;
};

PreparedLaunch prepare(PyKernel& kernel, const GridArg& grid,
                       const std::optional<Extent>& threadgroup,
                       const std::vector<BufferArg>& buffers, std::vector<HostArray> scalars,
                       const std::vector<size_t>& threadgroup_memory, size_t indirect_offset) {
    PreparedLaunch prepared;
    prepared.scalars = std::move(scalars);

    prepared.launch.pipeline = &kernel.pipeline();
    prepared.launch.threadgroup_memory = threadgroup_memory;
    prepared.keepalive.push_back(nb::find(kernel));

    if (PyBuffer* const* indirect = std::get_if<PyBuffer*>(&grid)) {
        if (!*indirect) throw std::invalid_argument("grid is None");
        if (!threadgroup) {
            throw std::invalid_argument(
                "an indirect grid needs an explicit threadgroup size: the buffer holds "
                "threadgroup counts, so there is no thread total to derive one from");
        }
        prepared.launch.indirect_grid = (*indirect)->buffer();
        prepared.launch.indirect_offset = indirect_offset;
        prepared.launch.threadgroup = to_dim3(*threadgroup, "threadgroup");
        prepared.keepalive.push_back(nb::find(**indirect));
    } else {
        if (indirect_offset != 0) {
            throw std::invalid_argument(
                "indirect_offset is only meaningful when grid is a Buffer of threadgroup counts");
        }
        if (const size_t* n = std::get_if<size_t>(&grid)) {
            prepared.launch.grid = to_dim3(*n, "grid");
        } else {
            prepared.launch.grid = to_dim3(std::get<std::vector<size_t>>(grid), "grid");
        }
        prepared.launch.threadgroup =
            threadgroup ? to_dim3(*threadgroup, "threadgroup")
                        : kernel.pipeline().default_threadgroup(prepared.launch.grid);
    }

    prepared.launch.buffers.reserve(buffers.size());
    for (const BufferArg& arg : buffers) {
        PyBuffer* buffer = nullptr;
        size_t offset = 0;
        if (PyBuffer* const* plain = std::get_if<PyBuffer*>(&arg)) {
            buffer = *plain;
        } else {
            const auto& [b, o] = std::get<std::pair<PyBuffer*, size_t>>(arg);
            buffer = b;
            offset = o;
        }
        if (!buffer) throw std::invalid_argument("buffers contains None");
        prepared.launch.buffers.push_back({buffer->buffer(), offset});
        prepared.keepalive.push_back(nb::find(*buffer));
    }

    prepared.launch.scalars.reserve(prepared.scalars.size());
    for (size_t i = 0; i < prepared.scalars.size(); ++i) {
        const HostArray& scalar = prepared.scalars[i];
        check_scalar_dtype(scalar, i);
        prepared.launch.scalars.push_back({scalar.data(), scalar.nbytes()});
    }
    return prepared;
}

void run(PyKernel& kernel, const GridArg& grid, const std::optional<Extent>& threadgroup,
         const std::vector<BufferArg>& buffers, std::vector<HostArray> scalars,
         const std::vector<size_t>& threadgroup_memory, size_t indirect_offset) {
    PreparedLaunch prepared = prepare(kernel, grid, threadgroup, buffers, std::move(scalars),
                                      threadgroup_memory, indirect_offset);
    nb::gil_scoped_release release;
    dispatch(runtime(), prepared.launch);
}

class PyBatch {
   public:
    explicit PyBatch(bool concurrent)
        : batch_(std::make_unique<CommandBatch>(runtime(), concurrent)) {}

    void add(PyKernel& kernel, const GridArg& grid, const std::optional<Extent>& threadgroup,
             const std::vector<BufferArg>& buffers, std::vector<HostArray> scalars,
             const std::vector<size_t>& threadgroup_memory, size_t indirect_offset) {
        PreparedLaunch prepared = prepare(kernel, grid, threadgroup, buffers, std::move(scalars),
                                          threadgroup_memory, indirect_offset);
        std::lock_guard<std::mutex> lock(keepalive_mutex_);
        batch_->add(prepared.launch);
        // Pinned until wait(); deduplicated for stepping loops.
        for (nb::object& obj : prepared.keepalive) {
            if (pinned_.insert(obj.ptr()).second) keepalive_.push_back(std::move(obj));
        }
    }

    void barrier() { batch_->barrier(); }

    void commit() {
        nb::gil_scoped_release release;
        batch_->commit();
    }

    void wait() {
        {
            nb::gil_scoped_release release;
            batch_->wait();
        }
        // Decref outside the lock: finalizers may re-enter this batch.
        std::vector<nb::object> released;
        {
            std::lock_guard<std::mutex> lock(keepalive_mutex_);
            released.swap(keepalive_);
            pinned_.clear();
        }
    }

    std::optional<double> gpu_time() const { return batch_->gpu_time(); }

   private:
    std::unique_ptr<CommandBatch> batch_;
    std::mutex keepalive_mutex_;
    std::vector<nb::object> keepalive_;
    std::unordered_set<PyObject*> pinned_;
};

#define METAL_RUNTIME_LAUNCH_PARAMS                           \
    "kernel: Kernel, grid: int | Sequence[int] | Buffer, "    \
    "threadgroup: int | Sequence[int] | None = None, "        \
    "buffers: Sequence[Buffer | tuple[Buffer, int]] = [], "   \
    "scalars: Sequence[numpy.ndarray | numpy.generic] = [], " \
    "threadgroup_memory: Sequence[int] = [], indirect_offset: int = 0"

constexpr const char* kLaunchSignature = "def run(" METAL_RUNTIME_LAUNCH_PARAMS ") -> None";
constexpr const char* kAddSignature = "def add(self, " METAL_RUNTIME_LAUNCH_PARAMS ") -> None";

nb::dict device_info() {
    MetalRuntime& rt = runtime();
    nb::dict info;
    info["name"] = rt.device_name();
    info["unified_memory"] = rt.has_unified_memory();
    info["recommended_max_working_set_size"] = rt.recommended_max_working_set_size();
    info["max_threads_per_threadgroup"] = rt.max_threads_per_threadgroup();
    info["max_threadgroup_memory_length"] = rt.max_threadgroup_memory_length();
    info["max_buffer_length"] = rt.max_buffer_length();
    info["supports_non_uniform_threadgroups"] = rt.supports_non_uniform_threadgroups();
    return info;
}

}  // namespace

NB_MODULE(_core, m) {
    // nanobind tries translators most-recent-first, so a derived exception
    // must be registered after its base.
    [[maybe_unused]] nb::object device_error = nb::exception<NoDeviceError>(m, "DeviceError");
    nb::object compile_error = nb::exception<MSLCompileError>(m, "CompileError");
    [[maybe_unused]] nb::object not_found =
        nb::exception<MSLFunctionNotFoundError>(m, "FunctionNotFoundError", compile_error);
    [[maybe_unused]] nb::object pipeline_build_error =
        nb::exception<PipelineBuildError>(m, "PipelineBuildError", compile_error);
    [[maybe_unused]] nb::object dispatch_error = nb::exception<DispatchError>(m, "DispatchError");
    [[maybe_unused]] nb::object capture_error = nb::exception<CaptureError>(m, "CaptureError");
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const AllocationError& e) {
            PyErr_SetString(PyExc_MemoryError, e.what());
        }
    });

    m.def(
        "start_capture", [](const std::string& path) { runtime().start_capture(path); }, "path"_a,
        R"doc(
Start capturing command buffers submitted to this runtime's Metal device.

Parameters
----------
path : str
    Destination path for the .gputrace document.

Raises
------
CaptureError
    Capture is already active, unsupported, or could not be started.
)doc");
    m.def(
        "stop_capture", []() { runtime().stop_capture(); },
        "Stop the active Metal trace capture and write its GPU trace document.");
    m.def(
        "is_capturing", []() { return runtime().is_capturing(); },
        "Whether a Metal trace capture is currently active in this process.");

    m.def(
        "device_name", []() { return runtime().device_name(); },
        R"doc(
Name of the default Metal device.

Returns
-------
str
)doc");
    m.def("device_info", &device_info,
          R"doc(
Device capabilities and limits.

Returns
-------
dict
    Keys: name, unified_memory, recommended_max_working_set_size,
    max_threads_per_threadgroup, max_threadgroup_memory_length,
    max_buffer_length, supports_non_uniform_threadgroups.
)doc");
    m.def("supported_dtypes", &supported_dtype_names,
          R"doc(
Comma-separated list of dtype names Buffer accepts.

Returns
-------
str
)doc");

    m.def(
        "library_cache_size", []() { return runtime().library_cache_size(); },
        R"doc(
Number of compiled MSL libraries currently cached.

Returns
-------
int
)doc");
    m.def(
        "library_cache_limit", []() { return runtime().library_cache_limit(); },
        R"doc(
Current cap on cached libraries.

Returns
-------
int
    0 means unlimited.
)doc");
    m.def(
        "set_library_cache_limit", [](size_t limit) { runtime().set_library_cache_limit(limit); },
        "limit"_a,
        R"doc(
Cap the library cache size, evicting least-recently-used entries.

Parameters
----------
limit : int
    0 disables eviction.
)doc");
    m.def(
        "clear_library_cache", []() { runtime().clear_library_cache(); },
        "Drop every cached library.");

    nb::class_<PyBuffer>(m, "Buffer")
        .def(nb::init<HostArray, const std::optional<std::string>&>(), "array"_a,
             "dtype"_a = nb::none(),
             R"doc(
Upload a NumPy array into a new device buffer.

Parameters
----------
array : numpy.ndarray
    C-contiguous, any dtype Metal can address; float64 is rejected.
dtype : str, optional
    Relabels the array's bytes rather than converting them (`.view`
    semantics). Element width must match.
)doc")
        .def_static("zeros", &PyBuffer::zeros, "shape"_a, "dtype"_a = "float32",
                    R"doc(
Allocate a zero-filled buffer without an upload.

Parameters
----------
shape : Sequence[int]
dtype : str, optional
    Defaults to "float32".

Returns
-------
Buffer
)doc")
        .def_static("empty", &PyBuffer::empty, "shape"_a, "dtype"_a = "float32",
                    R"doc(
Allocate an uninitialized buffer: no upload, no zero-fill.

Parameters
----------
shape : Sequence[int]
dtype : str, optional
    Defaults to "float32".

Returns
-------
Buffer
)doc")
        .def("copy_from", &PyBuffer::copy_from, "array"_a, "dtype"_a = nb::none(),
             R"doc(
Refill this buffer's allocation in place.

Parameters
----------
array : numpy.ndarray
    Byte count must match this buffer's; shape may differ.
dtype : str, optional
    Relabels rather than converts, like the constructor.

Raises
------
ValueError
    Byte count or resolved dtype doesn't match this buffer.
)doc")
        .def("to_numpy", &PyBuffer::to_numpy, "dtype"_a = nb::none(),
             R"doc(
A live NumPy view of this buffer's memory. Not a copy.

Parameters
----------
dtype : str, optional
    Relabels the bytes on the way out; element width must match.

Returns
-------
numpy.ndarray

Raises
------
TypeError
    dtype is bfloat16, which NumPy has no native dtype for.
)doc")
        .def(
            "__dlpack__",
            [](PyBuffer& b, nb::kwargs kwargs) {
                for (auto [key, value] : kwargs) {
                    const std::string name = nb::cast<std::string>(key);
                    if (name == "copy") {
                        if (!value.is_none() && !nb::isinstance<nb::bool_>(value)) {
                            throw nb::type_error(
                                "Buffer.__dlpack__(copy=...) expects bool or None");
                        }
                        if (!value.is_none() && nb::cast<bool>(value)) {
                            throw nb::type_error(
                                "Buffer.__dlpack__(copy=True) is unsupported; use "
                                "Buffer.to_numpy() "
                                "and copy the returned array");
                        }
                    } else if (name == "stream") {
                        if (!value.is_none()) {
                            throw nb::type_error(
                                "Buffer.__dlpack__ only supports stream=None for CPU-accessible "
                                "Metal shared memory");
                        }
                    } else if (name == "dl_device") {
                        std::pair<int, int> device{1, 0};
                        if (!value.is_none() && !nb::try_cast(value, device)) {
                            throw nb::type_error(
                                "Buffer.__dlpack__(dl_device=...) expects a tuple");
                        }
                        if (device != std::pair<int, int>{1, 0}) {
                            throw nb::type_error(
                                "Buffer.__dlpack__ only supports dl_device=(1, 0) (CPU)");
                        }
                    } else if (name == "max_version") {
                        if (!value.is_none()) {
                            throw nb::type_error(
                                "Buffer.__dlpack__ does not support versioned DLPack capsules");
                        }
                    } else {
                        const std::string message =
                            "Buffer.__dlpack__ got unsupported keyword '" + name + "'";
                        throw nb::type_error(message.c_str());
                    }
                }
                return b.to_dlpack_ndarray();
            },
            nb::sig("def __dlpack__(self, **kwargs) -> typing.Any"),
            "DLPack capsule for this buffer's memory. Zero-copy. Supports copy=None/False, "
            "stream=None, dl_device=(1, 0), and max_version=None. Unsupported protocol "
            "options are rejected instead of ignored.")
        .def(
            "__dlpack_device__", [](PyBuffer&) { return nb::make_tuple(1, 0); },
            "DLPack device tuple: (kDLCPU, 0), since Metal's shared storage is "
            "host-addressable.")
        .def_prop_ro(
            "shape",
            [](const PyBuffer& b) {
                nb::list dims;
                for (size_t d : b.shape()) dims.append(d);
                return nb::tuple(dims);
            },
            "Array shape.")
        .def_prop_ro("dtype", &PyBuffer::dtype, "dtype name, e.g. 'float32'.")
        .def_prop_ro("size", &PyBuffer::size, "Element count.")
        .def_prop_ro("nbytes", &PyBuffer::nbytes, "Byte count.")
        .def("__repr__", [](const PyBuffer& b) {
            std::string out = "Buffer(shape=(";
            for (size_t i = 0; i < b.shape().size(); ++i) {
                if (i) out += ", ";
                out += std::to_string(b.shape()[i]);
            }
            if (b.shape().size() == 1) out += ",";
            out += "), dtype='";
            out += b.dtype();
            out += "')";
            return out;
        });

    nb::enum_<MathMode>(m, "MathMode", nb::is_str(),
                        "How much freedom the Metal compiler has to rewrite floating-point "
                        "arithmetic.")
        .str_value("SAFE", MathMode::Safe, "safe",
                   "IEEE semantics. Required for compensated summation and double-single "
                   "arithmetic, whose error terms FAST is free to fold away.")
        .str_value("RELAXED", MathMode::Relaxed, "relaxed",
                   "Permits reassociation like FAST (so it also deletes compensated "
                   "arithmetic), but keeps infinities and NaNs well-defined instead of "
                   "assuming they never occur.")
        .str_value("FAST", MathMode::Fast, "fast",
                   "Metal's default: permits reassociation and flushes denormals.");

    nb::class_<PyKernel>(m, "Kernel")
        .def(nb::init<const std::string&, const std::string&, MathMode,
                      const std::map<std::string, std::string>&, const nb::dict&>(),
             "msl_source"_a, "function_name"_a, "math_mode"_a = MathMode::Fast,
             "defines"_a = std::map<std::string, std::string>(), "constants"_a = nb::dict(),
             R"doc(
Compile MSL source into a dispatchable kernel.

Parameters
----------
msl_source : str
function_name : str
    Entry point within `msl_source`.
math_mode : MathMode, optional
    Defaults to FAST.
defines : Mapping[str, str], optional
    Preprocessor macros, part of the library cache key.
constants : dict, optional
    Function constant values, part of the pipeline cache key.

Raises
------
CompileError
    msl_source doesn't compile.
FunctionNotFoundError
    function_name isn't in msl_source.
)doc")
        .def_prop_ro("function_name", &PyKernel::function_name, "Entry point name.")
        .def_prop_ro("math_mode", &PyKernel::math_mode, "Compiled math mode.")
        .def_prop_ro("defines", &PyKernel::defines,
                     "Preprocessor macros this kernel compiled with.")
        .def_prop_ro("constants", &PyKernel::constants,
                     "Function constant values this kernel was specialized with.")
        .def_prop_ro(
            "max_threads_per_threadgroup",
            [](PyKernel& k) { return k.pipeline().max_threads_per_threadgroup(); },
            "This kernel's own ceiling, which can be below the device maximum.")
        .def_prop_ro(
            "thread_execution_width",
            [](PyKernel& k) { return k.pipeline().thread_execution_width(); },
            "SIMD width for this kernel.")
        .def_prop_ro(
            "static_threadgroup_memory_length",
            [](PyKernel& k) { return k.pipeline().static_threadgroup_memory_length(); },
            "Bytes of threadgroup memory the kernel itself declares, before any "
            "dynamic threadgroup_memory passed to run().")
        .def("__repr__",
             [](PyKernel& k) { return "Kernel(function_name='" + k.function_name() + "')"; });

    m.def("run", &run, nb::sig(kLaunchSignature), "kernel"_a, "grid"_a,
          "threadgroup"_a = nb::none(), "buffers"_a = std::vector<BufferArg>(),
          "scalars"_a = std::vector<HostArray>(), "threadgroup_memory"_a = std::vector<size_t>(),
          "indirect_offset"_a = 0,
          R"doc(
Dispatch one kernel launch and block until it completes.

Parameters
----------
kernel : Kernel
grid : int or Sequence[int] or Buffer
    Thread count per dimension, or a Buffer of three uint32
    threadgroup counts for an indirect dispatch.
threadgroup : int or Sequence[int], optional
    Explicit threadgroup size. None lets the runtime choose.
buffers : Sequence[Buffer or tuple[Buffer, int]], optional
    Bind in order; a (Buffer, offset) tuple binds at a byte offset
    into that buffer's allocation.
scalars : Sequence[numpy.ndarray or numpy.generic], optional
    Bound inline with setBytes, after buffers.
threadgroup_memory : Sequence[int], optional
    Byte size for each [[threadgroup(i)]] allocation.
indirect_offset : int, optional
    Byte offset into `grid` when it is a Buffer.

Raises
------
ValueError
    The launch doesn't satisfy the kernel's own binding reflection.
DispatchError
    The GPU aborted the command buffer.
)doc");

    nb::class_<PyBatch>(m, "Batch")
        .def(nb::init<bool>(), "concurrent"_a = false,
             R"doc(
Encode several launches into one command buffer.

Parameters
----------
concurrent : bool, optional
    Lets independent launches overlap on the GPU; ordering then only
    exists across barrier(). Defaults to False (serial).
)doc")
        .def("add", &PyBatch::add, nb::sig(kAddSignature), "kernel"_a, "grid"_a,
             "threadgroup"_a = nb::none(), "buffers"_a = std::vector<BufferArg>(),
             "scalars"_a = std::vector<HostArray>(), "threadgroup_memory"_a = std::vector<size_t>(),
             "indirect_offset"_a = 0,
             R"doc(
Encode one launch into this batch. Same parameters as run().

Raises
------
DispatchError
    This batch has already been committed.
)doc")
        .def("barrier", &PyBatch::barrier,
             "Order buffer writes across a concurrent batch. Meaningless on a "
             "serial (default) batch.")
        .def("commit", &PyBatch::commit, "Close encoding and submit without blocking.")
        .def("wait", &PyBatch::wait,
             R"doc(
Commit if needed, then block until the GPU is done.

Raises
------
DispatchError
    The command buffer faulted.
)doc")
        .def_prop_ro("gpu_time", &PyBatch::gpu_time,
                     "Device-side execution seconds for the whole batch, set by wait().")
        .def(
            "__enter__", [](PyBatch& b) { return &b; }, nb::rv_policy::reference_internal,
            nb::sig("def __enter__(self) -> typing.Self"), "Returns self.")
        .def(
            "__exit__",
            [](PyBatch& b, nb::handle exc_type, nb::handle, nb::handle) {
                // If the body raised, the destructor discards the batch uncommitted.
                if (exc_type.is_none()) b.wait();
            },
            nb::arg("exc_type").none(), nb::arg("exc_value").none(), nb::arg("traceback").none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"),
            "Waits on the batch if the body didn't raise; otherwise discards it "
            "without committing.");
}
