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

// Grid and threadgroup extents accept a bare int for the common 1D case.
using Extent = std::variant<size_t, std::vector<size_t>>;

class PyBuffer;

// Host extents, or a Buffer of three uint32 threadgroup counts the GPU
// reads at dispatch time (indirect).
using GridArg = std::variant<size_t, std::vector<size_t>, PyBuffer*>;

// A buffer binding, optionally at a byte offset into the allocation.
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

// A wrong-width scalar doesn't fail, the kernel just reads garbage; float64
// (numpy's default) gets its own message.
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

// Resolves the dtype a buffer's bytes should be labelled with. An explicit
// name reinterprets rather than converts (numpy's `.view` semantics), which
// is the way in for element types NumPy itself can't hand across the boundary:
// an ml_dtypes bfloat16 array exports through neither DLPack nor the buffer
// protocol, so it arrives here as its uint16 view and gets relabelled.
DType resolve_dtype(const HostArray& array, const std::optional<std::string>& override_name) {
    if (!override_name) return from_dlpack(array.dtype());

    DType requested = dtype_from_name(*override_name);
    if (array.dtype().lanes != 1) {
        throw std::invalid_argument("vector dtypes are not supported; pass a scalar dtype");
    }
    // The only invariant reinterpretation has to preserve is element width, so
    // that the shape and the byte count still agree.
    if (array.itemsize() != requested.itemsize()) {
        throw std::invalid_argument(
            "dtype '" + *override_name + "' is " + std::to_string(requested.itemsize()) +
            " bytes per element, but the array is " + std::to_string(array.itemsize()) +
            "; reinterpreting bytes cannot change element width, so .view() the array to a "
            "matching width first");
    }
    return requested;
}

// Tracks shape and dtype from construction so to_numpy() can hand back a view
// shaped and typed like the original array.
class PyBuffer {
   public:
    explicit PyBuffer(HostArray array, const std::optional<std::string>& dtype)
        : shape_(array.shape_ptr(), array.shape_ptr() + array.ndim()),
          dtype_(resolve_dtype(array, dtype)),
          buffer_(std::make_unique<Buffer>(runtime().device(), nbytes())) {
        std::memcpy(buffer_->contents(), array.data(), array.nbytes());
    }

    // No upload: for kernels that write every output element themselves (a
    // Pallas rollout's out_ref). Zero-initialized rather than left
    // uninitialized, cheap next to the upload path it replaces, and it rules
    // out reading back garbage from a kernel that misses an element.
    static PyBuffer zeros(std::vector<size_t> shape, const std::string& dtype) {
        return PyBuffer(std::move(shape), dtype_from_name(dtype), /*zero_fill=*/true);
    }

    // Uninitialized: no memset, for outputs a kernel overwrites entirely.
    static PyBuffer empty(std::vector<size_t> shape, const std::string& dtype) {
        return PyBuffer(std::move(shape), dtype_from_name(dtype), /*zero_fill=*/false);
    }

    // Refills the allocation in place. Reinterprets via `dtype` like the
    // constructor; shape may differ, byte count may not.
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

    // `owner` ties the returned array's lifetime to this PyBuffer so it can't
    // outlive the memory it views.
    // `dtype` reinterprets on the way out, mirroring the constructor: the
    // escape hatch for element types NumPy can't represent on its own.
    nb::ndarray<nb::numpy> to_numpy(const std::optional<std::string>& dtype) {
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

    // JAX and MLX both consume a DLPack capsule directly (jax.dlpack.from_dlpack,
    // mlx.core.array) without going through NumPy's dtype table, so they carry
    // bfloat16 (and anything else DType supports) through untouched.
    nb::ndarray<nb::jax> to_jax() { return as_ndarray<nb::jax>(); }
    nb::ndarray<nb::mlx> to_mlx() { return as_ndarray<nb::mlx>(); }

    nb::ndarray<> to_dlpack_ndarray() { return as_ndarray<>(); }

    Buffer* buffer() { return buffer_.get(); }

    const std::vector<size_t>& shape() const { return shape_; }
    const char* dtype() const { return dtype_name(dtype_); }
    size_t size() const { return checked_element_count(shape_); }
    size_t nbytes() const {
        size_t count = size();
        if (count != 0 && dtype_.itemsize() > SIZE_MAX / count) {
            throw std::invalid_argument("buffer shape is too large: byte size overflows");
        }
        return count * dtype_.itemsize();
    }

   private:
    template <typename... Framework>
    nb::ndarray<Framework...> as_ndarray() {
        return nb::ndarray<Framework...>(buffer_->contents(), shape_.size(), shape_.data(),
                                         nb::find(*this), nullptr, to_dlpack(dtype_));
    }

    PyBuffer(std::vector<size_t> shape, DType dtype, bool zero_fill)
        : shape_(std::move(shape)),
          dtype_(dtype),
          buffer_(std::make_unique<Buffer>(runtime().device(), nbytes())) {
        if (zero_fill) std::memset(buffer_->contents(), 0, nbytes());
    }

    std::vector<size_t> shape_;
    DType dtype_;
    std::unique_ptr<Buffer> buffer_;
};

// bool/int/float coerce to the declared MSL type at specialization; a numpy
// scalar pins an exact width that must match the declaration.
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
            // A value meant for a `ulong` constant past INT64_MAX needs the
            // unsigned path, or the cast fails before reaching coerce().
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
    // Sorted so {a,b} and {b,a} hit the same pipeline cache entry.
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
          pipeline_(library_->pipeline_for(function_name, parse_constants(constants),
                                           runtime().pipeline_archive())),
          function_name_(function_name) {
        for (auto [key, value] : constants) constants_[key] = value;
    }

    MathMode math_mode() const { return options_.math_mode; }
    const std::map<std::string, std::string>& defines() const { return options_.defines; }
    const nb::dict& constants() const { return constants_; }

    ComputePipeline& pipeline() { return *pipeline_; }
    const std::string& function_name() const { return function_name_; }

   private:
    // Declared before library_: library_'s initializer reads it, and members
    // are initialized in declaration order.
    CompileOptions options_;
    // Held, not just borrowed: the library cache evicts, and a pipeline built
    // from an evicted Library must not be the thing that discovers it.
    std::shared_ptr<Library> library_;
    // Shared with the library's pipeline cache.
    std::shared_ptr<ComputePipeline> pipeline_;
    std::string function_name_;
    nb::dict constants_;
};

// A Launch plus everything it points into. The Python objects behind the
// buffers are pinned here because dispatch runs with the GIL released: on a
// free-threaded interpreter another thread clearing the caller's list would
// otherwise be free to drop the last reference mid-kernel.
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
        prepared.launch.buffers.emplace_back(buffer->buffer(), offset);
        prepared.keepalive.push_back(nb::find(*buffer));
    }

    prepared.launch.scalars.reserve(prepared.scalars.size());
    for (size_t i = 0; i < prepared.scalars.size(); ++i) {
        const HostArray& scalar = prepared.scalars[i];
        check_scalar_dtype(scalar, i);
        prepared.launch.scalars.emplace_back(scalar.data(), scalar.nbytes());
    }
    return prepared;
}

void run(PyKernel& kernel, const GridArg& grid, const std::optional<Extent>& threadgroup,
         const std::vector<BufferArg>& buffers, std::vector<HostArray> scalars,
         const std::vector<size_t>& threadgroup_memory, size_t indirect_offset) {
    PreparedLaunch prepared = prepare(kernel, grid, threadgroup, buffers, std::move(scalars),
                                      threadgroup_memory, indirect_offset);
    // Released only around the blocking part, and only after every Python
    // object the launch touches is pinned above.
    nb::gil_scoped_release release;
    dispatch(runtime().queue(), prepared.launch);
}

// Several kernels in one command buffer: one commit and one GPU round-trip
// for the whole sequence.
class PyBatch {
   public:
    explicit PyBatch(bool concurrent)
        : batch_(std::make_unique<CommandBatch>(runtime().queue(), concurrent)) {}

    void add(PyKernel& kernel, const GridArg& grid, const std::optional<Extent>& threadgroup,
             const std::vector<BufferArg>& buffers, std::vector<HostArray> scalars,
             const std::vector<size_t>& threadgroup_memory, size_t indirect_offset) {
        PreparedLaunch prepared = prepare(kernel, grid, threadgroup, buffers, std::move(scalars),
                                          threadgroup_memory, indirect_offset);
        batch_->add(prepared.launch);
        // The batch is committed later, so its buffers have to stay alive
        // until then, not just until this call returns. Deduplicated: a
        // stepping loop re-adds the same kernel and buffers per launch.
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
        keepalive_.clear();
        pinned_.clear();
    }

    std::optional<double> gpu_time() const { return batch_->gpu_time(); }

   private:
    std::unique_ptr<CommandBatch> batch_;
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
    // Registration order matters: nanobind prepends translators and tries them
    // most-recent-first, so a derived C++ exception has to be registered after
    // its base or the base's translator would swallow it. FunctionNotFoundError
    // hangs off CompileError so one `except CompileError` catches both halves of
    // "the generated MSL is wrong".
    [[maybe_unused]] nb::object device_error = nb::exception<NoDeviceError>(m, "DeviceError");
    nb::object compile_error = nb::exception<MSLCompileError>(m, "CompileError");
    [[maybe_unused]] nb::object not_found =
        nb::exception<MSLFunctionNotFoundError>(m, "FunctionNotFoundError", compile_error);
    [[maybe_unused]] nb::object dispatch_error = nb::exception<DispatchError>(m, "DispatchError");

    m.def("device_name", []() { return runtime().device_name(); });
    m.def("device_info", &device_info);
    m.def("supported_dtypes", &supported_dtype_names);

    m.def("library_cache_size", []() { return runtime().library_cache_size(); });
    m.def("library_cache_limit", []() { return runtime().library_cache_limit(); });
    m.def(
        "set_library_cache_limit", [](size_t limit) { runtime().set_library_cache_limit(limit); },
        "limit"_a);
    m.def("clear_library_cache", []() { runtime().clear_library_cache(); });

    m.def(
        "set_pipeline_cache_dir",
        [](std::optional<std::string> path) { runtime().set_pipeline_cache_dir(path); }, "path"_a);
    m.def("pipeline_cache_dir", []() { return runtime().pipeline_cache_dir(); });
    m.def("save_pipeline_cache", []() { runtime().save_pipeline_cache(); });

    nb::class_<PyBuffer>(m, "Buffer")
        .def(nb::init<HostArray, const std::optional<std::string>&>(), "array"_a,
             "dtype"_a = nb::none())
        .def_static("zeros", &PyBuffer::zeros, "shape"_a, "dtype"_a = "float32")
        .def_static("empty", &PyBuffer::empty, "shape"_a, "dtype"_a = "float32")
        .def("copy_from", &PyBuffer::copy_from, "array"_a, "dtype"_a = nb::none())
        .def("to_numpy", &PyBuffer::to_numpy, "dtype"_a = nb::none())
        .def("to_jax", &PyBuffer::to_jax, nb::sig("def to_jax(self) -> jax.Array"))
        .def("to_mlx", &PyBuffer::to_mlx, nb::sig("def to_mlx(self) -> mlx.core.array"))
        .def(
            "__dlpack__",
            [](PyBuffer& b, nb::kwargs) {
                // nb::ndarray<> (no framework) casts directly to a raw DLPack capsule.
                // Unlike routing through to_numpy(), this never consults NumPy's dtype
                //  table, so bfloat16 (and anything else DType supports) exports fine.
                return b.to_dlpack_ndarray();
            },
            nb::sig("def __dlpack__(self, **kwargs) -> typing.Any"))
        .def("__dlpack_device__",
             // (kDLCPU, 0): unified memory, so the bytes are host-addressable.
             [](PyBuffer&) { return nb::make_tuple(1, 0); })
        .def_prop_ro("shape",
                     [](const PyBuffer& b) {
                         nb::list dims;
                         for (size_t d : b.shape()) dims.append(d);
                         return nb::tuple(dims);
                     })
        .def_prop_ro("dtype", &PyBuffer::dtype)
        .def_prop_ro("size", &PyBuffer::size)
        .def_prop_ro("nbytes", &PyBuffer::nbytes)
        .def("__len__",
             [](const PyBuffer& b) {
                 if (b.shape().empty()) throw nb::type_error("len() of unsized object");
                 return b.shape()[0];
             })
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
             "defines"_a = std::map<std::string, std::string>(), "constants"_a = nb::dict())
        .def_prop_ro("function_name", &PyKernel::function_name)
        .def_prop_ro("math_mode", &PyKernel::math_mode)
        .def_prop_ro("defines", &PyKernel::defines)
        .def_prop_ro("constants", &PyKernel::constants)
        .def_prop_ro("max_threads_per_threadgroup",
                     [](PyKernel& k) { return k.pipeline().max_threads_per_threadgroup(); })
        .def_prop_ro("thread_execution_width",
                     [](PyKernel& k) { return k.pipeline().thread_execution_width(); })
        .def_prop_ro("static_threadgroup_memory_length",
                     [](PyKernel& k) { return k.pipeline().static_threadgroup_memory_length(); })
        .def("__repr__",
             [](PyKernel& k) { return "Kernel(function_name='" + k.function_name() + "')"; });

    m.def("run", &run, nb::sig(kLaunchSignature), "kernel"_a, "grid"_a,
          "threadgroup"_a = nb::none(), "buffers"_a = std::vector<BufferArg>(),
          "scalars"_a = std::vector<HostArray>(), "threadgroup_memory"_a = std::vector<size_t>(),
          "indirect_offset"_a = 0);

    nb::class_<PyBatch>(m, "Batch")
        .def(nb::init<bool>(), "concurrent"_a = false)
        .def("add", &PyBatch::add, nb::sig(kAddSignature), "kernel"_a, "grid"_a,
             "threadgroup"_a = nb::none(), "buffers"_a = std::vector<BufferArg>(),
             "scalars"_a = std::vector<HostArray>(), "threadgroup_memory"_a = std::vector<size_t>(),
             "indirect_offset"_a = 0)
        .def("barrier", &PyBatch::barrier)
        .def("commit", &PyBatch::commit)
        .def("wait", &PyBatch::wait)
        .def_prop_ro("gpu_time", &PyBatch::gpu_time)
        .def(
            "__enter__", [](PyBatch& b) { return &b; }, nb::rv_policy::reference_internal,
            nb::sig("def __enter__(self) -> typing.Self"))
        .def(
            "__exit__",
            [](PyBatch& b, nb::handle exc_type, nb::handle, nb::handle) {
                // Nothing is committed if the body raised: the destructor
                // closes the encoder and drops the command buffer instead, so
                // a half-encoded sequence never reaches the GPU.
                if (exc_type.is_none()) b.wait();
            },
            nb::arg("exc_type").none(), nb::arg("exc_value").none(), nb::arg("traceback").none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"));
}
