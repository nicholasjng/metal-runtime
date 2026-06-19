#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

Dim3 to_dim3(const Extent& extent, const char* what) {
    Dim3 out;
    if (const size_t* n = std::get_if<size_t>(&extent)) {
        out.x = *n;
        return out;
    }
    const std::vector<size_t>& dims = std::get<std::vector<size_t>>(extent);
    if (dims.empty() || dims.size() > 3) {
        throw std::invalid_argument(std::string(what) + " must have 1 to 3 dimensions, got " +
                                    std::to_string(dims.size()));
    }
    out.x = dims[0];
    if (dims.size() > 1) out.y = dims[1];
    if (dims.size() > 2) out.z = dims[2];
    return out;
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
        return PyBuffer(std::move(shape), dtype_from_name(dtype));
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
    PyBuffer(std::vector<size_t> shape, DType dtype)
        : shape_(std::move(shape)),
          dtype_(dtype),
          buffer_(std::make_unique<Buffer>(runtime().device(), nbytes())) {
        std::memset(buffer_->contents(), 0, nbytes());
    }

    std::vector<size_t> shape_;
    DType dtype_;
    std::unique_ptr<Buffer> buffer_;
};

class PyKernel {
   public:
    PyKernel(const std::string& msl_source, const std::string& function_name, MathMode math_mode,
             const std::map<std::string, std::string>& defines)
        : options_{math_mode, defines},
          library_(runtime().library_for(msl_source, options_)),
          pipeline_(runtime().device(), *library_, function_name),
          function_name_(function_name) {}

    MathMode math_mode() const { return options_.math_mode; }
    const std::map<std::string, std::string>& defines() const { return options_.defines; }

    ComputePipeline& pipeline() { return pipeline_; }
    const std::string& function_name() const { return function_name_; }

   private:
    // Declared before library_: library_'s initializer reads it, and members
    // are initialized in declaration order.
    CompileOptions options_;
    // Held, not just borrowed: the library cache evicts, and a pipeline built
    // from an evicted Library must not be the thing that discovers it.
    std::shared_ptr<Library> library_;
    ComputePipeline pipeline_;
    std::string function_name_;
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

PreparedLaunch prepare(PyKernel& kernel, const Extent& grid,
                       const std::optional<Extent>& threadgroup,
                       const std::vector<PyBuffer*>& buffers, std::vector<HostArray> scalars,
                       const std::vector<size_t>& threadgroup_memory) {
    PreparedLaunch prepared;
    prepared.scalars = std::move(scalars);

    prepared.launch.pipeline = &kernel.pipeline();
    prepared.launch.grid = to_dim3(grid, "grid");
    prepared.launch.threadgroup = threadgroup
                                      ? to_dim3(*threadgroup, "threadgroup")
                                      : kernel.pipeline().default_threadgroup(prepared.launch.grid);
    prepared.launch.threadgroup_memory = threadgroup_memory;

    prepared.keepalive.push_back(nb::find(kernel));
    prepared.launch.buffers.reserve(buffers.size());
    for (PyBuffer* b : buffers) {
        if (!b) throw std::invalid_argument("buffers contains None");
        prepared.launch.buffers.push_back(b->buffer());
        prepared.keepalive.push_back(nb::find(*b));
    }

    prepared.launch.scalars.reserve(prepared.scalars.size());
    for (const HostArray& scalar : prepared.scalars) {
        prepared.launch.scalars.emplace_back(scalar.data(), scalar.nbytes());
    }
    return prepared;
}

void run(PyKernel& kernel, const Extent& grid, const std::optional<Extent>& threadgroup,
         const std::vector<PyBuffer*>& buffers, std::vector<HostArray> scalars,
         const std::vector<size_t>& threadgroup_memory) {
    PreparedLaunch prepared =
        prepare(kernel, grid, threadgroup, buffers, std::move(scalars), threadgroup_memory);
    // Released only around the blocking part, and only after every Python
    // object the launch touches is pinned above.
    nb::gil_scoped_release release;
    dispatch(runtime().queue(), prepared.launch);
}

// Several kernels in one command buffer: one commit and one GPU round-trip
// for the whole sequence.
class PyBatch {
   public:
    PyBatch() : batch_(std::make_unique<CommandBatch>(runtime().queue())) {}

    void add(PyKernel& kernel, const Extent& grid, const std::optional<Extent>& threadgroup,
             const std::vector<PyBuffer*>& buffers, std::vector<HostArray> scalars,
             const std::vector<size_t>& threadgroup_memory) {
        PreparedLaunch prepared =
            prepare(kernel, grid, threadgroup, buffers, std::move(scalars), threadgroup_memory);
        batch_->add(prepared.launch);
        // The batch is committed later, so its buffers have to stay alive
        // until then, not just until this call returns.
        for (nb::object& obj : prepared.keepalive) keepalive_.push_back(std::move(obj));
    }

    void wait() {
        {
            nb::gil_scoped_release release;
            batch_->wait();
        }
        keepalive_.clear();
    }

   private:
    std::unique_ptr<CommandBatch> batch_;
    std::vector<nb::object> keepalive_;
};

#define METAL_RUNTIME_LAUNCH_PARAMS                                                    \
    "kernel: Kernel, grid: int | Sequence[int], "                                      \
    "threadgroup: int | Sequence[int] | None = None, buffers: Sequence[Buffer] = [], " \
    "scalars: Sequence[numpy.ndarray | numpy.generic] = [], "                          \
    "threadgroup_memory: Sequence[int] = []"

constexpr const char* kLaunchSignature = "def run(" METAL_RUNTIME_LAUNCH_PARAMS ") -> None";
constexpr const char* kAddSignature = "def add(self, " METAL_RUNTIME_LAUNCH_PARAMS ") -> None";

nb::dict device_info() {
    MetalRuntime& rt = runtime();
    nb::dict info;
    info["name"] = rt.device_name();
    info["unified_memory"] = rt.has_unified_memory();
    info["recommended_max_working_set_size"] = rt.recommended_max_working_set_size();
    info["max_threads_per_threadgroup"] = rt.max_threads_per_threadgroup();
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

    nb::class_<PyBuffer>(m, "Buffer")
        .def(nb::init<HostArray, const std::optional<std::string>&>(), "array"_a,
             "dtype"_a = nb::none())
        .def_static("zeros", &PyBuffer::zeros, "shape"_a, "dtype"_a = "float32")
        .def("to_numpy", &PyBuffer::to_numpy, "dtype"_a = nb::none())
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
             [](const PyBuffer& b) { return b.shape().empty() ? size_t(1) : b.shape()[0]; })
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
        .str_value("RELAXED", MathMode::Relaxed, "relaxed")
        .str_value("FAST", MathMode::Fast, "fast",
                   "Metal's default: permits reassociation and flushes denormals.");

    nb::class_<PyKernel>(m, "Kernel")
        .def(nb::init<const std::string&, const std::string&, MathMode,
                      const std::map<std::string, std::string>&>(),
             "msl_source"_a, "function_name"_a, "math_mode"_a = MathMode::Fast,
             "defines"_a = std::map<std::string, std::string>())
        .def_prop_ro("function_name", &PyKernel::function_name)
        .def_prop_ro("math_mode", &PyKernel::math_mode)
        .def_prop_ro("defines", &PyKernel::defines)
        .def_prop_ro("max_threads_per_threadgroup",
                     [](PyKernel& k) { return k.pipeline().max_threads_per_threadgroup(); })
        .def_prop_ro("thread_execution_width",
                     [](PyKernel& k) { return k.pipeline().thread_execution_width(); })
        .def("__repr__",
             [](PyKernel& k) { return "Kernel(function_name='" + k.function_name() + "')"; });

    m.def("run", &run, nb::sig(kLaunchSignature), "kernel"_a, "grid"_a,
          "threadgroup"_a = nb::none(), "buffers"_a = std::vector<PyBuffer*>(),
          "scalars"_a = std::vector<HostArray>(), "threadgroup_memory"_a = std::vector<size_t>());

    nb::class_<PyBatch>(m, "Batch")
        .def(nb::init<>())
        .def("add", &PyBatch::add, nb::sig(kAddSignature), "kernel"_a, "grid"_a,
             "threadgroup"_a = nb::none(), "buffers"_a = std::vector<PyBuffer*>(),
             "scalars"_a = std::vector<HostArray>(), "threadgroup_memory"_a = std::vector<size_t>())
        .def("wait", &PyBatch::wait)
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
