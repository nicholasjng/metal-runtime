#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "buffer.h"
#include "dispatch.h"
#include "library.h"
#include "metal.h"
#include "runtime.h"

namespace nb = nanobind;

namespace {

std::string device_name() {
    AutoreleaseScope scope;
    return std::string(runtime().device()->name()->utf8String());
}

size_t element_count(const std::vector<size_t>& shape) {
    size_t count = 1;
    for (size_t dim : shape) count *= dim;
    return count;
}

// Tracks the shape from construction so to_numpy() can hand back a view
// shaped like the original array.
class PyBuffer {
   public:
    explicit PyBuffer(nb::ndarray<float, nb::c_contig> array)
        : shape_(array.shape_ptr(), array.shape_ptr() + array.ndim()),
          buffer_(std::make_unique<Buffer>(runtime().device(), array.size() * sizeof(float))) {
        std::memcpy(buffer_->contents(), array.data(), array.size() * sizeof(float));
    }

    // No upload: for kernels that write every output element themselves (a
    // Pallas rollout's out_ref). Zero-initialized rather than left
    // uninitialized, cheap next to the upload path it replaces, and it rules
    // out reading back garbage from a kernel that misses an element.
    static PyBuffer zeros(std::vector<size_t> shape) { return PyBuffer(std::move(shape)); }

    // `owner` ties the returned array's lifetime to this PyBuffer so it can't
    // outlive the memory it views. Named to_numpy, not numpy: a same-named
    // method shadows the numpy module in the generated .pyi stub's own class
    // scope, since type checkers resolve names statically.
    nb::ndarray<nb::numpy, float> to_numpy() {
        return nb::ndarray<nb::numpy, float>(static_cast<float*>(buffer_->contents()),
                                             shape_.size(), shape_.data(), nb::find(*this));
    }

    Buffer* buffer() { return buffer_.get(); }

   private:
    explicit PyBuffer(std::vector<size_t> shape)
        : shape_(std::move(shape)),
          buffer_(
              std::make_unique<Buffer>(runtime().device(), element_count(shape_) * sizeof(float))) {
        std::memset(buffer_->contents(), 0, element_count(shape_) * sizeof(float));
    }

    std::vector<size_t> shape_;
    std::unique_ptr<Buffer> buffer_;
};

class PyKernel {
   public:
    PyKernel(const std::string& msl_source, const std::string& function_name)
        : pipeline_(runtime().device(), runtime().library_for(msl_source), function_name) {}

    ComputePipeline& pipeline() { return pipeline_; }

   private:
    ComputePipeline pipeline_;
};

void run(PyKernel& kernel, size_t grid_size, size_t threadgroup_size,
         std::vector<PyBuffer*> buffers) {
    std::vector<Buffer*> raw_buffers;
    raw_buffers.reserve(buffers.size());
    for (PyBuffer* b : buffers) raw_buffers.push_back(b->buffer());
    dispatch(runtime().queue(), kernel.pipeline(), raw_buffers, grid_size, threadgroup_size);
}

}  // namespace

NB_MODULE(_core, m) {
    // The constructor's side effect is the registration; the temporary isn't
    // meant to be kept, unlike the lock-guard bug this check targets.
    nb::exception<MSLCompileError>(m, "CompileError");  // NOLINT(bugprone-unused-raii)

    m.def("device_name", &device_name);

    nb::class_<PyBuffer>(m, "Buffer")
        .def(nb::init<nb::ndarray<float, nb::c_contig>>(), nb::arg("array"))
        .def_static("zeros", &PyBuffer::zeros, nb::arg("shape"))
        .def("to_numpy", &PyBuffer::to_numpy);

    nb::class_<PyKernel>(m, "Kernel")
        .def(nb::init<const std::string&, const std::string&>(), nb::arg("msl_source"),
             nb::arg("function_name"));

    m.def("run", &run, nb::arg("kernel"), nb::arg("grid_size"), nb::arg("threadgroup_size"),
          nb::arg("buffers"), nb::call_guard<nb::gil_scoped_release>());
}
