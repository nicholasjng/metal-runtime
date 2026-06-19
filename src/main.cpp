// Prints the system default GPU: proves the metal-cpp fetch and Objective-C
// bridge work.
//
//   cmake -S . -B build -G Ninja && cmake --build build
//   ./build/metal_probe   ->   Metal device: Apple M1 Pro
#include <cstdio>
#include <exception>

#include "runtime.h"

int main() {
    try {
        MetalRuntime& rt = runtime();
        std::printf("Metal device: %s\n", rt.device_name().c_str());
        std::printf("Device command queue alive at %p\n", (void*)rt.queue());
        std::printf("Unified memory: %s\n", rt.has_unified_memory() ? "yes" : "no");
        std::printf("Max threads per threadgroup: %zu\n", rt.max_threads_per_threadgroup());
        std::printf("Non-uniform threadgroups: %s\n",
                    rt.supports_non_uniform_threadgroups() ? "yes" : "no");
        std::printf("same instance: %s\n", &rt == &runtime() ? "yes" : "no");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
