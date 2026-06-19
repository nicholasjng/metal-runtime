// Minimal metal-cpp smoke test: print the system default GPU.
//
// This is the seed the runtime grows from (device/queue/buffers/MSL library
// cache + nanobind bindings — see ROADMAP.md). For now it just proves the
// metal-cpp fetch and the Objective-C bridge are wired up:
//
//   cmake -S . -B build -G Ninja && cmake --build build
//   ./build/metal_probe   ->   Metal device: Apple M1 Pro
//
#include <cstdio>

#include "metal.h"
#include "runtime.h"

int main() {
    MetalRuntime* rt = &runtime();
    if (!rt->device()) {
        std::fprintf(stderr, "no Metal device found\n");
        return 1;
    }
    if (!rt->queue()) {
        std::fprintf(stderr, "could not create command queue\n");
        return 1;
    }

    std::printf("Metal device: %s\n", rt->device()->name()->utf8String());
    std::printf("Device command queue alive at %p\n", (void*)rt->queue());

    MetalRuntime* rt2 = &runtime();
    std::printf("same instance: %s\n", rt == rt2 ? "yes" : "no");
    return 0;
}
