// Minimal metal-cpp smoke test: print the system default GPU.
//
// This is the seed the runtime grows from (device/queue/buffers/MSL library
// cache + nanobind bindings — see ROADMAP.md). For now it just proves the
// metal-cpp fetch and the Objective-C bridge are wired up:
//
//   cmake -S . -B build -G Ninja && cmake --build build
//   ./build/metal_probe   ->   Metal device: Apple M1 Pro
//
// metal-cpp is header-only; this is the one TU that emits its implementation.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <cstdio>

int main() {
  MTL::Device* device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    std::fprintf(stderr, "no Metal device found\n");
    return 1;
  }
  std::printf("Metal device: %s\n", device->name()->utf8String());
  device->release();
  return 0;
}
