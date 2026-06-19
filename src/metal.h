#pragma once

// metal-cpp is header-only; this is the one TU that emits its implementation.
//
// metal-cpp's own umbrella headers carry no IWYU export pragmas, so
// clang-include-cleaner resolves symbols like MTL::Device to the internal
// per-class header they're physically declared in (e.g. MTLDevice.hpp) and
// flags this file as "unused" in consumers that only include metal.h -- even
// though metal.h is this project's deliberate single entry point for
// metal-cpp types (see the "header firewall" note in notes/devlog.md). Mark
// it as a re-export so the tooling agrees.
// IWYU pragma: begin_exports
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
// IWYU pragma: end_exports

// Cocoa objects created via NS::String::string(...) / error-out-params / and
// autoreleased Metal objects (MTL::CommandBuffer, MTL::ComputeCommandEncoder) are
// only reclaimed by an enclosing NS::AutoreleasePool -- with none active they just
// leak silently for the process's lifetime instead of crashing. This bounds their
// lifetime to one logical unit of work (one dispatch, one compile).
struct AutoreleaseScope {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    ~AutoreleaseScope() { pool->release(); }
};
