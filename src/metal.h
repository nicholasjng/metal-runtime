#pragma once

// metal-cpp's umbrella headers carry no IWYU export pragmas, so
// clang-include-cleaner attributes types like MTL::Device to the internal
// per-class header they're declared in (e.g. MTLDevice.hpp), not here, and
// flags this include as unused wherever it's actually used. Marked as a
// re-export so the tooling agrees this is the project's single entry point
// for metal-cpp types.
// IWYU pragma: begin_exports
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
// IWYU pragma: end_exports

// NS::AutoreleasePool is what reclaims Cocoa objects from
// NS::String::string(...), error out-params, and autoreleased Metal objects
// (MTL::CommandBuffer, MTL::ComputeCommandEncoder). With none active they leak
// for the process's lifetime instead of crashing. Scopes that lifetime to one
// dispatch or compile.
struct AutoreleaseScope {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    ~AutoreleaseScope() { pool->release(); }
};
