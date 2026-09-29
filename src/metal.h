#pragma once

// IWYU pragma: begin_exports
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
// IWYU pragma: end_exports

// Without a pool, autoreleased Cocoa/Metal objects leak for the process lifetime.
struct AutoreleaseScope {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    ~AutoreleaseScope() { pool->release(); }
};
