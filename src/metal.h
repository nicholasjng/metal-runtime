#pragma once
#include <string>

// IWYU pragma: begin_exports
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
// IWYU pragma: end_exports

// Metal may return no NSError at all.
inline std::string describe(NS::Error* error) {
    return error ? error->localizedDescription()->utf8String() : "unknown error";
}

// Without a pool, autoreleased Cocoa/Metal objects leak for the process lifetime.
struct AutoreleaseScope {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    ~AutoreleaseScope() { pool->release(); }
};
