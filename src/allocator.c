// allocator.c - Simple memory allocator for Mixtral

#include "allocator.h"
#include <stdlib.h>

/// Allocate a block of memory of given size (in bytes)
void* mt_alloc(size_t size) {
    return malloc(size);
}

/// Free a block of memory previously allocated with mt_alloc
void mt_free(void* ptr) {
    free(ptr);
}

/// Increment reference count for a memory block (no-op in this implementation)
void mt_retain(void* ptr) {
    (void)ptr;  // No-op
}

/// Decrement reference count for a memory block (no-op in this implementation)
void mt_release(void* ptr) {
    (void)ptr;  // No-op
}
