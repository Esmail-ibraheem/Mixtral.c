#ifndef MIXTRAL_TENSOR_H
#define MIXTRAL_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include "allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Data types supported for tensor elements
typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT16,
    DTYPE_INT8_Q      // Quantized 8-bit with per-tensor scale
} dtype_t;

/// Core tensor structure
typedef struct {
    void      *data;      ///< Pointer to raw buffer
    int64_t   *shape;     ///< Array of dimension sizes (length = ndim)
    int        ndim;      ///< Number of dimensions
    int64_t   *strides;   ///< Row-major strides for each dimension
    dtype_t    dtype;     ///< Element data type
    int        refcount;  ///< Reference count for shared ownership
} Tensor;

/// Create a new tensor
/// - ndim: number of dimensions
/// - shape: array of length ndim specifying each dimension size
/// - dtype: data type for elements
Tensor* tensor_create(int ndim, const int64_t *shape, dtype_t dtype);

/// Increment reference count
void tensor_retain(Tensor *t);

/// Decrement reference count and free when zero
void tensor_release(Tensor *t);

/// Number of elements in tensor (product of shape)
size_t tensor_numel(const Tensor *t);

/// Get pointer to the underlying data buffer
void* tensor_data_ptr(const Tensor *t);

/// Reshape tensor (returns new view, shares underlying data)
Tensor* tensor_reshape(const Tensor *t, int ndim, const int64_t *new_shape);

/// Transpose two dimensions of a tensor (returns new view)
Tensor* tensor_transpose(const Tensor *t, int dim0, int dim1);

/// Slice a tensor along one dimension [start, end) (returns new view)
Tensor* tensor_slice(const Tensor *t, int dim, int64_t start, int64_t end);

#ifdef __cplusplus
}
#endif

#endif // MIXTRAL_TENSOR_H
