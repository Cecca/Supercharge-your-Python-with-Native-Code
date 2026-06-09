#include <nanobind/nanobind.h>
#include "lib.hpp"
#include "nanobind/ndarray.h"

namespace nb = nanobind;

nb::tuple
kmeans_wrapper(const nb::ndarray<float, nb::ndim<2>, nb::c_contig> &points,
               const size_t k, const size_t max_iter, const float tol,
               const uint64_t seed) {
    // TODO: write down the code for the wrapper
}

// TODO: invoke the wrapper macro
