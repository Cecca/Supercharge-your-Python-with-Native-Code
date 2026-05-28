#include <nanobind/nanobind.h>
#include "lib.hpp"
#include "nanobind/ndarray.h"

namespace nb = nanobind;
using namespace nb::literals;

nb::tuple
kmeans_wrapper(const nb::ndarray<float, nb::ndim<2>, nb::c_contig> &points,
               const size_t k, const size_t max_iter, const float tol,
               const uint64_t seed) {
  const float *const pts = points.data();
  const size_t n = points.shape(0);
  const size_t dimensions = points.shape(1);

  auto [assignment, wcss] = kmeans(pts, n, dimensions, k, max_iter, tol, seed);

  // move the assignment to the heap, so we can handle
  // ownership to the Python interpreter
  auto asmt = new std::vector(std::move(assignment));
  nb::capsule owner(asmt, [](void *p) noexcept {
    delete static_cast<std::vector<size_t> *>(p);
  });

  const size_t size = asmt->size();

  // this is the array metadata that we return
  nb::ndarray<nb::numpy, size_t, nb::ndim<1>, nb::c_contig> arr(
      asmt->data(), {size}, owner
  );

  return nb::make_tuple(arr, wcss);
}

NB_MODULE(supercharge, m) {
    m.def("kmeans", &kmeans_wrapper);
}
