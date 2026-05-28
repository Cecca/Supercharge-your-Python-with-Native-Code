#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

std::pair<std::vector<size_t>, float>
kmeans(const float *const points, const size_t n, const size_t dimensions,
       const size_t k, const size_t max_iter, const float tol,
       const uint64_t seed);
