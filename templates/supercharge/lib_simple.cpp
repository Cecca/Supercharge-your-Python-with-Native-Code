#include "lib.hpp"
#include <cstring>
#include <limits>
#include <random>
#include <unordered_set>

// A deliberately simple, unoptimized k-means (Lloyd's algorithm).
//
// It implements exactly the same `kmeans` interface as lib.cpp, but with none
// of the tricks: the squared distance is computed directly (subtract, square,
// add) in a plain scalar loop, there is no SIMD, no OpenMP, no precomputed
// norms, and assignment and centroid update are two separate passes over the
// data.
//
// Build this *instead of* lib.cpp (they both define `kmeans`): point the
// `supercharge_core` target in CMakeLists.txt at this file to use it.

// Straightforward squared Euclidean distance: sum over dimensions of (a-b)^2.
static float squared_distance(const float *a, const float *b,
                              const size_t dimensions) {
  float sum = 0.0f;
  for (size_t j = 0; j < dimensions; j++) {
    float diff = a[j] - b[j];
    sum += diff * diff;
  }
  return sum;
}

// Pick k distinct point indices uniformly at random (Robert Floyd's algorithm).
static std::vector<size_t> sample_k(const size_t n, const size_t k,
                                    const uint64_t seed) {
  std::mt19937_64 gen;
  gen.seed(seed);
  std::unordered_set<size_t> s;

  for (size_t i = n - k; i < n; ++i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    size_t t = dist(gen);
    if (!s.insert(t).second) {
      s.insert(i);
    }
  }

  return std::vector<size_t>(s.begin(), s.end());
}

// Initialise the centroids by copying k randomly chosen points.
static std::vector<float> random_centroids(const float *const points,
                                           const size_t n,
                                           const size_t dimensions,
                                           const size_t k,
                                           const uint64_t seed) {
  std::vector<float> centroids(dimensions * k);
  std::vector<size_t> idxs = sample_k(n, k, seed);
  for (size_t i = 0; i < k; i++) {
    size_t j = idxs[i];
    std::memcpy(centroids.data() + i * dimensions, points + j * dimensions,
                sizeof(float) * dimensions);
  }
  return centroids;
}

std::pair<std::vector<size_t>, float>
kmeans(const float *const points, const size_t n, const size_t dimensions,
       const size_t k, const size_t max_iter, const float tol,
       const uint64_t seed) {
  std::vector<size_t> assignment(n);
  std::vector<float> centroids =
      random_centroids(points, n, dimensions, k, seed);
  float prev_wcss = std::numeric_limits<float>::infinity();

  for (size_t iter = 0; iter < max_iter; iter++) {
    // --- Assignment step: assign each point to its nearest centroid. ---
    float wcss = 0.0f;
    for (size_t i = 0; i < n; i++) {
      const float *p = points + i * dimensions;
      size_t closest = 0;
      float best = squared_distance(p, centroids.data(), dimensions);
      for (size_t c = 1; c < k; c++) {
        float dist =
            squared_distance(p, centroids.data() + c * dimensions, dimensions);
        if (dist < best) {
          best = dist;
          closest = c;
        }
      }
      assignment[i] = closest;
      wcss += best;
    }

    // Stop once the cost stops improving by more than the tolerance.
    if (prev_wcss - wcss < tol * prev_wcss) {
      prev_wcss = wcss;
      break;
    }
    prev_wcss = wcss;

    // --- Update step: move each centroid to the mean of its cluster. ---
    std::vector<float> sums(dimensions * k, 0.0f);
    std::vector<size_t> counts(k, 0);
    for (size_t i = 0; i < n; i++) {
      size_t c = assignment[i];
      counts[c]++;
      const float *p = points + i * dimensions;
      float *acc = sums.data() + c * dimensions;
      for (size_t j = 0; j < dimensions; j++) {
        acc[j] += p[j];
      }
    }
    for (size_t c = 0; c < k; c++) {
      if (counts[c] == 0) {
        continue; // empty cluster: keep its previous position
      }
      for (size_t j = 0; j < dimensions; j++) {
        centroids[c * dimensions + j] = sums[c * dimensions + j] / counts[c];
      }
    }
  }

  return {assignment, prev_wcss};
}
