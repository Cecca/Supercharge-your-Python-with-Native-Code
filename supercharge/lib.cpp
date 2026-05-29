#include "lib.hpp"
#include <cstring>
#include <limits>
#include <random>
#include <unordered_set>

#if defined(__AVX2__)
#include <immintrin.h>

// 8-way unrolled AVX2 implementation: each loop iteration consumes
// 8 lanes * 8 unrolled ops = 64 floats, accumulating into 8 independent
// vector accumulators to keep the FP add pipeline full.
float squared_euclidean_distance(const float *a, const float *b,
                                 std::size_t n) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();
  __m256 acc7 = _mm256_setzero_ps();

  std::size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    __m256 d0 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + 0), _mm256_loadu_ps(b + i + 0));
    __m256 d1 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    __m256 d2 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
    __m256 d3 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
    __m256 d4 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + 32), _mm256_loadu_ps(b + i + 32));
    __m256 d5 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + 40), _mm256_loadu_ps(b + i + 40));
    __m256 d6 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + 48), _mm256_loadu_ps(b + i + 48));
    __m256 d7 =
        _mm256_sub_ps(_mm256_loadu_ps(a + i + 56), _mm256_loadu_ps(b + i + 56));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    acc2 = _mm256_fmadd_ps(d2, d2, acc2);
    acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    acc4 = _mm256_fmadd_ps(d4, d4, acc4);
    acc5 = _mm256_fmadd_ps(d5, d5, acc5);
    acc6 = _mm256_fmadd_ps(d6, d6, acc6);
    acc7 = _mm256_fmadd_ps(d7, d7, acc7);
  }

  // Remaining 8-wide chunks before the scalar tail.
  for (; i + 8 <= n; i += 8) {
    __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = _mm256_fmadd_ps(d, d, acc0);
  }

  __m256 s = _mm256_add_ps(
      _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)),
      _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7)));

  // Horizontal sum of the 8 lanes.
  __m128 lo = _mm256_castps256_ps128(s);
  __m128 hi = _mm256_extractf128_ps(s, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float sum = _mm_cvtss_f32(sum128);

  for (; i < n; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

#else

// Scalar fallback used when the translation unit is compiled without AVX2
// (i.e. the __AVX2__ macro is not defined). Enable AVX2 in the build flags
// (e.g. -mavx2 or -march=native) to select the vectorised path above.
float squared_euclidean_distance(const float *a, const float *b,
                                 std::size_t n) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

#endif

std::vector<size_t> sample_k(const size_t n, const size_t k,
                             const uint64_t seed) {
  // This uses Robert Floyd's algorithm:
  // https://dl.acm.org/doi/abs/10.1145/30401.315746
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

std::vector<float> random_centroids(const float *const points, const size_t n,
                                    const size_t dimensions, const size_t k,
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

float assign(const float *const points, const size_t n, const size_t dimensions,
             const size_t k, const std::vector<float> &centroids,
             std::vector<size_t> &assignment) {
  float wcss = 0.0;
  // Each point is assigned independently: assignment[i] is written by exactly
  // one iteration and wcss is a sum, so the loop parallelises cleanly with a
  // reduction on wcss.
#pragma omp parallel for reduction(+ : wcss) schedule(static)
  for (size_t i = 0; i < n; i++) {
    size_t closest = 0;
    float mindist = squared_euclidean_distance(
        points + i * dimensions, centroids.data() + closest * dimensions,
        dimensions);
    for (size_t j = 1; j < k; j++) {
      float d = squared_euclidean_distance(points + i * dimensions,
                                           centroids.data() + j * dimensions,
                                           dimensions);
      if (d < mindist) {
        mindist = d;
        closest = j;
      }
    }
    wcss += mindist;
    assignment[i] = closest;
  }
  return wcss;
}

std::pair<std::vector<size_t>, float>
kmeans(const float *const points, const size_t n, const size_t dimensions,
       const size_t k, const size_t max_iter, const float tol,
       const uint64_t seed) {
  std::vector<size_t> assignment(n);
  std::vector<size_t> counts(k); // the size of clusters
  std::vector<float> centroids =
      random_centroids(points, n, dimensions, k, seed);
  float prev_wcss = std::numeric_limits<float>::infinity();

  for (size_t iter = 0; iter < max_iter; iter++) {
    float wcss = assign(points, n, dimensions, k, centroids, assignment);
    if (prev_wcss - wcss < tol * prev_wcss) {
      prev_wcss = wcss;
      break;
    }
    prev_wcss = wcss;

    // update the centroids to the mean of their cluster
    std::fill(centroids.begin(), centroids.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    for (size_t i = 0; i < n; i++) {
      size_t c = assignment[i];
      counts[c]++;
      for (size_t j = 0; j < dimensions; j++) {
        centroids[c * dimensions + j] += *(points + i * dimensions + j);
      }
    }
    for (size_t c = 0; c < k; c++) {
      float size = (float)counts[c];
      if (size == 0) {
        continue; // skip emtpy clusters
      }
      for (size_t j = 0; j < dimensions; j++) {
        centroids[c * dimensions + j] /= size;
      }
    }
  }

  return {assignment, prev_wcss};
}
