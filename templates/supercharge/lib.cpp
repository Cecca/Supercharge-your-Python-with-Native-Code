#include "lib.hpp"
#include <cstring>
#include <limits>
#include <random>
#include <unordered_set>

#if defined(__AVX2__)
#include <immintrin.h>

// 8-way unrolled AVX2 dot product: each loop iteration consumes
// 8 lanes * 8 unrolled ops = 64 floats, accumulating into 8 independent
// vector accumulators to keep the FMA pipeline full. The assignment step
// expresses the squared distance as ‖a‖² + ‖b‖² − 2·(a·b), so the dot
// product is the only per-element work left on the hot path -- one FMA per
// element instead of the subtract+square+add a direct distance would need.
float dot_product(const float *a, const float *b, std::size_t n) {
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
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 0),
                           _mm256_loadu_ps(b + i + 0), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                           _mm256_loadu_ps(b + i + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                           _mm256_loadu_ps(b + i + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                           _mm256_loadu_ps(b + i + 24), acc3);
    acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 32),
                           _mm256_loadu_ps(b + i + 32), acc4);
    acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 40),
                           _mm256_loadu_ps(b + i + 40), acc5);
    acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 48),
                           _mm256_loadu_ps(b + i + 48), acc6);
    acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 56),
                           _mm256_loadu_ps(b + i + 56), acc7);
  }

  // Remaining 8-wide chunks before the scalar tail.
  for (; i + 8 <= n; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
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
    sum += a[i] * b[i];
  }
  return sum;
}

#else

// Scalar fallback used when the translation unit is compiled without AVX2
// (i.e. the __AVX2__ macro is not defined). Enable AVX2 in the build flags
// (e.g. -mavx2 or -march=native) to select the vectorised path above.
float dot_product(const float *a, const float *b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
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

// Assigns every point to its closest centroid AND accumulates the per-cluster
// coordinate sums / counts needed for the next centroid update -- all in a
// single pass over the point matrix. The matrix is far larger than the cache
// (n*dimensions floats), so streaming it once per iteration instead of twice
// (a separate assign + update pass) roughly halves DRAM traffic on this
// bandwidth-bound kernel. `sums` and `counts` must be zeroed by the caller.
float assign(const float *const points, const size_t n, const size_t dimensions,
             const size_t k, const std::vector<float> &centroids,
             const std::vector<float> &centroid_sq,
             const std::vector<float> &point_sq,
             std::vector<size_t> &assignment, std::vector<float> &sums,
             std::vector<size_t> &counts) {
  float wcss = 0.0;
  // ‖p − c‖² = ‖p‖² + ‖c‖² − 2·(p·c). Across the k centroids ‖p‖² is constant,
  // so the closest centroid is the one minimising  ‖c‖² − 2·(p·c)  -- a single
  // dot product per centroid instead of a full distance. ‖p‖² and ‖c‖² are
  // precomputed by the caller (point_sq / centroid_sq).
  //
  // Each point is assigned independently, so the loop parallelises with a
  // reduction on wcss. The cluster sums/counts would race (many points map to
  // the same cluster), so each thread accumulates into private buffers and we
  // merge them once at the end -- a k*dimensions merge per thread, negligible
  // next to the n*dimensions pass it rides along with.
#pragma omp parallel reduction(+ : wcss)
  {
    std::vector<float> local_sums(dimensions * k, 0.0f);
    std::vector<size_t> local_counts(k, 0);

#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      const float *p = points + i * dimensions;
      size_t closest = 0;
      float best_score =
          centroid_sq[0] - 2.0f * dot_product(p, centroids.data(), dimensions);
      for (size_t j = 1; j < k; j++) {
        float score = centroid_sq[j] - 2.0f * dot_product(p,
                                                          centroids.data() +
                                                              j * dimensions,
                                                          dimensions);
        if (score < best_score) {
          best_score = score;
          closest = j;
        }
      }
      // Recover the true squared distance for the cost. Cancellation between
      // the large ‖p‖²/‖c‖² terms can push this a hair below zero, so clamp it.
      float mindist = point_sq[i] + best_score;
      if (mindist < 0.0f) {
        mindist = 0.0f;
      }
      wcss += mindist;
      assignment[i] = closest;

      // Fold this point into its cluster's running sum for the update step.
      local_counts[closest]++;
      float *acc = local_sums.data() + closest * dimensions;
      for (size_t j = 0; j < dimensions; j++) {
        acc[j] += p[j];
      }
    }

#pragma omp critical
    {
      for (size_t c = 0; c < k; c++) {
        counts[c] += local_counts[c];
      }
      for (size_t t = 0; t < dimensions * k; t++) {
        sums[t] += local_sums[t];
      }
    }
  }
  return wcss;
}

std::pair<std::vector<size_t>, float>
kmeans(const float *const points, const size_t n, const size_t dimensions,
       const size_t k, const size_t max_iter, const float tol,
       const uint64_t seed) {
  std::vector<size_t> assignment(n);
  std::vector<size_t> counts(k);          // the size of clusters
  std::vector<float> sums(dimensions * k); // per-cluster coordinate sums
  std::vector<float> centroids =
      random_centroids(points, n, dimensions, k, seed);
  float prev_wcss = std::numeric_limits<float>::infinity();

  // ‖p‖² is constant across iterations, so compute it once up front; assign()
  // needs it to recover the true cost from the dot-product formulation.
  std::vector<float> point_sq(n);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) {
    point_sq[i] =
        dot_product(points + i * dimensions, points + i * dimensions, dimensions);
  }

  // ‖c‖² changes whenever the centroids move; refreshed each iteration below.
  std::vector<float> centroid_sq(k);
  for (size_t c = 0; c < k; c++) {
    centroid_sq[c] = dot_product(centroids.data() + c * dimensions,
                                 centroids.data() + c * dimensions, dimensions);
  }

  for (size_t iter = 0; iter < max_iter; iter++) {
    // assign() both assigns points and accumulates the cluster sums/counts, so
    // zero those buffers before the pass.
    std::fill(sums.begin(), sums.end(), 0.0f);
    std::fill(counts.begin(), counts.end(), 0);
    float wcss = assign(points, n, dimensions, k, centroids, centroid_sq,
                        point_sq, assignment, sums, counts);
    if (prev_wcss - wcss < tol * prev_wcss) {
      prev_wcss = wcss;
      break;
    }
    prev_wcss = wcss;

    // Update the centroids to the mean of their cluster. The sums were already
    // gathered during assign(); empty clusters keep their previous position.
    for (size_t c = 0; c < k; c++) {
      float size = (float)counts[c];
      if (size == 0) {
        continue; // skip emtpy clusters
      }
      for (size_t j = 0; j < dimensions; j++) {
        centroids[c * dimensions + j] = sums[c * dimensions + j] / size;
      }
    }

    // The centroids moved, so refresh ‖c‖² for the next assign().
    for (size_t c = 0; c < k; c++) {
      centroid_sq[c] = dot_product(centroids.data() + c * dimensions,
                                   centroids.data() + c * dimensions,
                                   dimensions);
    }
  }

  return {assignment, prev_wcss};
}
