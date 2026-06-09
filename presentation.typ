#import "@preview/touying:0.7.3": *
#import themes.simple: *

#import "@preview/algorithmic:1.0.7"
#import algorithmic: style-algorithm, algorithm-figure, algorithm
#show: style-algorithm

#show: simple-theme.with(
  aspect-ratio: "16-9",
  // Don't recolor strong text with the accent color, so the
  // algorithm keywords (rendered with `strong`) stay black.
  config-common(show-strong-with-alert: false),
  config-info(
    title: [Supercharge your Python with Native Code],
    author: [Matteo Ceccarello],
    institution: [University of Padova]
  )
)

#title-slide[Supercharge your Python with Native Code]

#set text(font: "Lato")

#show image: set align(center)
#show raw.where(block: true, lang: "python"): set text(size: .8em)
#show raw.where(block: true, lang: "profile"): set text(size: .8em)

// #title-slide[]

== Outline <touying:hidden>

#components.adaptive-columns(outline(depth: 1, title: none, indent: 1em))

= The case study

== $k$-means

A very popular clustering objective, where we seek to
discover _structure_ in the data.

Given observations $X=(x_1, x_2, dots, x_n)$ we seek a partition $S={S_1, dots, S_k}$ of $X$ in $k$ groups
so that the Within Cluster Sum of Squares is minimized:

$
"WCSS"(S) = sum_(i=1)^k sum_(X in S_i) ||x - mu_i||^2
#h(2em)
"with" mu_i = 1/(|S_i|) sum_(x in S_i) x
$

== 

#cols[
  #image("imgs/digits.webp")
][
  #image("imgs/example-digits.png")
]

#text(size: .7em)[An example on the `mnist` digits dataset: 60k points in 784 dimensions, discovering $k=10$ clusters in the data (points embedded using UMAP)]

==

#{
set text(size: .9em)
show strong: set text(fill: luma(0))
set grid(row-gutter: .1em)

algorithm(
  line-numbers: false,
  indent: 1em,
  // vstroke: 1pt + luma(100),
  {
    import algorithmic: *

    // Drop the closing "end" keywords: rebind block constructors without kw3.
    let While = iflike.with(kw1: "while", kw2: "do")
    let For = iflike.with(kw1: "for", kw2: "do")
    let If = iflike.with(kw1: "if", kw2: "then")
    // Procedure/Function go through `call`, which hardcodes kw3, so define our own.
    let Procedure(name, args, ..body) = iflike(
      kw1: "procedure",
      (smallcaps(name) + $(#arraify(args).join(", "))$),
      ..body,
    )

    Procedure(
      "KMeans",
      ($X$, $k$),
      Assign[centroids][arbitrary subset of $X$],
      Assign[converged][`false`],
      While([!converged], 
        Assign[clusters][list of $k$ empty clusters],

        For([$i arrow.l 1$ to $n$], 
          [Find the centroid $c$ closest to $X_i$ ],
          [Assign $X_i$ to cluster $c$]
        ),

        Assign[newCentroids][emtpy list],

        For([$i arrow.l 1 "to" k$], {
          [Append to `newCentroids` the mean of cluster $i$]
        }),
        Assign[converged][newCentroids = centroids]
      )

    )
  }
)
}

== Key points

- The algorithm is structured in two phases:
  - Update the point assignment
  - Update the centroids
- Point assignment takes $Theta(n k)$ distance computations
- Usually one checks the change in WCSS: if it is negligible, stop
- The number of iterations is also usually limited

= A simple Python implementation

== The main loop


```python
def lloyd(points, k, max_iter=300, epsilon=1e-4):
    centroids = random_centroids(points, k)
    clusters = assign_closest(points, centroids)
    wcss = cost(clusters, centroids)

    for iter in range(max_iter):
        new_centroids, new_clusters = lloyd_iter(points, centroids)
        new_wcss = cost(new_clusters, new_centroids)
        if wcss - new_wcss <= epsilon * wcss:  # relative WCSS decrease
            break
        centroids = new_centroids
        clusters = new_clusters
        wcss = new_wcss
    return wcss, clusters, centroids
```

== A single iteration

```python
def lloyd_iter(points, centroids):
    k = len(centroids)
    clusters = assign_closest(points, centroids)

    # Update
    new_centroids = []
    for i in range(k):
        new_centroids.append(mean(clusters[i]))

    return new_centroids, clusters
```

== Implementing the assignment

```python
def assign_closest(points, centroids):
    k = len(centroids)
    clusters = [[] for _ in range(k)]

    for point in points:
        closest = 0
        min_distance = euclidean(point, centroids[closest])
        for i in range(1, k):
            d = euclidean(point, centroids[i])
            if d < min_distance:
                min_distance = d
                closest = i
        clusters[closest].append(point)
    return clusters
```

== Is this any good?

Test data:

- `fashion-mnist`: dataset of 60000 images of apparel items, 784-dimensional vectors
- `glove`: 1 183 514 word embeddings in 100 dimensions

== Is this any good? ($k=10$, 20s timeout)

#image("imgs/performance-naive.png")

// #footnote[We set a timeout at 20 seconds]

= Managing the Python environment

== `uv`: one tool to rule them all

#place(top + right, image("imgs/uv.svg"))

- A single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
- 10-100x faster than pip.
- Provides comprehensive project management, with a universal lockfile.
- Runs scripts, with support for inline dependency metadata.
- Installs and manages Python versions.
- Runs and installs tools published as Python packages.
- Supports macOS, Linux, and Windows.

== `uv` cheat sheet

Reference page: https://docs.astral.sh/uv/

- `uv init` initializes a workspace
- `uv add` adds a dependency
- `uv run python` runs the project's Python version
- `uv lock` updates the lockfile (pinning down dependencies)

= An efficient baseline

== The scikit-learn implementation

Scikit-learn is a very well-known project implementing a whole host of machine learning algorithms, along with other facilities.

The package provides the class #link("https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html")[`sklearn.cluster.KMeans`] implementing $k$-means clustering.

To install the package:

#raw("uv add scikit-learn")

== Getting the data

We will use a couple of large-ish datasets, stored in the `hdf5` format.

To install: `uv add h5py`

```python
import h5py

with h5py.File("fashion-mnist-784-euclidean.hdf5") as fp:
    data = fp["/train"][:]
```

= Measuring performance

== The benchmark

Using the `fashion-mnist` and `glove` datasets, we fix:

- $k=10$
- 20 iterations
- target relative improvement 0

== Simply getting the time

```python
import h5py
from sklearn.cluster import KMeans
import time

with h5py.File("fashion-mnist-784-euclidean.hdf5") as fp:
    data = fp["/train"][:]
start = time.perf_counter()
labels = KMeans(10).fit_predict(data)
end = time.perf_counter()
print("elapsed:", end - start, "s")
```

== #emoji.face.explode

#align(center, image(width: 90%,"imgs/performance-baseline.png"))

== Where is time being spent?

Using a profiler helps with this question.

Python has the built-in `cProfile` module.

```python
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# here goes the code you want to profile!
profiler.disable()
stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats(top)
stats.dump_stats("profile.prof") # output file (used later)
```

== The output

```profile
kmeans_naive: dataset=fashion-mnist k=10 n=10000 dim=784 seed=1234 wcss=21068242817.66 time=49.8400s
         2300596 function calls (2300592 primitive calls) in 50.010 seconds

   Ordered by: cumulative time
   List reduced from 40 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.033    0.033   50.010   50.010 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/main.py:106(run_kmeans_naive)
        1    0.001    0.001   49.840   49.840 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_naive.py:71(lloyd)
  1100000   47.284    0.000   47.389    0.000 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_naive.py:4(euclidean)
       10    0.204    0.020   43.150    4.315 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_naive.py:38(assign_closest)
        9    0.000    0.000   41.489    4.610 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_naive.py:55(lloyd_iter)
       10    0.000    0.000    4.484    0.448 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_naive.py:31(cost)
      100    0.028    0.000    4.484    0.045 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_naive.py:24(sum_of_squares)
       90    2.205    0.024    2.205    0.024 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_naive.py:11(mean)
        1    0.137    0.137    0.137    0.137 {method 'tolist' of 'numpy.ndarray' objects}
  1100000    0.105    0.000    0.105    0.000 {built-in method math.sqrt}
   100090    0.013    0.000    0.013    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/main.py:93(record_result)
        1    0.000    0.000    0.000    0.000 {built-in method _io.open}
        1    0.000    0.000    0.000    0.000 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_naive.py:67(random_centroids)
        1    0.000    0.000    0.000    0.000 /home/matteo/.local/share/uv/python/cpython-3.13.2-linux-x86_64-gnu/lib/python3.13/random.py:363(sample)
      202    0.000    0.000    0.000    0.000 {built-in method builtins.len}
        1    0.000    0.000    0.000    0.000 {method '__exit__' of '_io._IOBase' objects}
        1    0.000    0.000    0.000    0.000 {built-in method builtins.print}
        3    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 <frozen abc>:117(__instancecheck__)
        1    0.000    0.000    0.000    0.000 /home/matteo/.local/share/uv/python/cpython-3.13.2-linux-x86_64-gnu/lib/python3.13/random.py:128(seed)
        1    0.000    0.000    0.000    0.000 {method 'writerow' of '_csv.writer' objects}
        1    0.000    0.000    0.000    0.000 {built-in method _abc._abc_instancecheck}
      3/1    0.000    0.000    0.000    0.000 <frozen abc>:121(__subclasscheck__)
        1    0.000    0.000    0.000    0.000 <frozen genericpath>:16(exists)
      3/1    0.000    0.000    0.000    0.000 {built-in method _abc._abc_subclasscheck}
       10    0.000    0.000    0.000    0.000 /home/matteo/.local/share/uv/python/cpython-3.13.2-linux-x86_64-gnu/lib/python3.13/random.py:245(_randbelow_with_getrandbits)
        1    0.000    0.000    0.000    0.000 {built-in method posix.stat}
        1    0.000    0.000    0.000    0.000 {function Random.seed at 0x7f1e7a4fede0}
```

== More intuitive profile exploration: `snakeviz` 

Run with `uvx snakeviz profile.prof`

#align(center, image(width:70%, "imgs/snakefiz.png"))

#focus-slide[
  The culprit is the computation of the Euclidean distance in the assignment function
]

= Optimizing with `numpy`

== NumPy #box(image(width: 2em, "imgs/numpy.svg"))

- "The fundamental package for scientific computing with Python"
- Provides multi-dimensional arrays
- Core is written in C

#cols[
  To install: 
  ```
  uv add numpy
  ```
][
  To use:
  ```
  import numpy as np
  ```
]

// == Optimizing the assignment
== 

```python
def assign_closest(points, centroids):
    n, k = points.shape[0], centroids.shape[0]
    assignment = np.zeros(n)
    wcss = 0.0
    for i in range(n):
        closest = 0
        min_distance = np.linalg.norm(points[i] - centroids[closest])
        for c in range(1, k):
            d = np.linalg.norm(points[i] - centroids[c])
            if d < min_distance:
                min_distance = d
                closest = c
        assignment[i] = closest
        wcss += min_distance*min_distance # we need the squared distance here
    return assignment, wcss
```

== 

#image("imgs/performance-numpy.png")

== Where is time being spent?

```profile
kmeans_numpy_a: dataset=fashion-mnist k=10 n=10000 dim=784 seed=1234 wcss=21061181440.00 time=6.3214s
         16002380 function calls in 6.322 seconds

   Ordered by: cumulative time
   List reduced from 54 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    6.322    6.322 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/main.py:125(run_kmeans_numpy_a)
        1    0.076    0.076    6.321    6.321 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_numpy_a.py:31(lloyd)
       20    2.205    0.110    6.211    0.311 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_numpy_a.py:4(assign_closest)
  2000000    1.895    0.000    3.872    0.000 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/.venv/lib/python3.13/site-packages/numpy/linalg/_linalg.py:2599(norm)
  2000000    0.927    0.000    0.927    0.000 {method 'dot' of 'numpy.ndarray' objects}
  2000000    0.286    0.000    0.420    0.000 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/.venv/lib/python3.13/site-packages/numpy/linalg/_linalg.py:168(isComplexType)
  2000000    0.313    0.000    0.313    0.000 {method 'ravel' of 'numpy.ndarray' objects}
  4000400    0.292    0.000    0.292    0.000 {built-in method builtins.issubclass}
  2000000    0.159    0.000    0.159    0.000 {built-in method numpy.asarray}
```

==

Looking at the profile, clearly most of the time is still spent in the distance
computation (`norm`)

#image("imgs/profile-numpy.png")

#focus-slide[
  The bottleneck is still the distance computation, and we have a lot of ground to cover to reach sklearn
]

== Why?

The culprit is this line of the profile:

```profile
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20    2.205    0.110    6.211    0.311 /home/matteo/Work/Supercharge-your-Python-with-Native-Code/kmeans_numpy_a.py:4(assign_closest)
```

2 seconds are spent executing code in `assign_closest` that is not directly related to distance computations: running for loops in python is slow

== Rewriting the computation (1)

With the aim of reducing the number of `for` loops in Python, we can rewrite the computation into something that `numpy` is heavily optimized for.

Observe that

$
||x - c||^2 =
||x||^2 + ||c||^2 - 2 x dot c
$

We can thus precompute the norms of points ($x$) and centroids ($y$)
and only compute dot products.

== Rewriting the computation (2)

We can also compute the distance between any point and any centroid with matrix operations:

$
  X_"sq" - X C^T + C_"sq"
$

where $X_"sq"$ and $C_"sq"$ are the vectors of squared norms of points and centroids

== Rewriting the computation (2)

#[
#set text(size: .8em)
The matrix multiplication effectively computes all the dot products we need.
]

#image(width: 65%, "imgs/dots.pdf")

== Rewriting the computation (3)

- The two ways of computing distances are mathematically equivalent
- They are not _numerically_ equivalent, though: our newest expression is succeptible to _catastrophic cancellation_, where the result may even become negative!
- This is a tradeoff to be aware of

== A rewrite of the `numpy` implementation

```python
def assign_closest(points, centroids, points_sq=None):
    if points_sq is None:
        points_sq = np.linalg.norm(points, axis=1)**2
    c_sq = np.linalg.norm(centroids, axis=1)**2
    d = points_sq[:, None] - 2.0 * (points @ centroids.T) + c_sq[None, :]
    # deal with the aforementioned tradeoff
    np.maximum(d, 0.0, out=d)

    assignment = np.argmin(d, axis=1)
    wcss = float(np.min(d, axis=1).sum(dtype=np.float64))
    return assignment, wcss
```

== How does it fare?

#image("imgs/performance-numpy-matrix.png")

== How does it fare?

We got closer to the `sklearn` implementation.

The gap is not closed yet because materializing the entire distance matrix (i.e. $X C^T$) is expensive in terms of memory.

= Optimizing with `numba`

== #box(height: 2em, image(alt: "numba", "imgs/numba.svg"))

- a compiler for Python that translates Python to machine code using LLVM
- works on a per-function basis
- designed to work with numpy
- gives access to parallelism

== compiled vs. interpreted languages

#[
  #set text(size: .8em)

  #cols[
    *Interpreted* (e.g. Python)

    - Source is executed by the _interpreter_, one statement at a time
    - No separate build step: run the source directly
    - Portable: the same source runs wherever the interpreter does
    - Each operation pays the overhead of being decoded at run time
    - Types are checked at run time
  ][
    *Compiled* (e.g. C, C++)

    - Source is translated _ahead of time_ into machine code by a _compiler_
    - Requires a build step, producing a platform-specific executable
    - The CPU runs the machine code directly: no per-statement overhead
    - The compiler can optimize aggressively (inlining, vectorization, ...)
    - Types are checked at compile time
  ]
]

#focus-slide[
  #set text(size: .9em)
  #show raw: set text(size: 1.1em)
  
  `numba` brings _just-in-time_ compilation to Python: it compiles
  annotated functions to machine code the first time they are called.
]

== Using `numba`

- Install: `uv add numba`
- import with `from numba import njit`
- Use: annotate functions with `@njit`
- Some limitations apply
- Compiles the code on-the-fly on first use


== Using `numba`

#[
#show raw: set text(size: .9em)

```python
from numba import njit, prange
import numpy as np

@njit(fastmath=True, parallel=True)
def euclidean_all(data: np.ndarray, centroids: np.ndarray):
    dims = data.shape[1]
    dists = np.empty((data.shape[0], centroids.shape[0]), dtype=np.float64)
    for i in prange(data.shape[0]):
        for c in range(centroids.shape[0]):
            dist = 0.0
            for j in range(dims):
                diff = data[i,j] - centroids[c,j]
                dist += diff*diff
            dists[i,c] = dist
    return dists
```
]


== Using `numba`

This code does not compile

```python
from numba import njit

@njit
def make_record():
    # A dict whose values have different types (str and int).
    # Plain Python is fine with this, but Numba's typed dict requires
    # all values to share a single type, so this does not compile
    # in nopython mode.
    return {"name": "Alice", "age": 30}
```


== Example: the assignment loop

```python
@njit(parallel=True, fastmath=True, cache=True)
def _assign(points, centroids):
    n, dim = points.shape
    k = centroids.shape[0]
    assignment = np.empty(n, dtype=np.int64)
    # ↓ this array can be used by the caller to compute the wcss
    min_dist = np.empty(n, dtype=np.float64)
    for i in prange(n): # <- Parallel loop!
        best = 0
        best_d = 0.0
        for d in range(dim):
            diff = points[i, d] - centroids[0, d]
            best_d += diff * diff
            # Continues on next slide
            
        for j in range(1, k):
            dist = 0.0
            for d in range(dim):
                diff = points[i, d] - centroids[j, d]
                dist += diff * diff
            if dist < best_d:
                best_d = dist
                best = j
        assignment[i] = best
        min_dist[i] = best_d
    return assignment, min_dist
```

== How does it fare?

#image("imgs/performance-numba.png")

#focus-slide[
  We finally got on par!
]

#focus-slide(background: orange)[
  Can we do better?
]

= Writing a C++ extension

== The architecture

#image(width: 70%, "imgs/stack.png")

== Tools involved

- `cmake`: a build system for C++
- `nanobind`: a library to develop C++ extensions for Python
- `uv`: to drive the installation process

== Project scaffolding

The environment and dependencies are managed by `uv`.

`uv sync` creates `.venv/`, installs dependencies, and—because of the build backend below—compiles the C++ module into the environment in one shot.

== Project scaffolding

The basic directory structure is the following

```
.
├── CMakeLists.txt
├── main.py
├── pyproject.toml
├── supercharge
│   ├── lib.cpp
│   ├── lib.hpp
│   └── wrapper.cpp
└── uv.lock
```

== Configuring the Python side of the project

#[

#set text(size: .7em)

```toml
[dependency-groups]
# ↓ this actually provides the bindings if you
#   invoke CMake manually
dev = ["nanobind>=2.12.0"]

[build-system]
# things listed here are downloaded as build dependencies
requires = [
  # ↓ this is required to interact with cmake
  "scikit-build-core >=0.4.3",
  # ↓ this is also required for the automatic invocation of CMake
  "nanobind >= 2.12.0"
]
# here we replace the default build backend
build-backend = "scikit_build_core.build"
```
]

== Configuring CMake

- Adding a new library compiled from our sources
  ```CMake
  add_library(supercharge_core STATIC supercharge/lib.cpp)
  ```

- Building the C++ code#footnote[does not make it visible to Python]
  ```
  cmake -S . -B build
  ```

== Configuring CMake

#[
#set text(size: .9em)

1. Find the right Python — `find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)`.
2. Locate nanobind via the interpreter — runs `python -m nanobind --cmake_dir` to set `nanobind_ROOT`, then `find_package(nanobind)`. This is why the venv must be active when running CMake directly — it asks the active interpreter where nanobind lives.
3. Default to a Release build (otherwise you'd get an unoptimized debug build).
4. PIC on the static core lib — `POSITION_INDEPENDENT_CODE ON`, because supercharge_core gets linked into a shared module; without it the Linux link fails.
5. Enable SIMD — `target_compile_options(... -mavx2 -mfma)` so lib.cpp selects the vectorized path (-mfma lets the compiler fuse mul+add).
6. Link OpenMP so the `#pragma omp parallel for` in `assign()` is honored.
7. nanobind_add_module quirk — use the keyword form `target_link_libraries(supercharge PRIVATE ...)`; nanobind uses the keyword form internally, so mixing forms errors.
8. Install rule — `install(TARGETS supercharge LIBRARY DESTINATION .)`. Without it `scikit-build-core` builds the module and then discards it, so import supercharge fails.
]

== The C++ code

#[
#set text(size: .8em)
In `lib.hpp` we add this function signature.

```cpp
std::pair<std::vector<size_t>, float>
kmeans(const float *const points,
       const size_t n, const size_t dimensions,
       const size_t k,
       const size_t max_iter,
       const float tol,
       const uint64_t seed);
```

Note that:
- it accepts a simple `const` array of floats
- it returns a pair of a `std::vector`#footnote[which owns the memory] and the wcss score
]

== The glue code

The role of the binding layer is to translate between C++ and Python types.

This is the start of `supercharge/wrapper.cpp`:

```cpp
#include <nanobind/nanobind.h>
#include "lib.hpp"
#include "nanobind/ndarray.h"

namespace nb = nanobind;
```

== The glue code

... and this is the end of the file, which defines a _Python_ function kmeans
that uses the code of the `kmeans_wrapper` function pointer#footnote[Note the `&` in front of the function name.].

```cpp
NB_MODULE(supercharge, m) {
    m.def("kmeans", &kmeans_wrapper);
}
```

== The glue code

#cols[
  #set text(size: .8em)
  Here is the function doing the translation, which we will review piece by piece.
][
#show raw: set text(size: .3em)

```cpp
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
```
]


== The glue code: signature

#[
#set text(size: .9em)

```cpp
nb::tuple // ← we return a Python pair
kmeans_wrapper(
  // ↓ we accept a Numpy array: nanobind has dedicated types
  const nb::ndarray<float, nb::ndim<2>, nb::c_contig> &points,
  // ↓ all the other parameters are automatically mapped
  const size_t k, const size_t max_iter, const float tol,
  const uint64_t seed
)
```
]

== The glue code: getting the data

#[
#set text(size: .9em)

The `nb::ndarray` class provides direct access to the backing array.
It is important to be aware of the data layout (row-major in this case).

```cpp
  const float *const pts = points.data();
  const size_t n = points.shape(0);
  const size_t dimensions = points.shape(1);
```

The shape methods allow to retrieve the number of rows and columns.

We can now run our code.

```cpp
  auto [assignment, wcss] =
      kmeans(pts, n, dimensions, k, max_iter, tol, seed);
```
]

== The glue code: who owns the data?

#only(1)[
Currently, the `assignment` array is a `std::vector`:
- the `std::vector` object is allocated on the stack
- the data lives on the heap
- the data is deallocated once the `std::vector` exits the scope (RAII)
]

#only(2)[
#image("imgs/memory.png")
]

== The glue code: moving data to the heap#footnote[An alternative would be to explicitly copy all
the data to a new allocation, but we don't want to pay this cost.]

```cpp
  // move the assignment to the heap, so we can handle
  // ownership to the Python interpreter
  auto asmt = new std::vector(std::move(assignment));
```

#image(width: 80%,"imgs/memory2.png")

== The glue code: handing ownership to Python

The `nb::capsule` class is a container that allows to run custom code
when the Python garbage collector makes its pass.

```cpp
  auto asmt = new std::vector(std::move(assignment));
  nb::capsule owner(asmt, [](void *p) noexcept {
    delete static_cast<std::vector<size_t> *>(p);
  });
```

This allows to run custom code on cleanup.

== The glue code: returning results to Python

```cpp
  const size_t size = asmt->size();

  // this is the array metadata that we return
  nb::ndarray<nb::numpy, size_t, nb::ndim<1>, nb::c_contig>
      arr(
        asmt->data(), // ← the data
        {size},       // ← the shape
        owner         // ← the owning capsule
      );

  return nb::make_tuple(arr, wcss);
```

== Optimizations in C++

- Allocate only what's stricly necessary
- Compute distances with $||x||^2 + ||y||^2 - 2 x y$
- Use AVX, FMA, loop unrolling in the dot product
- Only one pass over memory: point assignment and centroid update in one go
- Handle points in parallel using OpenMP

== How does it fare?

#image("imgs/performance-all.png")


#focus-slide[
  We did it!

  3.2x faster on `fashion`\ 2.2x faster on `glove`!
]

== Optimization techniques: SIMD

#only(1)[
- Single Instruction Multiple Data processes multiple data elements in parallel
- This is architecture specific
- On `x86` we have the AVX instruction set
]

#only(2)[#image(width: 50%, "imgs/sisd.png")]
#only(3)[#image(width: 50%, "imgs/simd.png")]

== Optimization techniques: Fused-Multiply-Add

#only(1)[
  #cols[
  #set text(size: .9em)
  ```cpp
  float dot_product(
    float * a,
    float * b,
    size_t n
  ) {
    float dotp = 0.0;
    for(size_t i=0; i<n; i++){
      dotp = a[i] * b[i] + dotp;
    }
    return dotp;
  }
  ```
  ][
    #set text(size: .8em)
    There are specialized instructions that allow to apply the body of this `for` loop to chunks of 8 floating point values simultaneusly!
  ]
]

#only(2)[
  #cols(columns: (60%, auto))[
  #set text(size: .75em)
  ```cpp
  float dot_product(
    float * a,
    float * b,
    size_t n
  ) {
    auto acc = _mm256_setzero_ps();
    for(size_t i=0; i<n; i=i+8){
      acc = _mm256_fmadd_ps(
        _mm256_loadu_ps(a + i),
        _mm256_loadu_ps(b + i),
        acc
      );
    }
    return hsum256_scalar(acc);
  }
  ```
  ][
    #image(width: 80%, "imgs/fma.png")
  ]
]

#only(3)[
  #set text(size: .75em)

  ```cpp
  float hsum256_scalar(__m256 v) {
      alignas(32) float tmp[8];
      // spill the vector to a 32-byte aligned array
      _mm256_store_ps(tmp, v);
      float sum = 0.0f;
      for (int i = 0; i < 8; ++i) {
          sum += tmp[i];
      }
      return sum;
  }
  
  ```
]

== Optimization techniques: loop unrolling

#cols()[
  #set text(size: .8em)

  ```cpp
  float dot_product(
    float * a,
    float * b,
    size_t n
  ) {
    float dotp = 0.0;
    for(size_t i=0; i<n; i++){
      dotp = a[i] * b[i] + dotp;
    }
    return dotp;
  }
  ```
][
  #set text(size: .6em)

  ```cpp
  float dot_product(
    float * a,
    float * b,
    size_t n
  ) {
    float dotp = 0.0;
    size_t i=0;
    for(; i + 4 < n; i += 4){
      dotp = a[i+0] * b[i+0] + dotp;
      dotp = a[i+1] * b[i+1] + dotp;
      dotp = a[i+2] * b[i+2] + dotp;
      dotp = a[i+3] * b[i+3] + dotp;
    }
    for(; i<n; i++){
      dotp = a[i] * b[i] + dotp;
    }
    return dotp;
  }
  ```
]

== Optimization techniques: parallelism

OpenMP is a specification (implemented by different compilers)
that provides facilities to write parallel code:

- `#pragma omp parallel` defines a parallel region
- `#pragma omp for` makes the following loop parallel
- `#pragma omp critical` marks a section as "critical"


