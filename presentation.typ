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

#set text(font: "Lato")

#show raw.where(block: true, lang: "python"): set text(size: .8em)

#title-slide[]

== Outline <touying:hidden>

#components.adaptive-columns(outline(title: none, indent: 1em))

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

```
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

= Optimizing with `numba`

= Writing a C++ extension
