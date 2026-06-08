# Supercharge your Python with Native Code

Material for a short, tutorial-style course on making Python code faster.

The approach is deliberately incremental. We start by *measuring* performance —
both time and memory — and only then reach for tools that give us progressively
more control over what the machine actually does:

```
pure Python  →  NumPy  →  Numba  →  native C++ (nanobind)
```

The running example throughout is **k-means clustering** (Lloyd's algorithm).
It is simple enough to fit on a screen, but its inner loop — computing distances
between many points and a handful of centroids — is exactly the kind of numeric
hot loop where each tool earns its keep. `scikit-learn`'s `KMeans` is included as
a "what good looks like" reference point.

## The implementations

Each file solves the same problem; comparing them is the point of the course.

| File | Approach | What to look at |
| --- | --- | --- |
| `kmeans_a.py` | Pure Python, lists and loops | The baseline. Readable, and slow. |
| `kmeans_numpy_naive.py` | Data stored with NumPy | Points are NumPy arrays. Faster but still far from good. |
| `kmeans_numpy.py` | Vectorised with NumPy | The whole distance matrix in a few array expressions (`\|\|x-c\|\|² = \|\|x\|\|² - 2x·c + \|\|c\|\|²`). |
| `kmeans_numba.py` | JIT-compiled with Numba | The same scalar loops as the baseline, but `@njit(parallel=True, fastmath=True)`; the n×k distance matrix is never materialised. |
| `supercharge/` | Native C++ via nanobind | Hand-written AVX2 + FMA distance kernel, OpenMP-parallelised assignment, exposed to Python as the `supercharge` module. |
| `sklearn` (in `main.py`) | `sklearn.cluster.KMeans` | A mature, optimised reference. |

The C++ core lives in `supercharge/lib.cpp` (the algorithm and the SIMD kernel)
and `supercharge/wrapper.cpp` (the nanobind bindings). The wrapper is worth
reading for one subtle point the course returns to: **ownership of the array
returned to Python**. The assignment vector is moved to the heap and handed to a
`nb::capsule` so that the NumPy array and the C++ allocation share a lifetime,
and neither is freed too early.

## Requirements

- Python 3.13 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- A C++ compiler with OpenMP and AVX2 support (recent GCC or Clang), plus CMake
  ≥ 3.15 — needed to build the native module

`nanobind` (the C++/Python binding library) and `scikit-build-core` (the build
backend that drives CMake) are declared in `pyproject.toml`, so you do not
install them by hand.

## Building

The native module is built automatically from the C++ sources whenever the
project is installed, because `pyproject.toml` uses `scikit-build-core` as its
build backend. The simplest path is therefore:

```bash
uv sync
```

This creates a virtual environment in `.venv/`, installs all dependencies, and
compiles the `supercharge` extension module into the environment. After it
finishes, `import supercharge` works:

```bash
uv run python -c "import supercharge; print(supercharge.__file__)"
```

Prefix commands with `uv run` to execute them inside the project environment, or
activate it once with `source .venv/bin/activate`.

### Rebuilding the C++ module during development

`uv sync` rebuilds the extension when you change the Python or build
configuration, but for a fast edit/compile/test loop on the C++ itself it is
convenient to drive CMake directly. **Do this with the virtual environment
activated**, so that CMake finds the right Python interpreter and the installed
`nanobind`:

```bash
source .venv/bin/activate

cmake -S . -B build
cmake --build build
```

CMake locates `nanobind` by asking the active interpreter where it is
(`python -m nanobind --cmake_dir`), which is why the venv must be active. The
build also enables AVX2/FMA (`-mavx2 -mfma`) and links OpenMP; if your CPU lacks
AVX2 the code falls back to a scalar distance kernel automatically.

The `build/` directory additionally produces `compile_commands.json`, which
editor tooling (clangd) can use for completion and diagnostics.

## Running the benchmark

```bash
uv run python main.py
```

On first run this downloads two datasets from
[ann-benchmarks.com](http://ann-benchmarks.com) — `fashion-mnist`
(60k × 784) and `glove-100` (≈1.2M × 100) — into the project directory as HDF5
files. It then runs every implementation on growing prefixes of each dataset
(1000 points, doubling up to the full size), recording the within-cluster sum of
squares and the wall-clock time of each run to `results.csv`. Implementations
that exceed a per-run timeout are dropped for the rest of that dataset, so the
pure-Python version does not hold everything up.

Useful options:

```bash
# Average several runs per configuration (results are appended, not recomputed)
uv run python main.py --runs 5
```

Already-recorded configurations are skipped, so re-running tops up missing data
rather than redoing work.

### Plotting

```bash
uv run python plot.py
```

reads `results.csv` and writes `performance.png`: runtime vs. number of points,
on log-log axes, one panel per dataset. This is the figure that tells the story
of the course.

## Profiling

`main.py` can run a single implementation under a profiler instead of the full
benchmark — this is the "measure first" half of the course.

### CPU time with cProfile

```bash
# Profile one implementation at a given problem size
uv run python main.py --profile kmeans_numpy --n 8000

# Dump the profile to a file for later inspection
uv run python main.py --profile kmeans_numpy --n 8000 --profile-output profile.prof
```

The choices for `--profile` are `kmeans_a`, `kmeans_numpy`, `kmeans_numba`,
`supercharge`, and `sklearn`. To explore a dumped `.prof` file interactively, use
[snakeviz](https://jiffyclub.github.io/snakeviz/):

```bash
uv run snakeviz profile.prof
```

### Memory with memray

```bash
uv run python main.py --memray kmeans_numpy --n 8000 --memray-output memray.bin
uv run memray flamegraph memray.bin   # or: uv run memray tree memray.bin
```

Add `--memray-native` to capture native stack frames as well, which is what lets
you see allocations happening inside NumPy/Numba/C++ rather than just at the
Python boundary.

