cmake_minimum_required(VERSION 3.15...4.2.0)
project(supercharge)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.13 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# OpenMP is used to parallelise the assignment step in lib.cpp.
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(supercharge_core STATIC supercharge/lib.cpp)
# PIC is needed because this static library gets linked into the shared
# nanobind module below; without it the link step fails on Linux.
set_target_properties(supercharge_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Enable AVX2 (defines __AVX2__) so lib.cpp selects the vectorised distance
# implementation. -mfma lets the compiler fuse the mul+add pairs into FMA.
target_compile_options(supercharge_core PRIVATE -mavx2 -mfma)
# Link OpenMP so the `#pragma omp parallel for` in assign() is honoured.
target_link_libraries(supercharge_core OpenMP::OpenMP_CXX)

nanobind_add_module(supercharge supercharge/wrapper.cpp)
# PRIVATE is required: nanobind_add_module internally uses the keyword form of
# target_link_libraries, so all uses on this target must also use a keyword.
target_link_libraries(supercharge PRIVATE supercharge_core)

# scikit-build-core only packages what CMake installs. Without this rule the
# compiled module is built and then discarded, so `import supercharge` fails.
install(TARGETS supercharge LIBRARY DESTINATION .)
