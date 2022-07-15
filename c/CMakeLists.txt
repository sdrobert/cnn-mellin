cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Cdrobert_Mellin VERSION ${CMAKE_PROJECT_VERSION})

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

include(GNUInstallDirs)

### Configuration
configure_file(
  config/cdrobert/mellin/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cdrobert/mellin/config.h
  @ONLY
)
option(
  MELLIN_SKIP_PERFORMANCE_BUILDS
  "Skip building performance-checking binaries"
  ON
)
option(
  MELLIN_BUILD_CUDA
  "Build cuda libraries"
  OFF
)

### Libraries
add_library(mellin INTERFACE)
add_library(Cdrobert::mellin ALIAS mellin)
target_compile_features(mellin INTERFACE cxx_constexpr)
target_include_directories(mellin
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

### Subdirectories
add_subdirectory(cpu)
target_link_libraries(mellin INTERFACE Cdrobert::mellin_cpu)
if (MELLIN_BUILD_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(mellin INTERFACE Cdrobert::mellin_cuda)
endif (MELLIN_BUILD_CUDA)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

## performance
if (NOT MELLIN_SKIP_PERFORMANCE_BUILDS)
  add_subdirectory(performance)
endif()

### Installation
install(
  TARGETS mellin
  EXPORT Cdrobert_MellinTargets
)
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

### Exportation
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/Cdrobert_MellinConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(config/Cdrobert_MellinConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/Cdrobert_MellinConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}"
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/Cdrobert_MellinConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/Cdrobert_MellinConfigVersion.cmake"
  DESTINATION
    "${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}"
)

install(
  EXPORT Cdrobert_MellinTargets
  FILE Cdrobert_MellinTargets.cmake
  NAMESPACE Cdrobert::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}"
)