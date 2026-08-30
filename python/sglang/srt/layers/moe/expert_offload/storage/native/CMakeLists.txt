cmake_minimum_required(VERSION 3.18)
project(sglang_expert_io_uring C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing>=2.4)

add_library(sglang_expert_io_uring SHARED io_uring_helper.c)
target_link_libraries(sglang_expert_io_uring PRIVATE PkgConfig::LIBURING)
target_compile_features(sglang_expert_io_uring PRIVATE c_std_11)
target_compile_options(
  sglang_expert_io_uring PRIVATE -O3 -Wall -Wextra -Werror
)
