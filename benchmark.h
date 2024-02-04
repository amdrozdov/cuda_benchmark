//
// Created by adrozdov on 04/02/24.
// CPU vs GPU benchmarks implementation
//

#ifndef CUDA_EX_BENCHMARK_H
#define CUDA_EX_BENCHMARK_H

#include "compute.h"

std::pair<int,int> sum_benchmark(int, int, int);
std::pair<int,int> matrix_benchmark(int);

#endif //CUDA_EX_BENCHMARK_H