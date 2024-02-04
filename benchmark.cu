#include <iostream>
#include <random>
#include <cassert>
#include "benchmark.h"

// Runs vectors sum benchmark for given vector size
std::pair<int, int> sum_benchmark(int size, int grid_d, int block_d) {
    std::cout << "Add bench: " << size << " elems" << ". Grid=" << grid_d << "x";
    std::cout << block_d << std::endl;

    auto a = create_vector(size);
    auto b = create_vector(size);
    auto res1 = create_vector(size);
    auto res2 = create_vector(size);

    // Create random vectors of given size
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(0.1, 1000.0);
    for (int i = 0; i < size; i++) {
        a[i] = distr(gen);
        b[i] = distr(gen);
    }

    // Perform addition
    auto gpu_time = gpu_add(a, b, res1, size, grid_d, block_d);
    auto cpu_time = cpu_add(a, b, res2, size);

    // Check that all blocks are calculated correctly
    for (int i = 0; i < size; i++) {
        assert(res2[i]== res1[i]);
    }

    std::cout << "CPU time " << cpu_time << " mcs" << std::endl;
    std::cout << "GPU time " << gpu_time << " mcs" << std::endl;
    std::cout << "Ratio: " << float(cpu_time) / float(gpu_time) << std::endl << std::endl;

    // Cleanup memory
    free(a);
    free(b);
    free(res1);
    free(res2);
    return std::pair(gpu_time, cpu_time);
}

// Runs matrix multiplication benchmark for given matrix N*N size
std::pair<int,int> matrix_benchmark(int size){
    std::cout << "Matrix bench: " << size << std::endl;
    float** a = create_matrix(size);
    float** b = create_matrix(size);

    // Create random matrix N*N
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distr(1.0, 3.0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i][j] = float(round(distr(gen)));
            b[i][j] = float(round(distr(gen)));
        }
    }

    // multiply C=A*B
    float** res1 = create_matrix(size);
    float** res2 = create_matrix(size);
    auto cpu_time = cpu_sq_matrix_mul(a, b, res1, size);
    auto gpu_time = gpu_sq_matrix_mul(a, b, res2, size);

    // validate result
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            assert(res1[i][j] == res2[i][j]);
        }
    }

    std::cout << "CPU time " << cpu_time << " mcs" << std::endl;
    std::cout << "GPU time " << gpu_time << " mcs" << std::endl;
    std::cout << "Ratio: " << float(cpu_time) / float(gpu_time) << std::endl << std::endl;

    // Clean memory
    free(a);
    free(b);
    free(res1);
    free(res2);

    return std::pair(gpu_time, cpu_time);
}
