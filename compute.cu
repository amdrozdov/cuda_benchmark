#include<iostream>
#include<chrono>
#include "compute.h"

#define assert_gpu_error(ans) { gpu_assert((ans), __FILE__, __LINE__); }

// Raises assertion in case of GPU errors (memory/operation errors)
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Naive CPU vector sum
u_int64_t cpu_add(float* a, float* b, float*c, int size) {
    auto start = std::chrono::system_clock::now();

    for(int i =0;i<size;i++){
        c[i] = a[i] + b[i];
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
    return elapsed.count();
}

// GPU vector sum kernel
__global__ void add(float* a, float* b, float* c) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

// Wrapper for GPU sum kernel (handles cuda memory allocation/transfer)
u_int64_t gpu_add(float* a, float *b, float* res, int size, int grid_d, int block_d){
    size_t total_bytes = size * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, total_bytes);
    cudaMalloc(&d_B, total_bytes);
    cudaMalloc(&d_C, total_bytes);

    cudaMemcpy(d_A, a, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, total_bytes, cudaMemcpyHostToDevice);

    auto start = std::chrono::system_clock::now();

    add<<<grid_d, block_d>>>(d_A, d_B, d_C);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);

    cudaMemcpy(res, d_C, total_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    assert_gpu_error( cudaPeekAtLastError() );
    assert_gpu_error( cudaDeviceSynchronize() );
    return elapsed.count();
}

// GPU matrix multiplication
__global__ void matrix_mul(float* a, float* b, float* c, int size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size){
        return;
    }

    float result = 0;
    for(int i =0;i<size;i++){
        result += a[y*size + i] * b[i*size + x];
    }
    c[y*size + x] = result;
}

// GPU matrix multiplication wrapper (handles cuda memory allocation/transfer)
u_int64_t gpu_sq_matrix_mul(float** a, float** b, float** c, int size){
    // For matrix we will use 32 elements blocks in the grid
    int block_size = 32;
    // Calculate blocks/grid size for parallel execution on GPU
    dim3 grid(ceil(size/float(block_size)), ceil(size/float(block_size)));
    dim3 block(block_size, block_size, 1);

    // Convert to 2d matrixes into 1d array of floats
    size_t total_bytes = sizeof(float) * size * size;
    float *c_a = convert_to_1d(a, size);
    float *c_b = convert_to_1d(b, size);
    float *c_c = convert_to_1d(c, size);

    // Allocate cuda memory for all operands
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, total_bytes);
    cudaMalloc(&d_B, total_bytes);
    cudaMalloc(&d_C, total_bytes);
    // Transfer data to VRAM
    cudaMemcpy(d_A, c_a, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, c_b, total_bytes, cudaMemcpyHostToDevice);

    // Run the multiplication kernel on GPU using transferred data
    // This call is wrapped in the std::chrono time measurement
    auto start = std::chrono::system_clock::now();
    matrix_mul<<<grid, block>>>(d_A, d_B, d_C, size);
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);

    // Transfer the multiplication result back from VRAM to regular memory
    cudaMemcpy(c_c, d_C, total_bytes, cudaMemcpyDeviceToHost);

    // Unpack 1d array into 2d matrix
    int i=0;
    int j=0;
    for(int x=0;x<size*size;x++){
        c[i][j] = c_c[x];
        j++;
        if(j >= size){
            j = 0;
            i++;
        }
    }

    // Cleanup VRAM and RAM
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(c_a);
    free(c_b);
    free(c_c);

    // Check GPU errors/assert
    assert_gpu_error( cudaPeekAtLastError() );
    assert_gpu_error( cudaDeviceSynchronize() );

    // Return calculation time, + float** c contains the result
    return elapsed.count();
}

// Naive CPU matrix muliplication
u_int64_t  cpu_sq_matrix_mul(float** a, float** b, float** c, int size){
    auto start = std::chrono::system_clock::now();
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            for(int k=0;k<size;k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
    return elapsed.count();
}