//
// Created by adrozdov on 04/02/24.
// CPU/GPU computation functions
//

#ifndef CUDA_EX_COMPUTE_H
#define CUDA_EX_COMPUTE_H

#include "utils.h"

u_int64_t gpu_add(float*, float*, float*, int, int, int);
u_int64_t cpu_add(float*, float*, float*, int);
u_int64_t cpu_sq_matrix_mul(float**, float**, float**, int);
u_int64_t gpu_sq_matrix_mul(float**, float**, float**, int);

#endif //CUDA_EX_COMPUTE_H
