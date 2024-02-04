//
// Created by adrozdov on 23/01/24.
// Basic vector/matrix toolset for minimal operations
//

#ifndef CUDA_EX_UTILS_H
#define CUDA_EX_UTILS_H

float* create_vector(int);
float** create_matrix(int);
void print_vector(float*, int);
void print_matrix(float**, int);
float* convert_to_1d(float**, int);

#endif //CUDA_EX_UTILS_H