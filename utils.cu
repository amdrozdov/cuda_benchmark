#include<iostream>
#include "utils.h"


float* create_vector(int size){
    auto f = (float*)(malloc(sizeof(float) * size));
    memset(f, 0, size);
    return f;
}

float** create_matrix(int size){
    auto res = (float**) malloc(sizeof(int*) * size);
    for(int i = 0;i < size; i ++){
        res[i] = create_vector(size);
    }
    return res;
}

float* convert_to_1d(float** x, int size){
    float * res = create_vector(size*size);
    int p = 0;
    for(int i =0;i<size;i++){
        for(int j =0;j<size;j++){
            res[p++] = x[i][j];
        }
    }
    return res;
}

void print_vector(float *vec, int size) {
    std::cout << "[";
    for (int i = 0; i < size; i++) {
        std::cout << vec[i];
        if(i != size - 1){
            std::cout << ", ";
        }
    }
    std::cout << "]" <<std::endl;
}

void print_matrix(float **m, int size){
    std::cout << "<" << std::endl;
    for(int i=0;i<size;i++){
        print_vector(m[i], size);
    }
    std::cout << ">" << std::endl;
}