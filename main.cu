#include<vector>
#include <iostream>
#include <fstream>
#include "benchmark.h"

void sum_bench(){
    std::vector<std::vector<int>> a;
    a.push_back(std::vector<int>({100, 1, 100}));
    a.push_back(std::vector<int>({1000, 1, 1000}));
    a.push_back(std::vector<int>({10000, 10, 1000}));
    a.push_back(std::vector<int>({100000, 100, 1000}));
    a.push_back(std::vector<int>({1000000, 1000, 1000}));

    std::ofstream output;
    output.open("add_benchmark.csv", std::ios_base::trunc);
    output << "Size"<< "," << "GPU" << "," << "CPU" << std::endl;
    for(auto elem: a) {
        auto res = sum_benchmark(elem[0], elem[1], elem[2]);
        output << elem[0] << "," << res.first << "," << res.second << std::endl;
    }
    output.close();
}

void matrix_bench(){
    std::vector<int> tests{64, 300, 500, 800, 1000};
    std::ofstream output;
    output.open("matrix_benchmark.csv", std::ios_base::trunc);
    output << "Size"<< "," << "GPU" << "," << "CPU" << std::endl;
    for(auto t: tests){
        auto res = matrix_benchmark(t);
        output << t << "," << res.first << "," << res.second << std::endl;
    }
    output.close();
}

int main() {
    matrix_bench();
    sum_bench();
    return 0;
}
