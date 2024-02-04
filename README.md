## cuda_benchmark
Code from my cuda presentation.

In this example we are running basic math operations (vector addition and matrix multiplication) on CPU and GPU using cuda. Using visualisation tool you can see how GPU overtakes CPU using parallel computing in bigger data sizes. In the first example you can see how CPU overtakes GPU on small amounts of data because of computational overhead.

Usage:
```
pip install -r vis_requirements.txt
# build the tool
make tool
...
# Run the benchmark
make run
...
# Visualize results
make vis
```

### Output examples
[Vector addition](https://github.com/amdrozdov/cuda_benchmark/blob/main/compute.cu#L38)
![Addition](https://github.com/amdrozdov/cuda_benchmark/blob/main/exampels/add.png)

[Matrix multiplication](https://github.com/amdrozdov/cuda_benchmark/blob/main/compute.cu#L84)
![Multiplication](https://github.com/amdrozdov/cuda_benchmark/blob/main/exampels/matrix_mul.png)

## Useful links
1. [CUDA programming model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-a-general-purpose-parallel-computing-platform-and-programming-model)
2. [NVCC](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)
3. [List of supported math functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix)
