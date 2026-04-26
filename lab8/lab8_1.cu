// ============================================================
// Лабораторная работа 8.1
// Поэлементное умножение и сложение векторов на CUDA
//
// Компиляция:
//   nvcc lab8_1.cu -o lab8_1
// Запуск:
//   ./lab8_1
// ============================================================

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// ---- размер вектора (кратен 512) ----
const int N = 1024;

// ================================================================
// CUDA-ядро: поэлементное УМНОЖЕНИЕ  c[i] = a[i] * b[i]
// ================================================================
__global__ void VecMulKernel(float* a, float* b, float* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] * b[i];
}

// ================================================================
// CUDA-ядро: поэлементное СЛОЖЕНИЕ  c[i] = a[i] + b[i]
// ================================================================
__global__ void VecAddKernel(float* a, float* b, float* c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}

// ================================================================
// Обёртка для умножения на GPU
// ================================================================
void vec_mul_cuda(float* a, float* b, float* c, int n) {
    int size = n * sizeof(float);
    float *a_gpu, *b_gpu, *c_gpu;

    cudaMalloc((void**)&a_gpu, size);
    cudaMalloc((void**)&b_gpu, size);
    cudaMalloc((void**)&c_gpu, size);

    cudaMemcpy(a_gpu, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, size, cudaMemcpyHostToDevice);

    dim3 threads(512, 1);
    dim3 blocks(n / threads.x, 1);
    VecMulKernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);

    cudaMemcpy(c, c_gpu, size, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}

// ================================================================
// Обёртка для сложения на GPU
// ================================================================
void vec_add_cuda(float* a, float* b, float* c, int n) {
    int size = n * sizeof(float);
    float *a_gpu, *b_gpu, *c_gpu;

    cudaMalloc((void**)&a_gpu, size);
    cudaMalloc((void**)&b_gpu, size);
    cudaMalloc((void**)&c_gpu, size);

    cudaMemcpy(a_gpu, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, size, cudaMemcpyHostToDevice);

    dim3 threads(512, 1);
    dim3 blocks(n / threads.x, 1);
    VecAddKernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);

    cudaMemcpy(c, c_gpu, size, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}

// ================================================================
// CPU-аналог сложения для проверки корректности
// ================================================================
void vec_add_cpu(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

// ================================================================
// main
// ================================================================
int main() {
    float a[N], b[N];
    float c_mul[N];         // результат умножения на GPU
    float c_add_gpu[N];     // результат сложения на GPU
    float c_add_cpu[N];     // результат сложения на CPU

    // заполняем случайными данными [0, 10]
    srand(42);
    for (int i = 0; i < N; ++i) {
        a[i] = (float)(rand() % 100) / 10.0f;
        b[i] = (float)(rand() % 100) / 10.0f;
    }

    // ---- Умножение на GPU ----
    vec_mul_cuda(a, b, c_mul, N);
    cout << "=== Поэлементное умножение (GPU), первые 10 элементов ===\n";
    for (int i = 0; i < 10; ++i)
        printf("  c[%d] = %.2f * %.2f = %.2f\n", i, a[i], b[i], c_mul[i]);

    // ---- Сложение на GPU ----
    vec_add_cuda(a, b, c_add_gpu, N);

    // ---- Сложение на CPU ----
    vec_add_cpu(a, b, c_add_cpu, N);

    // ---- Проверка корректности ----
    int errors = 0;
    for (int i = 0; i < N; ++i)
        if (fabsf(c_add_gpu[i] - c_add_cpu[i]) > 1e-4f)
            ++errors;

    cout << "\n=== Поэлементное сложение: GPU vs CPU ===\n";
    cout << "Первые 10 элементов:\n";
    for (int i = 0; i < 10; ++i)
        printf("  a[%d]=%.2f  b[%d]=%.2f  GPU=%.2f  CPU=%.2f\n",
               i, a[i], i, b[i], c_add_gpu[i], c_add_cpu[i]);

    cout << "\nПроверка корректности: ";
    if (errors == 0)
        cout << "PASSED (все " << N << " элементов совпадают)\n";
    else
        cout << "FAILED (" << errors << " отличий)\n";

    cout << "\nIt is working!\n";
    return 0;
}
