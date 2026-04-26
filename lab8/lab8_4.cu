// ============================================================
// Лабораторная работа 8.4
// Умножение матриц на CUDA: классический, с кэшированием,
// блочный алгоритм с разделяемой памятью
//
// Компиляция:
//   nvcc -O2 lab8_4.cu -o lab8_4
// Запуск:
//   ./lab8_4
// ============================================================

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// ---- параметры ----
#define S 16      // размер блока (tile) для блочного алгоритма
#define N 512     // размер матрицы (степень двойки, кратна S)
#define REPS 5    // повторений для замера времени

// ================================================================
// CUDA-ядро 1: Классическое умножение (без оптимизаций)
// Каждый поток вычисляет один элемент C[i][j]
// ================================================================
__global__ void MatMulKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // строка
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // столбец
    if (i >= n || j >= n) return;
    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
        sum += A[i*n + k] * B[k*n + j];
    C[i*n + j] = sum;
}

// ================================================================
// CUDA-ядро 2: Кэширование столбца B в разделяемой памяти
// Каждый блок кэширует j-й столбец B, вычисляет строку C
// ================================================================
__global__ void MatMulColCacheKernel(float* A, float* B, float* C, int n) {
    extern __shared__ float col[];  // кэш столбца B[*, j] размером n float

    int j = blockIdx.x;            // номер столбца
    int i = threadIdx.x;           // номер строки (один поток — одна строка)

    // загружаем j-й столбец B в разделяемую память
    if (i < n)
        col[i] = B[i*n + j];
    __syncthreads();

    if (i < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[i*n + k] * col[k];
        C[i*n + j] = sum;
    }
}

// ================================================================
// CUDA-ядро 3: Блочное умножение с разделяемой памятью (tiling)
// Каждый блок потоков вычисляет блок S×S результирующей матрицы
// ================================================================
__global__ void MatMulTiledKernel(float* A, float* B, float* C, int n) {
    __shared__ float tileA[S][S];
    __shared__ float tileB[S][S];

    int row = blockIdx.y * S + threadIdx.y;
    int col = blockIdx.x * S + threadIdx.x;
    float sum = 0.0f;

    // проходим по блокам вдоль общего измерения
    for (int t = 0; t < n / S; ++t) {
        // каждый поток загружает один элемент блока A и B
        tileA[threadIdx.y][threadIdx.x] = A[row*n + (t*S + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t*S + threadIdx.y)*n + col];
        __syncthreads();

        // умножаем загруженные блоки
        for (int k = 0; k < S; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < n)
        C[row*n + col] = sum;
}

// ================================================================
// CPU-умножение для проверки корректности
// ================================================================
void matmul_cpu(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0;
            for (int k = 0; k < n; ++k)
                s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}

// ================================================================
// Проверка корректности
// ================================================================
bool checkResult(float* ref, float* res, int n, float eps=1e-2f) {
    for (int i = 0; i < n*n; ++i)
        if (fabsf(ref[i] - res[i]) > eps) {
            printf("  FAIL at [%d]: ref=%.4f got=%.4f\n", i, ref[i], res[i]);
            return false;
        }
    return true;
}

// ================================================================
// Замер времени одного метода
// ================================================================
float runAndMeasure(void(*launch)(float*,float*,float*,int), 
                   float* A_d, float* B_d, float* C_d, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int r = 0; r < REPS; ++r) launch(A_d, B_d, C_d, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / REPS;
}

// ================================================================
// Вспомогательные функции запуска ядер
// ================================================================
void launch_classic(float* A, float* B, float* C, int n) {
    dim3 threads(S, S);
    dim3 blocks(n/S, n/S);
    MatMulKernel<<<blocks, threads>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

void launch_colcache(float* A, float* B, float* C, int n) {
    // n потоков в блоке, n блоков (по одному на столбец)
    // разделяемая память = n * sizeof(float)
    int threads = min(n, 1024);
    MatMulColCacheKernel<<<n, threads, n*sizeof(float)>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

void launch_tiled(float* A, float* B, float* C, int n) {
    dim3 threads(S, S);
    dim3 blocks(n/S, n/S);
    MatMulTiledKernel<<<blocks, threads>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

// ================================================================
// main
// ================================================================
int main() {
    int n = N;
    size_t bytes = (size_t)n * n * sizeof(float);

    printf("=== Умножение матриц %dx%d на CUDA ===\n", n, n);
    printf("Размер блока S=%d, повторений=%d\n\n", S, REPS);

    // ---- выделение памяти CPU ----
    float* A   = new float[n*n];
    float* B   = new float[n*n];
    float* C_cpu = new float[n*n];
    float* C_gpu = new float[n*n];

    // заполняем случайно
    srand(42);
    for (int i = 0; i < n*n; ++i) {
        A[i] = (float)(rand()%10) / 10.0f;
        B[i] = (float)(rand()%10) / 10.0f;
    }

    // ---- GPU-аналог CPU-умножения (эталон) ----
    printf("Вычисление эталона на CPU...\n");
    matmul_cpu(A, B, C_cpu, n);
    printf("  Готово.\n\n");

    // ---- выделение памяти GPU ----
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);

    cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);

    // CUDA events для замера
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double ops = 2.0 * n * n * n;  // число операций FP

    // ----------------------------------------------------------------
    // Метод 1: Классический
    // ----------------------------------------------------------------
    printf("--- Метод 1: Классический ---\n");
    cudaEventRecord(start);
    for (int r = 0; r < REPS; ++r) {
        dim3 threads(S, S);
        dim3 blocks(n/S, n/S);
        MatMulKernel<<<blocks, threads>>>(A_d, B_d, C_d, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);
    ms1 /= REPS;

    cudaMemcpy(C_gpu, C_d, bytes, cudaMemcpyDeviceToHost);
    bool ok1 = checkResult(C_cpu, C_gpu, n);
    double gflops1 = ops / (ms1 * 1e-3) / 1e9;
    printf("  Время: %.3f мс  Preal: %.3f GFLOP/s  %s\n\n",
           ms1, gflops1, ok1 ? "OK" : "FAIL");

    // ----------------------------------------------------------------
    // Метод 2: Кэширование столбца B
    // ----------------------------------------------------------------
    printf("--- Метод 2: Кэширование столбца B ---\n");
    cudaEventRecord(start);
    for (int r = 0; r < REPS; ++r) {
        int threads_per_block = min(n, 1024);
        MatMulColCacheKernel<<<n, threads_per_block, n*sizeof(float)>>>(A_d, B_d, C_d, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);
    ms2 /= REPS;

    cudaMemcpy(C_gpu, C_d, bytes, cudaMemcpyDeviceToHost);
    bool ok2 = checkResult(C_cpu, C_gpu, n);
    double gflops2 = ops / (ms2 * 1e-3) / 1e9;
    printf("  Время: %.3f мс  Preal: %.3f GFLOP/s  %s\n\n",
           ms2, gflops2, ok2 ? "OK" : "FAIL");

    // ----------------------------------------------------------------
    // Метод 3: Блочный (tiling) с разделяемой памятью
    // ----------------------------------------------------------------
    printf("--- Метод 3: Блочный (tiling S=%d) ---\n", S);
    cudaEventRecord(start);
    for (int r = 0; r < REPS; ++r) {
        dim3 threads(S, S);
        dim3 blocks(n/S, n/S);
        MatMulTiledKernel<<<blocks, threads>>>(A_d, B_d, C_d, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms3;
    cudaEventElapsedTime(&ms3, start, stop);
    ms3 /= REPS;

    cudaMemcpy(C_gpu, C_d, bytes, cudaMemcpyDeviceToHost);
    bool ok3 = checkResult(C_cpu, C_gpu, n);
    double gflops3 = ops / (ms3 * 1e-3) / 1e9;
    printf("  Время: %.3f мс  Preal: %.3f GFLOP/s  %s\n\n",
           ms3, gflops3, ok3 ? "OK" : "FAIL");

    // ----------------------------------------------------------------
    // Итоговая таблица
    // ----------------------------------------------------------------
    printf("=== ИТОГОВАЯ ТАБЛИЦА (N=%d) ===\n", n);
    printf("  %-30s %8.3f мс  %8.3f GFLOP/s\n", "Классический:",      ms1, gflops1);
    printf("  %-30s %8.3f мс  %8.3f GFLOP/s\n", "Кэш столбца B:",     ms2, gflops2);
    printf("  %-30s %8.3f мс  %8.3f GFLOP/s\n", "Блочный (tiling):",  ms3, gflops3);
    printf("\nУскорение блочного vs классического: %.2fx\n", ms1/ms3);

    // проверка ошибок CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("\nCUDA error: %s\n", cudaGetErrorString(err));
    else
        printf("\ncudaGetLastError: No errors\n");

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    delete[] A; delete[] B; delete[] C_cpu; delete[] C_gpu;

    return 0;
}
