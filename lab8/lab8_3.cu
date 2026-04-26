// ============================================================
// Лабораторная работа 8.3
// Измерение пропускной способности памяти GPU
//
// Компиляция:
//   nvcc lab8_3.cu -o lab8_3
// Запуск:
//   ./lab8_3
// ============================================================

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>

using namespace std;

// размер блока для копирования — 100 МБ
const size_t BLOCK_SIZE = 100 * 1024 * 1024;
const int    REPEATS    = 10;

// ---- замер времени через CUDA events ----
float measureMs(cudaEvent_t& start, cudaEvent_t& stop) {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

double bandwidthGBs(size_t bytes, float totalMs) {
    return (double)bytes / (totalMs / 1000.0) / (1024.0*1024.0*1024.0);
}

int main() {
    printf("CUDA memory bandwidth test (block size = %zu MB)\n\n",
           BLOCK_SIZE / (1024*1024));

    // проверяем наличие GPU
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("%d CUDA device(s) found\n\n", device_count);
    if (device_count == 0) return 1;

    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);
    printf("GPU 0: %s\n\n", dp.name);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ================================================================
    // 1. RAM -> RAM (обычный memcpy)
    // ================================================================
    printf("--- 1. RAM -> RAM (memcpy) ---\n");

    void* h_src = malloc(BLOCK_SIZE);
    void* h_dst = malloc(BLOCK_SIZE);
    memset(h_src, 1, BLOCK_SIZE);
    printf("  RAM allocating... OK\n");

    cudaEventRecord(start);
    for (int r = 0; r < REPEATS; ++r)
        memcpy(h_dst, h_src, BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1 = measureMs(start, stop) / REPEATS;
    printf("  Copying Host -> Host\n");
    printf("  Average bandwidth = %.5f GB/s\n\n", bandwidthGBs(BLOCK_SIZE, ms1));

    free(h_src); free(h_dst);

    // ================================================================
    // 2. CPU -> GPU и GPU -> CPU (обычная память)
    // ================================================================
    printf("--- 2. CPU <-> GPU (обычная malloc) ---\n");

    void* h_buf = malloc(BLOCK_SIZE);
    void* d_buf;
    cudaMalloc(&d_buf, BLOCK_SIZE);
    memset(h_buf, 1, BLOCK_SIZE);
    printf("  RAM allocating... OK\n");
    printf("  GPU global RAM allocating... OK\n");

    // CPU -> GPU
    cudaEventRecord(start);
    for (int r = 0; r < REPEATS; ++r)
        cudaMemcpy(d_buf, h_buf, BLOCK_SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2 = measureMs(start, stop) / REPEATS;
    printf("  Copying Host -> Device\n");
    printf("  Average bandwidth = %.5f GB/s\n\n", bandwidthGBs(BLOCK_SIZE, ms2));

    // GPU -> CPU
    cudaEventRecord(start);
    for (int r = 0; r < REPEATS; ++r)
        cudaMemcpy(h_buf, d_buf, BLOCK_SIZE, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms3 = measureMs(start, stop) / REPEATS;
    printf("  Copying Device -> Host\n");
    printf("  Average bandwidth = %.5f GB/s\n\n", bandwidthGBs(BLOCK_SIZE, ms3));

    free(h_buf);
    cudaFree(d_buf);

    // ================================================================
    // 3. CPU -> GPU и GPU -> CPU (page-locked память)
    // ================================================================
    printf("--- 3. CPU <-> GPU (page-locked cudaMallocHost) ---\n");

    void* h_pinned;
    void* d_buf2;
    cudaMallocHost(&h_pinned, BLOCK_SIZE);
    cudaMalloc(&d_buf2, BLOCK_SIZE);
    memset(h_pinned, 1, BLOCK_SIZE);
    printf("  Page-locked RAM allocating... OK\n");
    printf("  GPU global RAM allocating... OK\n");

    // CPU (pinned) -> GPU
    cudaEventRecord(start);
    for (int r = 0; r < REPEATS; ++r)
        cudaMemcpy(d_buf2, h_pinned, BLOCK_SIZE, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms4 = measureMs(start, stop) / REPEATS;
    printf("  Copying Host -> Device (using page-locked)\n");
    printf("  Average bandwidth = %.5f GB/s\n\n", bandwidthGBs(BLOCK_SIZE, ms4));

    // GPU -> CPU (pinned)
    cudaEventRecord(start);
    for (int r = 0; r < REPEATS; ++r)
        cudaMemcpy(h_pinned, d_buf2, BLOCK_SIZE, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms5 = measureMs(start, stop) / REPEATS;
    printf("  Copying Device -> Host (using page-locked)\n");
    printf("  Average bandwidth = %.5f GB/s\n\n", bandwidthGBs(BLOCK_SIZE, ms5));

    cudaFreeHost(h_pinned);
    cudaFree(d_buf2);

    // ================================================================
    // 4. GPU -> GPU (внутри видеопамяти)
    // ================================================================
    printf("--- 4. GPU -> GPU (внутри глобальной памяти) ---\n");

    void* d_src;
    void* d_dst;
    cudaMalloc(&d_src, BLOCK_SIZE);
    cudaMalloc(&d_dst, BLOCK_SIZE);
    printf("  GPU global RAM allocating... OK\n");

    cudaEventRecord(start);
    for (int r = 0; r < REPEATS; ++r) {
        cudaMemcpy(d_dst, d_src, BLOCK_SIZE, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms6 = measureMs(start, stop) / REPEATS;
    printf("  Copying Device -> Device\n");
    printf("  Average bandwidth = %.5f GB/s\n\n", bandwidthGBs(BLOCK_SIZE, ms6));

    cudaFree(d_src);
    cudaFree(d_dst);

    // ================================================================
    // Итоговая таблица
    // ================================================================
    printf("=== ИТОГОВАЯ ТАБЛИЦА ===\n");
    printf("  %-40s %.3f GB/s\n", "RAM -> RAM (memcpy):",               bandwidthGBs(BLOCK_SIZE, ms1));
    printf("  %-40s %.3f GB/s\n", "CPU -> GPU (malloc):",               bandwidthGBs(BLOCK_SIZE, ms2));
    printf("  %-40s %.3f GB/s\n", "GPU -> CPU (malloc):",               bandwidthGBs(BLOCK_SIZE, ms3));
    printf("  %-40s %.3f GB/s\n", "CPU -> GPU (page-locked):",          bandwidthGBs(BLOCK_SIZE, ms4));
    printf("  %-40s %.3f GB/s\n", "GPU -> CPU (page-locked):",          bandwidthGBs(BLOCK_SIZE, ms5));
    printf("  %-40s %.3f GB/s\n", "GPU -> GPU (device to device):",     bandwidthGBs(BLOCK_SIZE, ms6));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nDone\n");
    return 0;
}
