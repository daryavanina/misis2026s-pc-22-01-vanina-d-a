// ============================================================
// Лабораторная работа 8.2
// Определение параметров видеокарты CUDA
//
// Компиляция:
//   nvcc lab8_2.cu -o lab8_2
// Запуск:
//   ./lab8_2
// ============================================================

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    cout << "CUDA device count: " << device_count << "\n\n";

    if (device_count == 0) {
        cout << "CUDA-устройства не найдены.\n";
        return 1;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, i);

        cout << "=== GPU " << i << ": " << dp.name << " ===\n\n";

        cout << "--- Память ---\n";
        printf("  Глобальная память:             %lu МБ\n",
               dp.totalGlobalMem / (1024*1024));
        printf("  Память констант:               %lu КБ\n",
               dp.totalConstMem / 1024);
        printf("  Разделяемая память на блок:    %lu КБ\n",
               dp.sharedMemPerBlock / 1024);
        printf("  Кэш L2:                        %d КБ\n",
               dp.l2CacheSize / 1024);
        printf("  Ширина шины памяти:            %d бит\n",
               dp.memoryBusWidth);

        cout << "\n--- Вычисления ---\n";
        printf("  Версия вычислит. возможностей: %d.%d\n",
               dp.major, dp.minor);
        printf("  Число SM (мультипроцессоров):  %d\n",
               dp.multiProcessorCount);
        printf("  Тактовая частота ядра:         %d МГц\n",
               dp.clockRate / 1000);
        printf("  Частота памяти:                %d МГц\n",
               dp.memoryClockRate / 1000);

        cout << "\n--- Потоки и блоки ---\n";
        printf("  Регистров на блок:             %d\n",
               dp.regsPerBlock);
        printf("  Размер WARP:                   %d\n",
               dp.warpSize);
        printf("  Макс. потоков в блоке:         %d\n",
               dp.maxThreadsPerBlock);
        printf("  Макс. размер блока (x,y,z):    %d x %d x %d\n",
               dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
        printf("  Макс. размер сетки (x,y,z):    %d x %d x %d\n",
               dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);

        cout << "\n--- Дополнительно ---\n";
        printf("  Одновременно копир. и вычисл.: %d\n",
               dp.asyncEngineCount);
        printf("  Кол-во асинхр. движков:        %d\n",
               dp.asyncEngineCount);
        printf("  ECC поддержка:                 %s\n",
               dp.ECCEnabled ? "YES" : "NO");

        cout << "\n";
    }

    return 0;
}
