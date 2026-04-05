#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <intrin.h>
#include <windows.h>

static volatile long long sink = 0;

// ---- имя выходного файла ----

const char* JSON_FILE = "C:/Users/darya/Desktop/misis/8sem/pc/misis2026s-pc-22-01-vanina-d-a/lab4/latency_results.json";

// ---- частота процессора через RDTSC + Sleep ----

double freqMHz() {
    unsigned __int64 t0 = __rdtsc();
    Sleep(200);
    unsigned __int64 t1 = __rdtsc();
    return (double)(t1 - t0) / 200000.0;  // тактов/мкс = МГц
}

// ---- один замер: возвращает тактов на итерацию ----

// режим 1: последовательный обход
double measureSeq(int* a, size_t n, int repeats) {
    long long s = 0;
    unsigned __int64 t0 = __rdtsc();
    for (int r = 0; r < repeats; ++r)
        for (size_t i = 0; i < n; ++i)
            s += a[i];
    unsigned __int64 t1 = __rdtsc();
    sink = s;
    return (double)(t1 - t0) / ((double)n * repeats);
}

// режим 2: случайный обход (rand() внутри цикла)
double measureRand(int* a, size_t n, int repeats) {
    long long s = 0;
    unsigned __int64 t0 = __rdtsc();
    for (int r = 0; r < repeats; ++r)
        for (size_t i = 0; i < n; ++i)
            s += a[(unsigned int)rand() % n];
    unsigned __int64 t1 = __rdtsc();
    sink = s;
    return (double)(t1 - t0) / ((double)n * repeats);
}

// режим 3: случайный с массивом индексов (rand() вне цикла)
double measureRandIdx(int* a, size_t n, unsigned int* idx, int repeats) {
    long long s = 0;
    unsigned __int64 t0 = __rdtsc();
    for (int r = 0; r < repeats; ++r)
        for (size_t i = 0; i < n; ++i)
            s += a[idx[i]];
    unsigned __int64 t1 = __rdtsc();
    sink = s;
    return (double)(t1 - t0) / ((double)n * repeats);
}

// ---- JSON-запись ----

struct Json {
    std::ofstream f;
    bool firstSection = true;

    Json(const char* path) : f(path) { f << std::fixed << std::setprecision(4); f << "{\n"; }

    void beginArr(const char* key) {
        if (!firstSection) f << ",\n";
        firstSection = false;
        f << "  \"" << key << "\": [\n";
    }

    void writePoint(size_t sizeKb, double seq, double rnd, double idx, bool last) {
        f << "    {\"kb\":" << sizeKb
          << ",\"seq\":" << seq
          << ",\"rnd\":" << rnd
          << ",\"idx\":" << idx << "}";
        if (!last) f << ",";
        f << "\n";
    }

    void endArr() { f << "  ]"; }
    void close()  { f << "\n}\n"; f.close(); }
};

// ---- sweep по диапазону размеров ----

struct SweepConfig {
    size_t startKb;
    size_t endKb;
    size_t stepKb;
    int    repeats;
    const char* label;
};

void runSweep(Json& js, const SweepConfig& cfg) {
    std::cout << "\n=== " << cfg.label << " ===\n";
    std::cout << std::left << std::setw(10) << "Size(KB)"
              << std::setw(12) << "Seq(clk)"
              << std::setw(12) << "Rand(clk)"
              << std::setw(12) << "RandIdx(clk)" << "\n";
    std::cout << std::string(46, '-') << "\n";

    // считаем число точек
    size_t nPoints = (cfg.endKb - cfg.startKb) / cfg.stepKb + 1;
    size_t pt = 0;

    js.beginArr(cfg.label);

    for (size_t sizeKb = cfg.startKb; sizeKb <= cfg.endKb; sizeKb += cfg.stepKb) {
        size_t n = sizeKb * 1024 / sizeof(int);
        if (n < 1) n = 1;

        // выделяем массив данных и массив индексов
        int*          a   = new int[n];
        unsigned int* idx = new unsigned int[n];

        // заполняем случайными значениями
        for (size_t i = 0; i < n; ++i) a[i]   = rand() % 1000;
        for (size_t i = 0; i < n; ++i) idx[i] = (unsigned int)((unsigned int)rand() % (unsigned int)n);

        double seq = measureSeq    (a, n, cfg.repeats);
        double rnd = measureRand   (a, n, cfg.repeats);
        double rix = measureRandIdx(a, n, idx, cfg.repeats);

        bool last = (sizeKb + cfg.stepKb > cfg.endKb);
        js.writePoint(sizeKb, seq, rnd, rix, last);

        std::cout << std::left  << std::setw(10) << sizeKb
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << seq
                  << std::setw(12) << rnd
                  << std::setw(12) << rix << "\n";

        delete[] a;
        delete[] idx;
        ++pt;
    }

    js.endArr();
}

// ---- main ----

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    srand((unsigned)time(nullptr));

    std::cout << "Определение частоты процессора...\n";
    double mhz = freqMHz();
    std::cout << "  ~" << (int)mhz << " МГц\n";

    Json js(JSON_FILE);

    // диапазон 1: 1 КБ → 2 МБ, шаг 1 КБ  (L1d и L2)
    runSweep(js, {1, 2048, 1, 20, "range1_L1_L2"});

    // диапазон 2: 1 МБ → 32 МБ, шаг 512 КБ  (L2 и L3)
    runSweep(js, {1024, 32768, 512, 5, "range2_L2_L3"});

    // диапазон 3: 5 МБ → 150 МБ, шаг 5 МБ  (L3 и RAM)
    runSweep(js, {5120, 153600, 5120, 2, "range3_L3_RAM"});

    js.close();
    std::cout << "\n" << JSON_FILE << " saved\n";
    std::cout << "sink=" << sink << "\n";
    return 0;
}
