#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <windows.h>
#include <intrin.h>

static volatile int sink = 0;

const int N = 2000;
const int K = 10;

// ---- массив ----

void fillRandom(int* a, int n) {
    for (int i = 0; i < n; ++i) a[i] = rand() % 100000;
}

void bubbleSort(int* a, int n) {
    bool changed;
    do {
        changed = false;
        for (int i = 0; i < n - 1; ++i) {
            if (a[i] > a[i + 1]) {
                int t = a[i]; 
                a[i] = a[i + 1]; 
                a[i + 1] = t;
                changed = true;
            }
        }
    } while (changed);
}

// ---- статистика ----

double meanArr(const double* a, int n) {
    long double s = 0;
    for (int i = 0; i < n; ++i) s += a[i];
    return (double)(s / n);
}

double varianceK(const double* a, int n, double avg) {
    long double s = 0;
    for (int i = 0; i < n; ++i) {
        long double d = (long double)a[i] - (long double)avg;
        s += d * d;
    }
    return (double)(s / n);
}

double minArr(const double* a, int n) {
    double mn = a[0];
    for (int i = 1; i < n; ++i) if (a[i] < mn) mn = a[i];
    return mn;
}

double confDelta95(double s, int n) {
    return 1.96 * s / std::sqrt((double)n);
}

// фильтрация 3-sigma, возвращает новый размер
int filter3Sigma(const double* a, int n, double* out) {
    double avg = meanArr(a, n);
    double sigma = std::sqrt(varianceK(a, n, avg));
    int cnt = 0;
    for (int i = 0; i < n; ++i)
        if (std::fabs(a[i] - avg) <= 3.0 * sigma) out[cnt++] = a[i];
    return cnt;
}

// метод минимального окна
void minWindow(const double* a, int n, double p, double& lo, double& hi, double& avg) {
    double* sorted = new double[n];
    for (int i = 0; i < n; ++i) sorted[i] = a[i];
    // простая сортировка вставками
    for (int i = 1; i < n; ++i) {
        double key = sorted[i]; int j = i - 1;
        while (j >= 0 && sorted[j] > key) { sorted[j + 1] = sorted[j]; --j; }
        sorted[j + 1] = key;
    }
    int I = (int)std::ceil(n * p);
    if (I < 1) I = 1; if (I > n) I = n;
    int best = 0;
    double bestW = sorted[I - 1] - sorted[0];
    for (int i = 1; i + I - 1 < n; ++i) {
        double w = sorted[i + I - 1] - sorted[i];
        if (w < bestW) { bestW = w; best = i; }
    }
    lo = sorted[best]; hi = sorted[best + I - 1];
    long double s = 0;
    for (int i = best; i < best + I; ++i) s += sorted[i];
    avg = (double)(s / I);
    delete[] sorted;
}

// ---- вывод ----

void printStats(const char* name, const double* times, int k) {
    double filtered[K];
    int kf = filter3Sigma(times, k, filtered);

    double tmin   = minArr(times, k);
    double avg    = meanArr(times, k);
    double s      = std::sqrt(varianceK(times, k, avg));
    double delta  = confDelta95(s, k);

    double avg2   = meanArr(filtered, kf);
    double s2     = std::sqrt(varianceK(filtered, kf, avg2));
    double delta2 = confDelta95(s2, kf);

    double lo, hi, avgW;
    minWindow(times, k, 0.95, lo, hi, avgW);

    std::cout << name << ":\n";
    std::cout << "  t_min   = " << tmin << " ms\n";
    std::cout << "  t_avg   = " << avg  << " ms,  CI95: ["
              << (avg - delta) << "; " << (avg + delta) << "],  delta=" << delta << " ms\n";
    std::cout << "  t_avg'  = " << avg2 << " ms,  CI95: ["
              << (avg2 - delta2) << "; " << (avg2 + delta2) << "],  delta=" << delta2
              << " ms  (3sigma, n=" << kf << ")\n";
    std::cout << "  t_avg'' = " << avgW << " ms,  window: [" << lo << "; " << hi << "]  (p=0.95)\n";
}

void printAll(const char* name, const double* a, int n) {
    std::cout << name << ": ";
    for (int i = 0; i < n; ++i) {
        std::cout << a[i] << " ms";
        if (i + 1 < n) std::cout << ", ";
    }
    std::cout << "\n";
}

// ---- JSON ----

void writeJson(const double* gtc, const double* tsc, const double* qpc, int k) {
    std::ofstream out("C:/Users/darya/Desktop/misis/8sem/pc/misis2026s-pc-22-01-vanina-d-a/times.json");
    out << "{\n";
    out << "  \"K\": " << k << ",\n";
    out << "  \"N\": " << N << ",\n";
    auto writeArr = [&](const char* key, const double* a, bool comma) {
        out << "  \"" << key << "\": [";
        for (int i = 0; i < k; ++i) {
            out << std::setprecision(17) << a[i];
            if (i + 1 < k) out << ", ";
        }
        out << "]";
        if (comma) out << ",";
        out << "\n";
    };
    writeArr("GetTickCount_ms", gtc, true);
    writeArr("RDTSC_ms",        tsc, true);
    writeArr("QPC_ms",          qpc, false);
    out << "}\n";
    out.close();
}

// ---- замеры ----

double measureGTC() {
    int* a = new int[N]; fillRandom(a, N);
    DWORD t0 = GetTickCount();
    bubbleSort(a, N);
    DWORD t1 = GetTickCount();
    sink = a[0];
    delete[] a;
    return (double)(t1 - t0);
}

double measureRDTSC() {
    int* a = new int[N]; fillRandom(a, N);
    unsigned __int64 t0 = __rdtsc();
    bubbleSort(a, N);
    unsigned __int64 t1 = __rdtsc();
    sink = a[0];
    delete[] a;
    // частота: считаем такты за Sleep(100 мс)
    unsigned __int64 s0 = __rdtsc(); Sleep(100); unsigned __int64 s1 = __rdtsc();
    double freq_ms = (double)(s1 - s0) / 100.0; // тактов в мс
    return (double)(t1 - t0) / freq_ms;
}

double measureQPC() {
    int* a = new int[N]; fillRandom(a, N);
    __int64 t0, t1, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&t0);
    bubbleSort(a, N);
    QueryPerformanceCounter((LARGE_INTEGER*)&t1);
    sink = a[0];
    delete[] a;
    return (double)(t1 - t0) / (double)freq * 1000.0;
}

// ---- main ----

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    srand((unsigned)time(nullptr));

    // прогрев
    { int* w = new int[N]; fillRandom(w, N); bubbleSort(w, N); sink = w[0]; delete[] w; }

    double gtc_times[K], tsc_times[K], qpc_times[K];

    for (int i = 0; i < K; ++i) {
        gtc_times[i] = measureGTC();
        tsc_times[i] = measureRDTSC();
        qpc_times[i] = measureQPC();
    }

    std::cout << "K=" << K << ", N=" << N << "\n\n";

    printAll("GetTickCount", gtc_times, K);
    printAll("RDTSC       ", tsc_times, K);
    printAll("QPC         ", qpc_times, K);
    std::cout << "\n";

    printStats("GetTickCount", gtc_times, K);
    std::cout << "\n";
    printStats("RDTSC",        tsc_times, K);
    std::cout << "\n";
    printStats("QPC",          qpc_times, K);
    std::cout << "\n";

    std::cout << "sink=" << sink << "\n";

    writeJson(gtc_times, tsc_times, qpc_times, K);
    std::cout << "times.json saved\n";

    return 0;
}
