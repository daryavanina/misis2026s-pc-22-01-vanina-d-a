#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <vector>
#include <string>
#include <windows.h>

static volatile float sink = 0.0f;

// ---- конфигурация перебора ----

const int    FIXED_N   = 512;
const int    FIXED_S   = 64;
const int    FIXED_M   = 4;
const int    M_VALUES[] = { 1, 2, 4, 8, 16 };
const int    S_VALUES[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
const int    N_VALUES[] = { 32, 64, 128, 256, 512, 1024, 2048 };
const int    N_SWEEP_M  = 8;   // оптимальное M для sweep по N
const int    N_SWEEP_S  = 64;  // оптимальный S для sweep по N

// ---- матрицы ----

float* allocMat(int n) { return new float[(size_t)n * n](); }
void   freeMat(float* m) { delete[] m; }

void fillRandom(float* m, int n) {
    for (int i = 0; i < n * n; ++i)
        m[i] = (float)(rand() % 100) / 10.0f;
}

void zeroMat(float* m, int n) {
    memset(m, 0, (size_t)n * n * sizeof(float));
}

bool matsEqual(const float* a, const float* b, int n, float eps = 0.5f) {
    for (int i = 0; i < n * n; ++i)
        if (std::fabs(a[i] - b[i]) > eps) return false;
    return true;
}

// ---- алгоритмы ----

void mulClassic(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int k = 0; k < n; ++k)
                s += A[(size_t)i*n+k] * B[(size_t)k*n+j];
            C[(size_t)i*n+j] = s;
        }
}

void transposeInto(const float* B, float* Bt, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Bt[(size_t)j*n+i] = B[(size_t)i*n+j];
}

void mulTransposeNoT(const float* A, const float* Bt, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int k = 0; k < n; ++k)
                s += A[(size_t)i*n+k] * Bt[(size_t)j*n+k];
            C[(size_t)i*n+j] = s;
        }
}

void mulTranspose(const float* A, const float* B, float* C, int n) {
    float* Bt = allocMat(n);
    transposeInto(B, Bt, n);
    mulTransposeNoT(A, Bt, C, n);
    freeMat(Bt);
}

void mulBuffered(const float* A, const float* B, float* C, int n, int m) {
    float* tmp = new float[n];
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) tmp[k] = B[(size_t)k*n+j];
        for (int i = 0; i < n; ++i) {
            float s = 0.0f;
            int k = 0;
            for (; k <= n - m; k += m) {
                float acc = 0.0f;
                for (int u = 0; u < m; ++u)
                    acc += A[(size_t)i*n+k+u] * tmp[k+u];
                s += acc;
            }
            for (; k < n; ++k) s += A[(size_t)i*n+k] * tmp[k];
            C[(size_t)i*n+j] = s;
        }
    }
    delete[] tmp;
}

void mulBlock(const float* A, const float* B, float* C, int n, int s, int m) {
    zeroMat(C, n);
    float* tA = new float[(size_t)s * s];
    float* tB = new float[(size_t)s * s];
    for (int ii = 0; ii < n; ii += s)
    for (int jj = 0; jj < n; jj += s)
    for (int kk = 0; kk < n; kk += s) {
        int si = (ii+s<=n)?s:n-ii;
        int sj = (jj+s<=n)?s:n-jj;
        int sk = (kk+s<=n)?s:n-kk;
        for (int i=0;i<si;++i)
            for (int k=0;k<sk;++k)
                tA[(size_t)i*sk+k] = A[(size_t)(ii+i)*n+(kk+k)];
        for (int k=0;k<sk;++k)
            for (int j=0;j<sj;++j)
                tB[(size_t)j*sk+k] = B[(size_t)(kk+k)*n+(jj+j)];
        for (int i=0;i<si;++i)
        for (int j=0;j<sj;++j) {
            float acc = 0.0f;
            int k = 0;
            for (; k<=sk-m; k+=m) {
                float a2 = 0.0f;
                for (int u=0;u<m;++u)
                    a2 += tA[(size_t)i*sk+k+u] * tB[(size_t)j*sk+k+u];
                acc += a2;
            }
            for (; k<sk; ++k) acc += tA[(size_t)i*sk+k]*tB[(size_t)j*sk+k];
            C[(size_t)(ii+i)*n+(jj+j)] += acc;
        }
    }
    delete[] tA;
    delete[] tB;
}

// ---- замер QPC ----

double qpcMs(__int64 t0, __int64 t1, __int64 freq) {
    return (double)(t1 - t0) / (double)freq * 1000.0;
}

double gflops(int n, double ms) {
    return 2.0 * (double)n * (double)n * (double)n / (ms * 1e-3) / 1e9;
}

// ---- JSON-writer (без сторонних библиотек) ----

struct JsonWriter {
    std::ofstream f;
    bool firstEntry = true;

    JsonWriter(const char* path) : f(path) {
        f << std::setprecision(6) << std::fixed;
        f << "{\n";
    }

    void beginKey(const char* key) {
        if (!firstEntry) f << ",\n";
        firstEntry = false;
        f << "  \"" << key << "\": ";
    }

    void writeDouble(const char* key, double v) {
        beginKey(key); f << v;
    }

    void writeInt(const char* key, int v) {
        beginKey(key); f << v;
    }

    void writeString(const char* key, const char* v) {
        beginKey(key); f << "\"" << v << "\"";
    }

    // записывает массив double
    void writeArr(const char* key, const std::vector<double>& a) {
        beginKey(key);
        f << "[";
        for (size_t i = 0; i < a.size(); ++i) {
            f << a[i];
            if (i+1 < a.size()) f << ", ";
        }
        f << "]";
    }

    // записывает массив int
    void writeArrInt(const char* key, const std::vector<int>& a) {
        beginKey(key);
        f << "[";
        for (size_t i = 0; i < a.size(); ++i) {
            f << a[i];
            if (i+1 < a.size()) f << ", ";
        }
        f << "]";
    }

    void close() { f << "\n}\n"; f.close(); }
};

// ---- прогрев ----

void warmup(int n) {
    float* A = allocMat(n); float* B = allocMat(n); float* C = allocMat(n);
    fillRandom(A, n); fillRandom(B, n);
    mulClassic(A, B, C, n);
    sink = C[0];
    freeMat(A); freeMat(B); freeMat(C);
}

// ---- вывод прогресса ----

void progress(const char* label) {
    std::cout << "  " << label << "..." << std::flush;
}

void done(double ms, int n) {
    std::cout << " " << ms << " ms  (" << gflops(n, ms) << " GFLOP/s)\n";
}

// ---- main ----

int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    srand((unsigned)time(nullptr));

    __int64 freq, t0, t1;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    std::cout << "=== Прогрев ===\n";
    warmup(FIXED_N);
    std::cout << "  done\n\n";

    JsonWriter jw("lab2/lab2_results.json");
    //JsonWriter jw("C:/Users/darya/Desktop/misis/8sem/pc/misis2026s-pc-22-01-vanina-d-a/lab2/lab2_results.json");
    jw.writeInt("fixed_N", FIXED_N);
    jw.writeInt("fixed_S", FIXED_S);
    jw.writeInt("fixed_M", FIXED_M);

    std::cout << "=== Все алгоритмы (N=" << FIXED_N << ", S=" << FIXED_S << ", M=" << FIXED_M << ") ===\n";
    {
        int n = FIXED_N, s = FIXED_S, m = FIXED_M;
        float* A  = allocMat(n); float* B  = allocMat(n);
        float* Bt = allocMat(n);
        float* C1 = allocMat(n); float* C2 = allocMat(n);
        float* C3 = allocMat(n); float* C4 = allocMat(n);
        fillRandom(A, n); fillRandom(B, n);
        transposeInto(B, Bt, n);

        progress("classic");
        QueryPerformanceCounter((LARGE_INTEGER*)&t0);
        mulClassic(A, B, C1, n);
        QueryPerformanceCounter((LARGE_INTEGER*)&t1);
        double t_cl = qpcMs(t0, t1, freq);
        sink = C1[0]; done(t_cl, n);

        progress("transpose (with T)");
        QueryPerformanceCounter((LARGE_INTEGER*)&t0);
        mulTranspose(A, B, C2, n);
        QueryPerformanceCounter((LARGE_INTEGER*)&t1);
        double t_tr = qpcMs(t0, t1, freq);
        sink = C2[0]; done(t_tr, n);

        progress("transpose (no T)");
        QueryPerformanceCounter((LARGE_INTEGER*)&t0);
        mulTransposeNoT(A, Bt, C2, n);
        QueryPerformanceCounter((LARGE_INTEGER*)&t1);
        double t_trN = qpcMs(t0, t1, freq);
        sink = C2[0]; done(t_trN, n);

        progress("buffered");
        QueryPerformanceCounter((LARGE_INTEGER*)&t0);
        mulBuffered(A, B, C3, n, m);
        QueryPerformanceCounter((LARGE_INTEGER*)&t1);
        double t_buf = qpcMs(t0, t1, freq);
        sink = C3[0]; done(t_buf, n);

        progress("block");
        QueryPerformanceCounter((LARGE_INTEGER*)&t0);
        mulBlock(A, B, C4, n, s, m);
        QueryPerformanceCounter((LARGE_INTEGER*)&t1);
        double t_blk = qpcMs(t0, t1, freq);
        sink = C4[0]; done(t_blk, n);

        std::cout << "  Проверка: classic==transpose:" << (matsEqual(C1,C2,n)?"OK":"FAIL")
                  << "  classic==buffered:" << (matsEqual(C1,C3,n)?"OK":"FAIL")
                  << "  classic==block:"    << (matsEqual(C1,C4,n)?"OK":"FAIL") << "\n\n";

        jw.writeDouble("classic_ms",       t_cl);
        jw.writeDouble("transpose_ms",     t_tr);
        jw.writeDouble("transpose_noT_ms", t_trN);
        jw.writeDouble("buffered_ms",      t_buf);
        jw.writeDouble("block_ms",         t_blk);

        freeMat(A); freeMat(B); freeMat(Bt);
        freeMat(C1); freeMat(C2); freeMat(C3); freeMat(C4);
    }

    // ================================================================
    // Буферизация: перебор M
    // ================================================================
    std::cout << "=== Буферизация: перебор M (N=" << FIXED_N << ") ===\n";
    {
        int n = FIXED_N;
        float* A = allocMat(n); float* B = allocMat(n); float* C = allocMat(n);
        fillRandom(A, n); fillRandom(B, n);

        std::vector<double> times;
        std::vector<int> mvals(std::begin(M_VALUES), std::end(M_VALUES));
        for (int m : mvals) {
            char lbl[32]; sprintf(lbl, "M=%d", m);
            progress(lbl);
            QueryPerformanceCounter((LARGE_INTEGER*)&t0);
            mulBuffered(A, B, C, n, m);
            QueryPerformanceCounter((LARGE_INTEGER*)&t1);
            double t = qpcMs(t0, t1, freq);
            sink = C[0]; times.push_back(t); done(t, n);
        }
        jw.writeArrInt("buffered_M_values", mvals);
        jw.writeArr("buffered_M_times_ms", times);
        std::cout << "\n";
        freeMat(A); freeMat(B); freeMat(C);
    }

    // ================================================================
    // Блочное: перебор S
    // ================================================================
    std::cout << "=== Блочное: перебор S (N=" << FIXED_N << ", M=" << FIXED_M << ") ===\n";
    {
        int n = FIXED_N, m = FIXED_M;
        float* A = allocMat(n); float* B = allocMat(n); float* C = allocMat(n);
        fillRandom(A, n); fillRandom(B, n);

        std::vector<double> times;
        std::vector<int> svals(std::begin(S_VALUES), std::end(S_VALUES));
        for (int s : svals) {
            char lbl[32]; sprintf(lbl, "S=%d", s);
            progress(lbl);
            QueryPerformanceCounter((LARGE_INTEGER*)&t0);
            mulBlock(A, B, C, n, s, m);
            QueryPerformanceCounter((LARGE_INTEGER*)&t1);
            double t = qpcMs(t0, t1, freq);
            sink = C[0]; times.push_back(t); done(t, n);
        }
        jw.writeArrInt("block_S_values", svals);
        jw.writeArr("block_S_times_ms", times);
        std::cout << "\n";
        freeMat(A); freeMat(B); freeMat(C);
    }

    // ================================================================
    // Блочное: перебор M (при FIXED_S)
    // ================================================================
    std::cout << "=== Блочное: перебор M (N=" << FIXED_N << ", S=" << FIXED_S << ") ===\n";
    {
        int n = FIXED_N, s = FIXED_S;
        float* A = allocMat(n); float* B = allocMat(n); float* C = allocMat(n);
        fillRandom(A, n); fillRandom(B, n);

        std::vector<double> times;
        std::vector<int> mvals(std::begin(M_VALUES), std::end(M_VALUES));
        for (int m : mvals) {
            char lbl[32]; sprintf(lbl, "M=%d", m);
            progress(lbl);
            QueryPerformanceCounter((LARGE_INTEGER*)&t0);
            mulBlock(A, B, C, n, s, m);
            QueryPerformanceCounter((LARGE_INTEGER*)&t1);
            double t = qpcMs(t0, t1, freq);
            sink = C[0]; times.push_back(t); done(t, n);
        }
        jw.writeArrInt("block_M_values", mvals);
        jw.writeArr("block_M_times_ms", times);
        std::cout << "\n";
        freeMat(A); freeMat(B); freeMat(C);
    }

    // ================================================================
    // Sweep по N: все 4 алгоритма с оптимальными параметрами
    // ================================================================
    std::cout << "=== Sweep по N (S=" << N_SWEEP_S << ", M=" << N_SWEEP_M << ") ===\n";
    {
        std::vector<int>    nvals(std::begin(N_VALUES), std::end(N_VALUES));
        std::vector<double> cl_t, tr_t, buf_t, blk_t;

        for (int n : nvals) {
            std::cout << "  N=" << n << "\n";
            float* A  = allocMat(n); float* B  = allocMat(n);
            float* Bt = allocMat(n);
            float* C1 = allocMat(n); float* C2 = allocMat(n);
            float* C3 = allocMat(n); float* C4 = allocMat(n);
            fillRandom(A, n); fillRandom(B, n);
            transposeInto(B, Bt, n);

            progress("  classic");
            QueryPerformanceCounter((LARGE_INTEGER*)&t0);
            mulClassic(A, B, C1, n);
            QueryPerformanceCounter((LARGE_INTEGER*)&t1);
            double t = qpcMs(t0, t1, freq);
            cl_t.push_back(t); sink = C1[0]; done(t, n);

            progress("  transpose");
            QueryPerformanceCounter((LARGE_INTEGER*)&t0);
            mulTransposeNoT(A, Bt, C2, n);
            QueryPerformanceCounter((LARGE_INTEGER*)&t1);
            t = qpcMs(t0, t1, freq);
            tr_t.push_back(t); sink = C2[0]; done(t, n);

            progress("  buffered");
            QueryPerformanceCounter((LARGE_INTEGER*)&t0);
            mulBuffered(A, B, C3, n, N_SWEEP_M);
            QueryPerformanceCounter((LARGE_INTEGER*)&t1);
            t = qpcMs(t0, t1, freq);
            buf_t.push_back(t); sink = C3[0]; done(t, n);

            progress("  block");
            QueryPerformanceCounter((LARGE_INTEGER*)&t0);
            mulBlock(A, B, C4, n, N_SWEEP_S, N_SWEEP_M);
            QueryPerformanceCounter((LARGE_INTEGER*)&t1);
            t = qpcMs(t0, t1, freq);
            blk_t.push_back(t); sink = C4[0]; done(t, n);

            freeMat(A); freeMat(B); freeMat(Bt);
            freeMat(C1); freeMat(C2); freeMat(C3); freeMat(C4);
        }
        jw.writeArrInt("sweep_N_values",          nvals);
        jw.writeArr("sweep_classic_ms",    cl_t);
        jw.writeArr("sweep_transpose_ms",  tr_t);
        jw.writeArr("sweep_buffered_ms",   buf_t);
        jw.writeArr("sweep_block_ms",      blk_t);
    }

    jw.close();
    std::cout << "\nlab2_results.json saved\n";
    std::cout << "sink=" << sink << "\n";

    return 0;
}
