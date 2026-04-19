#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <immintrin.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <windows.h>
#include <intrin.h>

// ============================================================
// Вариант 5: длина вектора int8
//   l = sqrt( sum( a[i]*a[i] ) )
// Схема: MMX (int8 -> int16 -> int32) -> SSE (sqrtf)
//        AVX-512 (int8 -> int16 -> int32 -> reduce -> sqrtf)
// ============================================================

static volatile float sink_f = 0.0f;

const int N = 1000000;   // кратно 64 для AVX-512 и 8 для MMX

const char* JSON_FILE = "C:/Users/darya/Desktop/misis/8sem/pc/misis2026s-pc-22-01-vanina-d-a/lab5/lab5_results.json";

// ---- замер QPC ----

double qpcNow() {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart * 1000.0;
}

// ================================================================
// 1. Обычный C++
// ================================================================
float lengthScalar(const signed char* a, int n) {
    long long s = 0;
    for (int i = 0; i < n; ++i)
        s += (long long)a[i] * a[i];
    return sqrtf((float)s);
}

// ================================================================
// 2. MMX скалярный (по 8 байт, без раскрутки)
//    int8 -> знаковое расширение в int16 через PUNPCKLBW/PUNPCKHBW
//    -> PMADDWD (int16*int16 -> int32, пары складываются)
//    -> накопление в int32
// ================================================================
float lengthMMX_scalar(const signed char* a, int n) {
    __m64 acc  = _mm_setzero_si64();
    // маска для знакового расширения: нам нужен _mm_cvtsi32_si64 с байтами
    // используем PSRAW для получения знака
    for (int i = 0; i < n; i += 8) {
        __m64 v = *reinterpret_cast<const __m64*>(a + i);
        // знаковое расширение: сдвиг влево на 8, потом арифм. вправо на 8
        __m64 lo = _mm_srai_pi16(_mm_unpacklo_pi8(_mm_setzero_si64(), v), 8);
        __m64 hi = _mm_srai_pi16(_mm_unpackhi_pi8(_mm_setzero_si64(), v), 8);
        acc = _mm_add_pi32(acc, _mm_madd_pi16(lo, lo));
        acc = _mm_add_pi32(acc, _mm_madd_pi16(hi, hi));
    }
    // редукция: два int32 в acc -> сложить
    __m64 hi64 = _mm_unpackhi_pi32(acc, _mm_setzero_si64());
    __m64 sum  = _mm_add_pi32(acc, hi64);
    int s = _mm_cvtsi64_si32(sum);
    _mm_empty();
    return sqrtf((float)s);
}

// ================================================================
// 3. MMX векторный x2
// ================================================================
float lengthMMX_x2(const signed char* a, int n) {
    __m64 acc0 = _mm_setzero_si64(), acc1 = _mm_setzero_si64();
    for (int i = 0; i < n; i += 16) {
        __m64 v0 = *reinterpret_cast<const __m64*>(a + i);
        __m64 lo0 = _mm_srai_pi16(_mm_unpacklo_pi8(_mm_setzero_si64(), v0), 8);
        __m64 hi0 = _mm_srai_pi16(_mm_unpackhi_pi8(_mm_setzero_si64(), v0), 8);
        acc0 = _mm_add_pi32(acc0, _mm_madd_pi16(lo0, lo0));
        acc0 = _mm_add_pi32(acc0, _mm_madd_pi16(hi0, hi0));

        __m64 v1 = *reinterpret_cast<const __m64*>(a + i + 8);
        __m64 lo1 = _mm_srai_pi16(_mm_unpacklo_pi8(_mm_setzero_si64(), v1), 8);
        __m64 hi1 = _mm_srai_pi16(_mm_unpackhi_pi8(_mm_setzero_si64(), v1), 8);
        acc1 = _mm_add_pi32(acc1, _mm_madd_pi16(lo1, lo1));
        acc1 = _mm_add_pi32(acc1, _mm_madd_pi16(hi1, hi1));
    }
    __m64 acc = _mm_add_pi32(acc0, acc1);
    __m64 hi64 = _mm_unpackhi_pi32(acc, _mm_setzero_si64());
    acc = _mm_add_pi32(acc, hi64);
    int s = _mm_cvtsi64_si32(acc);
    _mm_empty();
    return sqrtf((float)s);
}

// ================================================================
// 4. MMX векторный x4
// ================================================================
float lengthMMX_x4(const signed char* a, int n) {
    __m64 acc0=_mm_setzero_si64(), acc1=_mm_setzero_si64();
    __m64 acc2=_mm_setzero_si64(), acc3=_mm_setzero_si64();
    for (int i = 0; i < n; i += 32) {
        auto step = [&](__m64& acc, int off) {
            __m64 v  = *reinterpret_cast<const __m64*>(a + i + off);
            __m64 lo = _mm_srai_pi16(_mm_unpacklo_pi8(_mm_setzero_si64(), v), 8);
            __m64 hi = _mm_srai_pi16(_mm_unpackhi_pi8(_mm_setzero_si64(), v), 8);
            acc = _mm_add_pi32(acc, _mm_madd_pi16(lo, lo));
            acc = _mm_add_pi32(acc, _mm_madd_pi16(hi, hi));
        };
        step(acc0, 0); step(acc1, 8); step(acc2, 16); step(acc3, 24);
    }
    __m64 acc = _mm_add_pi32(_mm_add_pi32(acc0,acc1), _mm_add_pi32(acc2,acc3));
    __m64 hi64 = _mm_unpackhi_pi32(acc, _mm_setzero_si64());
    acc = _mm_add_pi32(acc, hi64);
    int s = _mm_cvtsi64_si32(acc);
    _mm_empty();
    return sqrtf((float)s);
}

// ================================================================
// 5. MMX векторный x8
// ================================================================
float lengthMMX_x8(const signed char* a, int n) {
    __m64 acc[8];
    for (int k = 0; k < 8; ++k) acc[k] = _mm_setzero_si64();
    for (int i = 0; i < n; i += 64) {
        for (int k = 0; k < 8; ++k) {
            __m64 v  = *reinterpret_cast<const __m64*>(a + i + k*8);
            __m64 lo = _mm_srai_pi16(_mm_unpacklo_pi8(_mm_setzero_si64(), v), 8);
            __m64 hi = _mm_srai_pi16(_mm_unpackhi_pi8(_mm_setzero_si64(), v), 8);
            acc[k] = _mm_add_pi32(acc[k], _mm_madd_pi16(lo, lo));
            acc[k] = _mm_add_pi32(acc[k], _mm_madd_pi16(hi, hi));
        }
    }
    __m64 total = _mm_setzero_si64();
    for (int k = 0; k < 8; ++k) total = _mm_add_pi32(total, acc[k]);
    __m64 hi64 = _mm_unpackhi_pi32(total, _mm_setzero_si64());
    total = _mm_add_pi32(total, hi64);
    int s = _mm_cvtsi64_si32(total);
    _mm_empty();
    return sqrtf((float)s);
}

// ================================================================
// 6. AVX-512 x1
//    64 байта -> cvtepi8_epi16 (32 x int16) x2 -> madd -> int32 -> reduce
// ================================================================
float lengthAVX512_x1(const signed char* a, int n) {
    __m512i acc = _mm512_setzero_si512();
    for (int i = 0; i < n; i += 64) {
        __m512i v   = _mm512_loadu_si512((const __m512i*)(a + i));
        __m256i vlo = _mm512_castsi512_si256(v);
        __m256i vhi = _mm512_extracti64x4_epi64(v, 1);
        __m512i w0  = _mm512_cvtepi8_epi16(vlo);   // знаковое расшир. int8->int16
        __m512i w1  = _mm512_cvtepi8_epi16(vhi);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(w0, w0));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(w1, w1));
    }
    return sqrtf((float)_mm512_reduce_add_epi32(acc));
}

// ================================================================
// 7. AVX-512 x2
// ================================================================
float lengthAVX512_x2(const signed char* a, int n) {
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
    for (int i = 0; i < n; i += 128) {
        auto step = [&](__m512i& acc, int off){
            __m512i v   = _mm512_loadu_si512((const __m512i*)(a+i+off));
            __m256i vlo = _mm512_castsi512_si256(v);
            __m256i vhi = _mm512_extracti64x4_epi64(v,1);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_cvtepi8_epi16(vlo),_mm512_cvtepi8_epi16(vlo)));
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_cvtepi8_epi16(vhi),_mm512_cvtepi8_epi16(vhi)));
        };
        step(acc0,0); step(acc1,64);
    }
    __m512i acc = _mm512_add_epi32(acc0, acc1);
    return sqrtf((float)_mm512_reduce_add_epi32(acc));
}

// ================================================================
// 8. AVX-512 x4
// ================================================================
float lengthAVX512_x4(const signed char* a, int n) {
    __m512i acc0=_mm512_setzero_si512(), acc1=_mm512_setzero_si512();
    __m512i acc2=_mm512_setzero_si512(), acc3=_mm512_setzero_si512();
    for (int i = 0; i < n; i += 256) {
        auto step = [&](__m512i& acc, int off){
            __m512i v   = _mm512_loadu_si512((const __m512i*)(a+i+off));
            __m256i vlo = _mm512_castsi512_si256(v);
            __m256i vhi = _mm512_extracti64x4_epi64(v,1);
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_cvtepi8_epi16(vlo),_mm512_cvtepi8_epi16(vlo)));
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_cvtepi8_epi16(vhi),_mm512_cvtepi8_epi16(vhi)));
        };
        step(acc0,0); step(acc1,64); step(acc2,128); step(acc3,192);
    }
    __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0,acc1),_mm512_add_epi32(acc2,acc3));
    return sqrtf((float)_mm512_reduce_add_epi32(acc));
}

// ================================================================
// 9. AVX-512 x8
// ================================================================
float lengthAVX512_x8(const signed char* a, int n) {
    __m512i acc[8];
    for (int k = 0; k < 8; ++k) acc[k] = _mm512_setzero_si512();
    for (int i = 0; i < n; i += 512) {
        for (int k = 0; k < 8; ++k) {
            __m512i v   = _mm512_loadu_si512((const __m512i*)(a+i+k*64));
            __m256i vlo = _mm512_castsi512_si256(v);
            __m256i vhi = _mm512_extracti64x4_epi64(v,1);
            acc[k] = _mm512_add_epi32(acc[k], _mm512_madd_epi16(_mm512_cvtepi8_epi16(vlo),_mm512_cvtepi8_epi16(vlo)));
            acc[k] = _mm512_add_epi32(acc[k], _mm512_madd_epi16(_mm512_cvtepi8_epi16(vhi),_mm512_cvtepi8_epi16(vhi)));
        }
    }
    __m512i total = _mm512_setzero_si512();
    for (int k = 0; k < 8; ++k) total = _mm512_add_epi32(total, acc[k]);
    return sqrtf((float)_mm512_reduce_add_epi32(total));
}

// ================================================================
// замер
// ================================================================
typedef float (*FnPtr)(const signed char*, int);

struct Result { double ms; float val; };

Result measure(FnPtr fn, const signed char* a, int n, int reps) {
    // прогрев
    fn(a, n);
    double t0 = qpcNow();
    float v = 0;
    for (int r = 0; r < reps; ++r) v = fn(a, n);
    double t1 = qpcNow();
    sink_f = v;
    return {(t1-t0)/reps, v};
}

void printRow(const char* name, double ms, float val, float ref) {
    float diff = fabsf(val - ref);
    bool ok = diff < 1.5f;
    std::cout << std::left  << std::setw(28) << name
              << std::right << std::setw(10) << std::fixed << std::setprecision(4) << ms << " ms"
              << std::setw(14) << std::setprecision(2)  << val
              << "  " << (ok ? "OK" : "FAIL") << "\n";
}

void writeJson(const char* path,
               Result sc, Result ms, Result mx2, Result mx4, Result mx8,
               Result a1, Result a2, Result a4, Result a8) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);
    f << "{\n"
      << "  \"N\": " << N << ",\n"
      << "  \"scalar_ms\":"     << sc.ms  << ", \"scalar_result\":"      << sc.val  << ",\n"
      << "  \"mmx_scalar_ms\":" << ms.ms  << ", \"mmx_scalar_result\":"  << ms.val  << ",\n"
      << "  \"mmx_x2_ms\":"     << mx2.ms << ", \"mmx_x2_result\":"      << mx2.val << ",\n"
      << "  \"mmx_x4_ms\":"     << mx4.ms << ", \"mmx_x4_result\":"      << mx4.val << ",\n"
      << "  \"mmx_x8_ms\":"     << mx8.ms << ", \"mmx_x8_result\":"      << mx8.val << ",\n"
      << "  \"avx512_x1_ms\":"  << a1.ms  << ", \"avx512_x1_result\":"   << a1.val  << ",\n"
      << "  \"avx512_x2_ms\":"  << a2.ms  << ", \"avx512_x2_result\":"   << a2.val  << ",\n"
      << "  \"avx512_x4_ms\":"  << a4.ms  << ", \"avx512_x4_result\":"   << a4.val  << ",\n"
      << "  \"avx512_x8_ms\":"  << a8.ms  << ", \"avx512_x8_result\":"   << a8.val  << "\n"
      << "}\n";
    f.close();
}

// ================================================================
// main
// ================================================================
int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    srand((unsigned)time(nullptr));

    signed char* a = (signed char*)_aligned_malloc(N, 64);
    for (int i = 0; i < N; ++i)
        a[i] = (signed char)(rand() % 91 - 45);

    const int REPS = 200;

    std::cout << "Вариант 5: длина вектора int8, N=" << N << ", повторений=" << REPS << "\n";
    std::cout << std::string(60,'-') << "\n";
    std::cout << std::left << std::setw(28) << "Метод"
              << std::right << std::setw(12) << "Время"
              << std::setw(14) << "Результат" << "  Корр.\n";
    std::cout << std::string(60,'-') << "\n";

    auto sc  = measure(lengthScalar,    a, N, REPS);
    printRow("Scalar C++",  sc.ms,  sc.val,  sc.val);

    auto ms  = measure(lengthMMX_scalar, a, N, REPS);
    printRow("MMX scalar",  ms.ms,  ms.val,  sc.val);

    auto mx2 = measure(lengthMMX_x2, a, N, REPS);
    printRow("MMX vector x2", mx2.ms, mx2.val, sc.val);

    auto mx4 = measure(lengthMMX_x4, a, N, REPS);
    printRow("MMX vector x4", mx4.ms, mx4.val, sc.val);

    auto mx8 = measure(lengthMMX_x8, a, N, REPS);
    printRow("MMX vector x8", mx8.ms, mx8.val, sc.val);

    auto a1  = measure(lengthAVX512_x1, a, N, REPS);
    printRow("AVX-512 x1",  a1.ms,  a1.val,  sc.val);

    auto a2  = measure(lengthAVX512_x2, a, N, REPS);
    printRow("AVX-512 x2",  a2.ms,  a2.val,  sc.val);

    auto a4  = measure(lengthAVX512_x4, a, N, REPS);
    printRow("AVX-512 x4",  a4.ms,  a4.val,  sc.val);

    auto a8  = measure(lengthAVX512_x8, a, N, REPS);
    printRow("AVX-512 x8",  a8.ms,  a8.val,  sc.val);

    std::cout << std::string(60,'-') << "\n";

    std::cout << "\nУскорения относительно Scalar C++:\n";
    auto spd = [&](const char* name, double t) {
        std::cout << "  " << std::left << std::setw(26) << name
                  << std::right << std::fixed << std::setprecision(2)
                  << sc.ms/t << "x\n";
    };
    spd("MMX scalar",    ms.ms);
    spd("MMX vector x2", mx2.ms);
    spd("MMX vector x4", mx4.ms);
    spd("MMX vector x8", mx8.ms);
    spd("AVX-512 x1",    a1.ms);
    spd("AVX-512 x2",    a2.ms);
    spd("AVX-512 x4",    a4.ms);
    spd("AVX-512 x8",    a8.ms);

    writeJson(JSON_FILE, sc, ms, mx2, mx4, mx8, a1, a2, a4, a8);
    std::cout << "\n" << JSON_FILE << " saved\nsink=" << sink_f << "\n";

    _aligned_free(a);
    return 0;
}