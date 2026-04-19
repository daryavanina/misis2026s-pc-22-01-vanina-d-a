#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <immintrin.h>   // AVX/AVX2/AVX-512
#include <emmintrin.h>   // SSE2
#include <xmmintrin.h>   // SSE
#include <windows.h>
#include <intrin.h>

// ============================================================
// Лабораторная работа № 7
// Вариант 5: Оператор Собела
//
// Маски:
//   Mh = [[ 1, 2, 1],   Mv = [[ 1, 0,-1],
//          [ 0, 0, 0],          [ 2, 0,-2],
//          [-1,-2,-1]]          [ 1, 0,-1]]
//
// Упрощённые формулы:
//   t1 = A - I
//   t2 = C - G
//   Hh = 2*(D - F) + t1 - t2
//   Hv = 2*(B - H) + t1 + t2
//   d  = floor( 256/1140 * sqrt(Hh^2 + Hv^2) )
//
// Окрестность точки (x,y):
//   A B C      [x-1,y-1] [x,y-1] [x+1,y-1]
//   D E F  =   [x-1,y  ] [x,y  ] [x+1,y  ]
//   G H I      [x-1,y+1] [x,y+1] [x+1,y+1]
// ============================================================

static volatile int   sink_i = 0;
static volatile float sink_f = 0.0f;

// ---- изображение ----

const int IMG_W = 1920;
const int IMG_H = 1080;

// ---- замер QPC ----

const char* JSON_FILE = "C:/Users/darya/Desktop/misis/8sem/pc/misis2026s-pc-22-01-vanina-d-a/lab7/lab7_results.json";
double qpcMs() {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart * 1000.0;
}

// ---- заполнение тестового изображения ----

void fillImage(unsigned char* img, int w, int h) {
    srand(42);
    for (int i = 0; i < w * h; ++i)
        img[i] = (unsigned char)(rand() % 256);
}

// ================================================================
// 1. Scalar C++
// ================================================================
void sobelScalar(const unsigned char* src, unsigned char* dst, int w, int h) {
    const float norm = 256.0f / 1140.0f;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int A = src[(y-1)*w + (x-1)];
            int B = src[(y-1)*w +  x   ];
            int C = src[(y-1)*w + (x+1)];
            int D = src[ y   *w + (x-1)];
            // E не используется
            int F = src[ y   *w + (x+1)];
            int G = src[(y+1)*w + (x-1)];
            int H = src[(y+1)*w +  x   ];
            int I = src[(y+1)*w + (x+1)];

            int t1 = A - I;
            int t2 = C - G;
            int Hh = 2*(D - F) + t1 - t2;
            int Hv = 2*(B - H) + t1 + t2;

            float d = norm * sqrtf((float)(Hh*Hh + Hv*Hv));
            if (d > 255.0f) d = 255.0f;
            dst[y*w + x] = (unsigned char)d;
        }
    }
    sink_i = dst[w+1];
}

// ================================================================
// 2. SSE2 (целые) + SSE (float)
//    Обрабатываем 8 пикселей за итерацию (int16 x 8 = 128 бит)
// ================================================================
void sobelSSE2(const unsigned char* src, unsigned char* dst, int w, int h) {
    const float norm = 256.0f / 1140.0f;
    const __m128 vnorm = _mm_set1_ps(norm);
    const __m128 v255  = _mm_set1_ps(255.0f);
    const __m128 vzero = _mm_setzero_ps();

    for (int y = 1; y < h - 1; ++y) {
        // обрабатываем по 8 пикселей, оставляем по 1 пикселю с каждого края
        int x = 1;
        for (; x <= w - 9; x += 8) {
            // загружаем 8 байт из каждой строки окрестности
            // и расширяем беззнаково в int16
            __m128i row0 = _mm_loadl_epi64((const __m128i*)(src + (y-1)*w + x - 1));
            __m128i row1 = _mm_loadl_epi64((const __m128i*)(src +  y   *w + x - 1));
            __m128i row2 = _mm_loadl_epi64((const __m128i*)(src + (y+1)*w + x - 1));
            // расширяем 8 байт -> 8 x int16
            __m128i z = _mm_setzero_si128();
            row0 = _mm_unpacklo_epi8(row0, z);  // [A0..A7] -> int16
            row1 = _mm_unpacklo_epi8(row1, z);  // [D0..D7] -> int16
            row2 = _mm_unpacklo_epi8(row2, z);  // [G0..G7] -> int16

            // Загрузим также сдвиги на 1 и 2 байта вправо для B,C,E,F,H,I
            __m128i row0_1 = _mm_loadl_epi64((const __m128i*)(src + (y-1)*w + x));
            __m128i row0_2 = _mm_loadl_epi64((const __m128i*)(src + (y-1)*w + x + 1));
            __m128i row1_1 = _mm_loadl_epi64((const __m128i*)(src +  y   *w + x));
            __m128i row1_2 = _mm_loadl_epi64((const __m128i*)(src +  y   *w + x + 1));
            __m128i row2_1 = _mm_loadl_epi64((const __m128i*)(src + (y+1)*w + x));
            __m128i row2_2 = _mm_loadl_epi64((const __m128i*)(src + (y+1)*w + x + 1));

            row0_1 = _mm_unpacklo_epi8(row0_1, z);  // B
            row0_2 = _mm_unpacklo_epi8(row0_2, z);  // C
            row1_1 = _mm_unpacklo_epi8(row1_1, z);  // E (не нужен)
            row1_2 = _mm_unpacklo_epi8(row1_2, z);  // F
            row2_1 = _mm_unpacklo_epi8(row2_1, z);  // H
            row2_2 = _mm_unpacklo_epi8(row2_2, z);  // I

            // A=row0, B=row0_1, C=row0_2
            // D=row1, F=row1_2
            // G=row2, H=row2_1, I=row2_2
            __m128i* A = &row0;   __m128i* B = &row0_1; __m128i* C_= &row0_2;
            __m128i* D = &row1;   __m128i* F = &row1_2;
            __m128i* G = &row2;   __m128i* H_ = &row2_1; __m128i* I_ = &row2_2;

            // t1 = A - I
            __m128i t1 = _mm_sub_epi16(*A, *I_);
            // t2 = C - G
            __m128i t2 = _mm_sub_epi16(*C_, *G);
            // Hh = 2*(D-F) + t1 - t2
            __m128i df  = _mm_sub_epi16(*D, *F);
            __m128i df2 = _mm_add_epi16(df, df);
            __m128i Hh  = _mm_add_epi16(_mm_sub_epi16(t1, t2), df2);
            // Hv = 2*(B-H) + t1 + t2
            __m128i bh  = _mm_sub_epi16(*B, *H_);
            __m128i bh2 = _mm_add_epi16(bh, bh);
            __m128i Hv  = _mm_add_epi16(_mm_add_epi16(t1, t2), bh2);

            // Переводим int16 -> int32 -> float, обрабатываем 4 за раз
            for (int half = 0; half < 2; ++half) {
                __m128i hh16, hv16;
                if (half == 0) {
                    hh16 = _mm_unpacklo_epi16(Hh, _mm_srai_epi16(Hh, 15));
                    hv16 = _mm_unpacklo_epi16(Hv, _mm_srai_epi16(Hv, 15));
                } else {
                    hh16 = _mm_unpackhi_epi16(Hh, _mm_srai_epi16(Hh, 15));
                    hv16 = _mm_unpackhi_epi16(Hv, _mm_srai_epi16(Hv, 15));
                }
                __m128 fHh = _mm_cvtepi32_ps(hh16);
                __m128 fHv = _mm_cvtepi32_ps(hv16);
                __m128 fSq = _mm_add_ps(_mm_mul_ps(fHh, fHh), _mm_mul_ps(fHv, fHv));
                __m128 fd  = _mm_mul_ps(_mm_sqrt_ps(fSq), vnorm);
                fd = _mm_min_ps(_mm_max_ps(fd, vzero), v255);
                // конвертируем float -> int и записываем
                __m128i id = _mm_cvtps_epi32(fd);
                // pack int32 -> int16 -> uint8
                __m128i packed = _mm_packs_epi32(id, _mm_setzero_si128());
                packed = _mm_packus_epi16(packed, _mm_setzero_si128());
                // записываем 4 байта
                int result = _mm_cvtsi128_si32(packed);
                memcpy(dst + y*w + x + half*4, &result, 4);
            }
        }
        // остаток скалярно
        for (; x < w - 1; ++x) {
            int A=src[(y-1)*w+(x-1)], B=src[(y-1)*w+x], C=src[(y-1)*w+(x+1)];
            int D=src[y*w+(x-1)],                        F=src[y*w+(x+1)];
            int G=src[(y+1)*w+(x-1)], H=src[(y+1)*w+x], I=src[(y+1)*w+(x+1)];
            int t1=A-I, t2=C-G;
            int Hh=2*(D-F)+t1-t2, Hv=2*(B-H)+t1+t2;
            float d=(256.0f/1140.0f)*sqrtf((float)(Hh*Hh+Hv*Hv));
            if(d>255.0f)d=255.0f;
            dst[y*w+x]=(unsigned char)d;
        }
    }
    sink_i = dst[w+1];
}

// ================================================================
// 3. AVX2 (целые) + AVX (float)
//    Обрабатываем 16 пикселей за итерацию (int16 x 16 = 256 бит)
// ================================================================
void sobelAVX2(const unsigned char* src, unsigned char* dst, int w, int h) {
    const float norm = 256.0f / 1140.0f;
    const __m256 vnorm = _mm256_set1_ps(norm);
    const __m256 v255  = _mm256_set1_ps(255.0f);
    const __m256 vzero = _mm256_setzero_ps();

    for (int y = 1; y < h - 1; ++y) {
        int x = 1;
        for (; x <= w - 17; x += 16) {
            // загружаем 16 байт, расширяем в int16
            __m128i z128 = _mm_setzero_si128();
            auto load16 = [&](int row, int col) -> __m256i {
                __m128i v = _mm_loadu_si128((const __m128i*)(src + row*w + col));
                return _mm256_cvtepu8_epi16(v);  // 16 x uint8 -> 16 x int16
            };

            __m256i vA = load16(y-1, x-1);
            __m256i vB = load16(y-1, x  );
            __m256i vC = load16(y-1, x+1);
            __m256i vD = load16(y  , x-1);
            __m256i vF = load16(y  , x+1);
            __m256i vG = load16(y+1, x-1);
            __m256i vH = load16(y+1, x  );
            __m256i vI = load16(y+1, x+1);

            __m256i t1 = _mm256_sub_epi16(vA, vI);
            __m256i t2 = _mm256_sub_epi16(vC, vG);
            __m256i df = _mm256_sub_epi16(vD, vF);
            __m256i df2= _mm256_add_epi16(df, df);
            __m256i Hh = _mm256_add_epi16(_mm256_sub_epi16(t1, t2), df2);
            __m256i bh = _mm256_sub_epi16(vB, vH);
            __m256i bh2= _mm256_add_epi16(bh, bh);
            __m256i Hv = _mm256_add_epi16(_mm256_add_epi16(t1, t2), bh2);

            // обрабатываем 4 группы по 4 элемента
            for (int q = 0; q < 4; ++q) {
                __m128i hh16 = (q < 2)
                    ? _mm256_castsi256_si128(Hh)
                    : _mm256_extracti128_si256(Hh, 1);
                __m128i hv16 = (q < 2)
                    ? _mm256_castsi256_si128(Hv)
                    : _mm256_extracti128_si256(Hv, 1);
                if (q % 2 == 0) {
                    hh16 = _mm_unpacklo_epi16(hh16, _mm_srai_epi16(hh16, 15));
                    hv16 = _mm_unpacklo_epi16(hv16, _mm_srai_epi16(hv16, 15));
                } else {
                    hh16 = _mm_unpackhi_epi16(hh16, _mm_srai_epi16(hh16, 15));
                    hv16 = _mm_unpackhi_epi16(hv16, _mm_srai_epi16(hv16, 15));
                }
                __m128 fHh = _mm_cvtepi32_ps(hh16);
                __m128 fHv = _mm_cvtepi32_ps(hv16);
                __m128 fSq = _mm_add_ps(_mm_mul_ps(fHh,fHh), _mm_mul_ps(fHv,fHv));
                __m128 fd  = _mm_mul_ps(_mm_sqrt_ps(fSq), _mm256_castps256_ps128(vnorm));
                fd = _mm_min_ps(_mm_max_ps(fd, _mm_setzero_ps()),
                                _mm256_castps256_ps128(v255));
                __m128i id = _mm_cvtps_epi32(fd);
                __m128i packed = _mm_packs_epi32(id, _mm_setzero_si128());
                packed = _mm_packus_epi16(packed, _mm_setzero_si128());
                int result = _mm_cvtsi128_si32(packed);
                memcpy(dst + y*w + x + q*4, &result, 4);
            }
        }
        for (; x < w-1; ++x) {
            int A=src[(y-1)*w+(x-1)],B=src[(y-1)*w+x],C=src[(y-1)*w+(x+1)];
            int D=src[y*w+(x-1)],F=src[y*w+(x+1)];
            int G=src[(y+1)*w+(x-1)],H=src[(y+1)*w+x],I=src[(y+1)*w+(x+1)];
            int t1=A-I,t2=C-G;
            int Hh=2*(D-F)+t1-t2,Hv=2*(B-H)+t1+t2;
            float d=(256.0f/1140.0f)*sqrtf((float)(Hh*Hh+Hv*Hv));
            if(d>255.0f)d=255.0f;
            dst[y*w+x]=(unsigned char)d;
        }
    }
    sink_i = dst[w+1];
}

// ================================================================
// 4. AVX-512: обрабатываем 32 пикселя за итерацию (int16 x 32 = 512 бит)
// ================================================================
void sobelAVX512(const unsigned char* src, unsigned char* dst, int w, int h) {
    const float norm = 256.0f / 1140.0f;
    const __m512 vnorm = _mm512_set1_ps(norm);
    const __m512 v255  = _mm512_set1_ps(255.0f);
    const __m512 vzero = _mm512_setzero_ps();

    for (int y = 1; y < h - 1; ++y) {
        int x = 1;
        for (; x <= w - 33; x += 32) {
            // загружаем 32 байта, расширяем в int16 (32 x uint8 -> 32 x int16)
            auto load32 = [&](int row, int col) -> __m512i {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + row*w + col));
                return _mm512_cvtepu8_epi16(v);
            };

            __m512i vA = load32(y-1, x-1);
            __m512i vB = load32(y-1, x  );
            __m512i vC = load32(y-1, x+1);
            __m512i vD = load32(y  , x-1);
            __m512i vF = load32(y  , x+1);
            __m512i vG = load32(y+1, x-1);
            __m512i vH = load32(y+1, x  );
            __m512i vI = load32(y+1, x+1);

            __m512i t1  = _mm512_sub_epi16(vA, vI);
            __m512i t2  = _mm512_sub_epi16(vC, vG);
            __m512i df  = _mm512_sub_epi16(vD, vF);
            __m512i df2 = _mm512_add_epi16(df, df);
            __m512i Hh  = _mm512_add_epi16(_mm512_sub_epi16(t1, t2), df2);
            __m512i bh  = _mm512_sub_epi16(vB, vH);
            __m512i bh2 = _mm512_add_epi16(bh, bh);
            __m512i Hv  = _mm512_add_epi16(_mm512_add_epi16(t1, t2), bh2);

            // обрабатываем 8 групп по 4 элемента
            for (int q = 0; q < 8; ++q) {
                // извлекаем нужные 4 элемента int16 из Hh/Hv
                __m128i hh16, hv16;
                int quarter = q / 2;
                int which   = q % 2;
                auto get128 = [&](__m512i v, int qt) -> __m128i {
                    switch(qt) {
                        case 0: return _mm512_castsi512_si128(v);
                        case 1: return _mm256_extracti128_si256(_mm512_castsi512_si256(v), 1);
                        case 2: return _mm512_extracti32x4_epi32(v, 2);
                        default:return _mm512_extracti32x4_epi32(v, 3);
                    }
                };
                __m128i hh_part = get128(Hh, quarter);
                __m128i hv_part = get128(Hv, quarter);
                if (which == 0) {
                    hh16 = _mm_unpacklo_epi16(hh_part, _mm_srai_epi16(hh_part, 15));
                    hv16 = _mm_unpacklo_epi16(hv_part, _mm_srai_epi16(hv_part, 15));
                } else {
                    hh16 = _mm_unpackhi_epi16(hh_part, _mm_srai_epi16(hh_part, 15));
                    hv16 = _mm_unpackhi_epi16(hv_part, _mm_srai_epi16(hv_part, 15));
                }
                __m128 fHh = _mm_cvtepi32_ps(hh16);
                __m128 fHv = _mm_cvtepi32_ps(hv16);
                __m128 fSq = _mm_add_ps(_mm_mul_ps(fHh,fHh), _mm_mul_ps(fHv,fHv));
                __m128 fd  = _mm_mul_ps(_mm_sqrt_ps(fSq),
                                        _mm512_castps512_ps128(vnorm));
                fd = _mm_min_ps(_mm_max_ps(fd, _mm_setzero_ps()),
                                _mm512_castps512_ps128(v255));
                __m128i id = _mm_cvtps_epi32(fd);
                __m128i packed = _mm_packs_epi32(id, _mm_setzero_si128());
                packed = _mm_packus_epi16(packed, _mm_setzero_si128());
                int result = _mm_cvtsi128_si32(packed);
                memcpy(dst + y*w + x + q*4, &result, 4);
            }
        }
        for (; x < w-1; ++x) {
            int A=src[(y-1)*w+(x-1)],B=src[(y-1)*w+x],C=src[(y-1)*w+(x+1)];
            int D=src[y*w+(x-1)],F=src[y*w+(x+1)];
            int G=src[(y+1)*w+(x-1)],H=src[(y+1)*w+x],I=src[(y+1)*w+(x+1)];
            int t1=A-I,t2=C-G;
            int Hh=2*(D-F)+t1-t2,Hv=2*(B-H)+t1+t2;
            float d=(256.0f/1140.0f)*sqrtf((float)(Hh*Hh+Hv*Hv));
            if(d>255.0f)d=255.0f;
            dst[y*w+x]=(unsigned char)d;
        }
    }
    sink_i = dst[w+1];
}

// ================================================================
// Проверка корректности
// ================================================================
int compareImages(const unsigned char* a, const unsigned char* b, int n) {
    int diff = 0;
    for (int i = 0; i < n; ++i)
        if (abs((int)a[i] - (int)b[i]) > 1) ++diff;
    return diff;
}

// ================================================================
// Сохранение изображения в PGM (для визуальной проверки)
// ================================================================
void savePGM(const char* fname, const unsigned char* img, int w, int h) {
    FILE* f = fopen(fname, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(img, 1, w*h, f);
    fclose(f);
    std::cout << "  saved: " << fname << "\n";
}

// ================================================================
// JSON
// ================================================================
void writeJson(const char* path,
               double ms_sc_rel, double ms_sc_dbg,
               double ms_sse2, double ms_avx2, double ms_avx512,
               int diff_sse2, int diff_avx2, int diff_avx512,
               int w, int h) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(4);
    f << "{\n"
      << "  \"width\": " << w << ", \"height\": " << h << ",\n"
      << "  \"pixels\": " << (w-2)*(h-2) << ",\n"
      << "  \"scalar_release_ms\": " << ms_sc_rel << ",\n"
      << "  \"scalar_debug_ms\":   " << ms_sc_dbg << ",\n"
      << "  \"sse2_ms\":           " << ms_sse2   << ",\n"
      << "  \"avx2_ms\":           " << ms_avx2   << ",\n"
      << "  \"avx512_ms\":         " << ms_avx512 << ",\n"
      << "  \"diff_sse2\":         " << diff_sse2  << ",\n"
      << "  \"diff_avx2\":         " << diff_avx2  << ",\n"
      << "  \"diff_avx512\":       " << diff_avx512 << "\n"
      << "}\n";
    f.close();
}

// ================================================================
// main
// ================================================================
int main() {
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    const int W = IMG_W, H = IMG_H;
    const int N = W * H;
    const int REPS = 20;

    unsigned char* src     = new unsigned char[N];
    unsigned char* dst_ref = new unsigned char[N]();
    unsigned char* dst_sse2= new unsigned char[N]();
    unsigned char* dst_avx2= new unsigned char[N]();
    unsigned char* dst_avx = new unsigned char[N]();

    fillImage(src, W, H);

    std::cout << "Оператор Собела, изображение " << W << "x" << H
              << ", повторений=" << REPS << "\n";
    std::cout << std::string(54, '-') << "\n";

    // -- Scalar (эталон) --
    double t0 = qpcMs();
    for (int r = 0; r < REPS; ++r) sobelScalar(src, dst_ref, W, H);
    double ms_scalar_rel = (qpcMs() - t0) / REPS;
    std::cout << std::left << std::setw(20) << "Scalar Release"
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << ms_scalar_rel << " ms   1.00x\n";

    // -- SSE2+SSE --
    t0 = qpcMs();
    for (int r = 0; r < REPS; ++r) sobelSSE2(src, dst_sse2, W, H);
    double ms_sse2 = (qpcMs() - t0) / REPS;
    int diff_sse2 = compareImages(dst_ref, dst_sse2, N);
    std::cout << std::left << std::setw(20) << "SSE2+SSE"
              << std::right << std::setw(10) << ms_sse2 << " ms"
              << std::setw(7) << std::setprecision(2) << ms_scalar_rel/ms_sse2 << "x"
              << "  diff=" << diff_sse2 << "\n";

    // -- AVX2+AVX --
    t0 = qpcMs();
    for (int r = 0; r < REPS; ++r) sobelAVX2(src, dst_avx2, W, H);
    double ms_avx2 = (qpcMs() - t0) / REPS;
    int diff_avx2 = compareImages(dst_ref, dst_avx2, N);
    std::cout << std::left << std::setw(20) << "AVX2+AVX"
              << std::right << std::setw(10) << ms_avx2 << " ms"
              << std::setw(7) << ms_scalar_rel/ms_avx2 << "x"
              << "  diff=" << diff_avx2 << "\n";

    // -- AVX-512 --
    t0 = qpcMs();
    for (int r = 0; r < REPS; ++r) sobelAVX512(src, dst_avx, W, H);
    double ms_avx512 = (qpcMs() - t0) / REPS;
    int diff_avx512 = compareImages(dst_ref, dst_avx, N);
    std::cout << std::left << std::setw(20) << "AVX-512"
              << std::right << std::setw(10) << ms_avx512 << " ms"
              << std::setw(7) << ms_scalar_rel/ms_avx512 << "x"
              << "  diff=" << diff_avx512 << "\n";

    std::cout << std::string(54, '-') << "\n";

    // сохраняем изображения для визуальной проверки
    savePGM("C:\\Users\\darya\\Desktop\\misis\\8sem\\pc\\misis2026s-pc-22-01-vanina-d-a\\lab7\\src.pgm",     src,     W, H);
    savePGM("C:\\Users\\darya\\Desktop\\misis\\8sem\\pc\\misis2026s-pc-22-01-vanina-d-a\\lab7\\dst_ref.pgm", dst_ref, W, H);
    savePGM("C:\\Users\\darya\\Desktop\\misis\\8sem\\pc\\misis2026s-pc-22-01-vanina-d-a\\lab7\\dst_avx.pgm", dst_avx, W, H);

    // Debug scalar — запускаем отдельно через константу
    // (при компиляции с -O0 ms_scalar_rel будет другим)
    double ms_scalar_dbg = ms_scalar_rel;  // заполнится при запуске с -O0
#ifdef NDEBUG
    ms_scalar_dbg = ms_scalar_rel;
#endif

    writeJson(JSON_FILE,
              ms_scalar_rel, ms_scalar_dbg,
              ms_sse2, ms_avx2, ms_avx512,
              diff_sse2, diff_avx2, diff_avx512,
              W, H);
    std::cout << "lab7_results.json saved\n";
    std::cout << "sink=" << sink_i << "\n";

    delete[] src; delete[] dst_ref;
    delete[] dst_sse2; delete[] dst_avx2; delete[] dst_avx;
    return 0;
}
