#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <intrin.h>
#include <windows.h>

// ---- вспомогательные функции ----

// читает один бит из значения регистра
bool bit(unsigned int val, int pos) {
    return (val >> pos) & 1u;
}

// читает поле [hi:lo] из значения регистра
unsigned int field(unsigned int val, int lo, int hi) {
    unsigned int mask = (1u << (hi - lo + 1)) - 1u;
    return (val >> lo) & mask;
}

// печатает строку флага: "  NAME         = YES / NO"
void printFlag(const char* name, bool val) {
    std::cout << "  " << std::left << std::setw(24) << name
              << " = " << (val ? "YES" : "NO") << "\n";
}

void printVal(const char* name, unsigned int val, const char* unit = "") {
    std::cout << "  " << std::left << std::setw(24) << name
              << " = " << val << " " << unit << "\n";
}

void printStr(const char* name, const char* val) {
    std::cout << "  " << std::left << std::setw(24) << name
              << " = " << val << "\n";
}

void printHex(const char* name, unsigned int val) {
    std::cout << "  " << std::left << std::setw(24) << name
              << " = 0x" << std::hex << std::uppercase << val
              << std::dec << "\n";
}

void section(const char* title) {
    std::cout << "\n=== " << title << " ===\n";
}

// ---- JSON-запись ----

struct Json {
    std::ofstream f;
    bool first = true;

    Json(const char* path) : f(path) { f << "{\n"; }

    void str(const char* key, const char* val) {
        if (!first) f << ",\n"; first = false;
        f << "  \"" << key << "\": \"" << val << "\"";
    }
    void num(const char* key, unsigned int val) {
        if (!first) f << ",\n"; first = false;
        f << "  \"" << key << "\": " << val;
    }
    void flag(const char* key, bool val) {
        if (!first) f << ",\n"; first = false;
        f << "  \"" << key << "\": " << (val ? "true" : "false");
    }
    void close() { f << "\n}\n"; f.close(); }
};

// ---- имя выходного файла (поменяй здесь перед запуском на каждом процессоре) ----

const char* JSON_FILE = "C:\\Users\\darya\\Desktop\\misis\\8sem\\pc\\misis2026s-pc-22-01-vanina-d-a\\lab3\\cpuid_results.json";

// ---- main ----

int main() {
    int info[4];   // EAX, EBX, ECX, EDX

    Json js(JSON_FILE);

    // ================================================================
    // EAX=0 — производитель и число базовых функций
    // ================================================================
    section("EAX=0: vendor & max basic function");
    __cpuid(info, 0);

    unsigned int maxBasic = (unsigned int)info[0];

    // vendor string: EBX, EDX, ECX (именно в таком порядке)
    char vendor[13] = {};
    memcpy(vendor + 0, &info[1], 4);
    memcpy(vendor + 4, &info[3], 4);
    memcpy(vendor + 8, &info[2], 4);
    vendor[12] = '\0';

    printVal("Max basic function", maxBasic);
    printStr("Vendor",            vendor);

    js.num("max_basic_function", maxBasic);
    js.str("vendor", vendor);

    bool isIntel = (strcmp(vendor, "GenuineIntel") == 0);
    bool isAMD   = (strcmp(vendor, "AuthenticAMD") == 0);

    // ================================================================
    // EAX=1 — версия процессора, флаги EDX/ECX
    // ================================================================
    section("EAX=1: version & feature flags");
    __cpuid(info, 1);

    unsigned int _eax = (unsigned int)info[0];
    unsigned int _ebx = (unsigned int)info[1];
    unsigned int _ecx = (unsigned int)info[2];
    unsigned int _edx = (unsigned int)info[3];

    unsigned int steppingID    = field(_eax,  0,  3);
    unsigned int model         = field(_eax,  4,  7);
    unsigned int family        = field(_eax,  8, 11);
    unsigned int procType      = field(_eax, 12, 13);
    unsigned int extModel      = field(_eax, 16, 19);
    unsigned int extFamily     = field(_eax, 20, 27);
    unsigned int logicalPerPhy = field(_ebx, 16, 23);
    unsigned int localApicId   = field(_ebx, 24, 31);

    printHex("EAX raw",          _eax);
    printVal("Stepping ID",       steppingID);
    printVal("Model",             model);
    printVal("Family",            family);
    printVal("Processor type",    procType);
    printVal("Extended model",    extModel);
    printVal("Extended family",   extFamily);
    printVal("Logical/physical",  logicalPerPhy);
    printVal("Local APIC ID",     localApicId);

    js.num("stepping_id",    steppingID);
    js.num("model",          model);
    js.num("family",         family);
    js.num("ext_model",      extModel);
    js.num("ext_family",     extFamily);
    js.num("logical_per_phy",logicalPerPhy);
    js.num("local_apic_id",  localApicId);

    // EDX флаги
    std::cout << "\n  -- EDX flags --\n";
    bool fpu  = bit(_edx, 0);
    bool tsc  = bit(_edx, 4);
    bool mmx  = bit(_edx, 23);
    bool sse  = bit(_edx, 25);
    bool sse2 = bit(_edx, 26);
    bool htt  = bit(_edx, 28);

    printFlag("FPU",  fpu);
    printFlag("TSC",  tsc);
    printFlag("MMX",  mmx);
    printFlag("SSE",  sse);
    printFlag("SSE2", sse2);
    printFlag("HTT",  htt);

    js.flag("fpu",  fpu);
    js.flag("tsc",  tsc);
    js.flag("mmx",  mmx);
    js.flag("sse",  sse);
    js.flag("sse2", sse2);
    js.flag("htt",  htt);

    // ECX флаги
    std::cout << "\n  -- ECX flags --\n";
    bool sse3   = bit(_ecx,  0);
    bool ssse3  = bit(_ecx,  9);
    bool fma3   = bit(_ecx, 12);
    bool sse4_1 = bit(_ecx, 19);
    bool sse4_2 = bit(_ecx, 20);
    bool avx    = bit(_ecx, 28);

    printFlag("SSE3",   sse3);
    printFlag("SSSE3",  ssse3);
    printFlag("FMA3",   fma3);
    printFlag("SSE4.1", sse4_1);
    printFlag("SSE4.2", sse4_2);
    printFlag("AVX",    avx);

    js.flag("sse3",   sse3);
    js.flag("ssse3",  ssse3);
    js.flag("fma3",   fma3);
    js.flag("sse4_1", sse4_1);
    js.flag("sse4_2", sse4_2);
    js.flag("avx",    avx);

    // ================================================================
    // EAX=4 (Intel) / EAX=8000001Dh (AMD) — кэш-память
    // ================================================================
    section("Cache topology");

    unsigned int cacheLeaf = isAMD ? 0x8000001Du : 4u;
    int cacheIdx = 0;

    while (true) {
        __cpuidex(info, (int)cacheLeaf, cacheIdx);
        unsigned int ca = (unsigned int)info[0];
        unsigned int cb = (unsigned int)info[1];
        unsigned int cc = (unsigned int)info[2];
        unsigned int cd = (unsigned int)info[3];

        unsigned int cacheType  = field(ca, 0,  4);
        if (cacheType == 0) break;   // нет больше кэшей

        unsigned int cacheLevel = field(ca, 5,  7);
        bool         fullyAssoc = bit(ca, 9);
        unsigned int threadsPC  = field(ca, 14, 25) + 1;
        unsigned int procCores  = field(ca, 26, 31) + 1;

        unsigned int lineSize   = field(cb,  0, 11) + 1;
        unsigned int partitions = field(cb, 12, 21) + 1;
        unsigned int ways       = field(cb, 22, 31) + 1;
        unsigned int sets       = cc + 1;
        bool         inclusive  = bit(cd, 1);

        unsigned long long cacheSize =
            (unsigned long long)lineSize * partitions * ways * sets;

        const char* typeStr = (cacheType == 1) ? "Data" :
                              (cacheType == 2) ? "Instruction" : "Unified";

        char label[32];
        sprintf(label, "L%u %s", cacheLevel, typeStr);
        std::cout << "\n  [" << label << "]\n";
        printVal("  Cache line size (B)", lineSize);
        printVal("  Ways (associativity)", ways);
        if (!fullyAssoc)
            printVal("  Sets",             sets);
        else
            std::cout << "  Fully associative\n";
        printVal("  Partitions",          partitions);
        printVal("  Threads sharing",     threadsPC);
        printVal("  Phys cores",          procCores);
        std::cout << "  " << std::left << std::setw(24) << "Inclusiveness"
                  << " = " << (inclusive ? "Inclusive" : "Exclusive") << "\n";

        if (cacheSize < 1024)
            std::cout << "  Cache size           = " << cacheSize << " B\n";
        else if (cacheSize < 1024*1024)
            std::cout << "  Cache size           = " << cacheSize/1024 << " KB\n";
        else
            std::cout << "  Cache size           = " << cacheSize/1024/1024 << " MB\n";

        // JSON — кэш под ключом "cacheN"
        char jkey[32];
        sprintf(jkey, "cache%d_label", cacheIdx);
        js.str(jkey, label);
        sprintf(jkey, "cache%d_size_kb", cacheIdx);
        js.num(jkey, (unsigned int)(cacheSize / 1024));

        cacheIdx++;
    }

    js.num("cache_levels_found", (unsigned int)cacheIdx);

    // ================================================================
    // EAX=7, ECX=0 — расширенные инструкции
    // ================================================================
    if (maxBasic >= 7) {
        section("EAX=7 ECX=0: extended instruction sets");
        __cpuidex(info, 7, 0);
        unsigned int ebx7 = (unsigned int)info[1];
        unsigned int ecx7 = (unsigned int)info[2];
        unsigned int edx7 = (unsigned int)info[3];

        bool avx2      = bit(ebx7,  5);
        bool rtm       = bit(ebx7, 11);
        bool avx512f   = bit(ebx7, 16);
        bool sha       = bit(ebx7, 29);
        bool gfni      = bit(ecx7,  8);
        bool amx_bf16  = bit(edx7, 22);
        bool amx_tile  = bit(edx7, 24);
        bool amx_int8  = bit(edx7, 25);

        printFlag("AVX2",      avx2);
        printFlag("RTM/TSX",   rtm);
        printFlag("AVX512-F",  avx512f);
        printFlag("SHA",       sha);
        printFlag("GFNI",      gfni);
        printFlag("AMX-BF16",  amx_bf16);
        printFlag("AMX-TILE",  amx_tile);
        printFlag("AMX-INT8",  amx_int8);

        js.flag("avx2",     avx2);
        js.flag("rtm_tsx",  rtm);
        js.flag("avx512f",  avx512f);
        js.flag("sha",      sha);
        js.flag("gfni",     gfni);
        js.flag("amx_bf16", amx_bf16);
        js.flag("amx_tile", amx_tile);
        js.flag("amx_int8", amx_int8);
    }

    // ================================================================
    // EAX=16h — тактовые частоты
    // ================================================================
    if (maxBasic >= 0x16) {
        section("EAX=16h: clock frequencies");
        __cpuid(info, 0x16);
        unsigned int baseFreq = (unsigned int)info[0] & 0xFFFF;
        unsigned int maxFreq  = (unsigned int)info[1] & 0xFFFF;
        unsigned int busFreq  = (unsigned int)info[2] & 0xFFFF;

        printVal("Base frequency",  baseFreq, "MHz");
        printVal("Max (boost) freq",maxFreq,  "MHz");
        printVal("Bus frequency",   busFreq,  "MHz");

        js.num("base_freq_mhz", baseFreq);
        js.num("max_freq_mhz",  maxFreq);
        js.num("bus_freq_mhz",  busFreq);
    }

    // ================================================================
    // EAX=80000000h — число расширенных функций
    // ================================================================
    section("EAX=80000000h: max extended function");
    __cpuid(info, (int)0x80000000u);
    unsigned int maxExt = (unsigned int)info[0];
    printHex("Max extended fn", maxExt);
    js.num("max_ext_function", maxExt);

    // ================================================================
    // EAX=80000001h — расширения AMD
    // ================================================================
    if (maxExt >= 0x80000001u) {
        section("EAX=80000001h: AMD extended flags");
        __cpuid(info, (int)0x80000001u);
        unsigned int ecx1 = (unsigned int)info[2];
        unsigned int edx1 = (unsigned int)info[3];

        bool sse4a  = bit(ecx1, 6);
        bool fma4   = bit(ecx1, 16);
        bool dnow   = bit(edx1, 31);
        bool dnow_e = bit(edx1, 30);

        printFlag("SSE4a",        sse4a);
        printFlag("FMA4",         fma4);
        printFlag("3DNow!",       dnow);
        printFlag("Ext 3DNow!",   dnow_e);

        js.flag("sse4a",    sse4a);
        js.flag("fma4",     fma4);
        js.flag("3dnow",    dnow);
        js.flag("3dnow_ext",dnow_e);
    }

    // ================================================================
    // EAX=80000002h–80000004h — название процессора
    // ================================================================
    if (maxExt >= 0x80000004u) {
        section("EAX=80000002h-04h: CPU brand string");
        char brand[49] = {};
        for (int leaf = 0; leaf < 3; ++leaf) {
            __cpuid(info, (int)(0x80000002u + leaf));
            memcpy(brand + leaf * 16, info, 16);
        }
        brand[48] = '\0';
        // обрезаем ведущие пробелы
        const char* b = brand;
        while (*b == ' ') ++b;
        printStr("Brand string", b);
        js.str("brand_string", b);
    }

    js.close();

    std::cout << "\n" << JSON_FILE << " saved\n";
    return 0;
}