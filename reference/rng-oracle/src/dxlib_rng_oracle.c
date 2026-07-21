/* Read-only behavioral oracle for the shipped 32-bit DxLib.dll RNG exports. */

typedef unsigned long DWORD;
typedef int BOOL;
typedef void *HANDLE;
typedef void *HMODULE;
typedef void *FARPROC;

#define WINAPI __stdcall
#define DLLIMPORT __declspec(dllimport)
#define GENERIC_WRITE 0x40000000UL
#define CREATE_ALWAYS 2UL
#define FILE_ATTRIBUTE_NORMAL 0x00000080UL

DLLIMPORT HANDLE WINAPI CreateFileA(const char *, DWORD, DWORD, void *, DWORD,
                                    DWORD, HANDLE);
DLLIMPORT BOOL WINAPI WriteFile(HANDLE, const void *, DWORD, DWORD *, void *);
DLLIMPORT BOOL WINAPI CloseHandle(HANDLE);
DLLIMPORT HMODULE WINAPI LoadLibraryA(const char *);
DLLIMPORT FARPROC WINAPI GetProcAddress(HMODULE, const char *);
DLLIMPORT BOOL WINAPI FreeLibrary(HMODULE);
DLLIMPORT void WINAPI ExitProcess(unsigned int);

typedef int(WINAPI *get_rand_fn)(int);
typedef int(WINAPI *seed_rand_fn)(int);

static HANDLE output;

static void write_bytes(const char *text, DWORD length) {
    DWORD written = 0;
    WriteFile(output, text, length, &written, 0);
}

static void write_u32(DWORD value) {
    char digits[10];
    char ordered[10];
    DWORD count = 0;
    DWORD index;
    do {
        digits[count++] = (char)('0' + value % 10);
        value /= 10;
    } while (value);
    for (index = 0; index < count; ++index)
        ordered[index] = digits[count - index - 1];
    write_bytes(ordered, count);
}

static void emit(DWORD seed, DWORD maximum, DWORD result) {
    write_u32(seed);
    write_bytes(" ", 1);
    write_u32(maximum);
    write_bytes(" ", 1);
    write_u32(result);
    write_bytes("\r\n", 2);
}

void mainCRTStartup(void) {
    static const DWORD seeds[] = {0, 1, 0x12345678UL, 0x3fffffffUL};
    static const DWORD maxima[] = {100, 12, 69, 404, 1000, 3, 100, 1000,
                                   5, 100};
    HMODULE dll = LoadLibraryA("DxLib.dll");
    seed_rand_fn seed_rand;
    get_rand_fn get_rand;
    DWORD seed_index;
    DWORD value_index;
    if (!dll)
        ExitProcess(2);
    seed_rand = (seed_rand_fn)GetProcAddress(dll, "_dx_SRand@4");
    get_rand = (get_rand_fn)GetProcAddress(dll, "_dx_GetRand@4");
    if (!seed_rand || !get_rand)
        ExitProcess(3);
    output = CreateFileA("dxlib-rng-oracle.txt", GENERIC_WRITE, 0, 0,
                         CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
    if ((DWORD)output == 0xffffffffUL)
        ExitProcess(4);
    for (seed_index = 0; seed_index < sizeof(seeds) / sizeof(seeds[0]);
         ++seed_index) {
        seed_rand((int)seeds[seed_index]);
        for (value_index = 0;
             value_index < sizeof(maxima) / sizeof(maxima[0]); ++value_index) {
            DWORD result = (DWORD)get_rand((int)maxima[value_index]);
            emit(seeds[seed_index], maxima[value_index], result);
        }
    }
    CloseHandle(output);
    FreeLibrary(dll);
    ExitProcess(0);
}
