/*
 * Redistributable trace-forwarder for the shipped IriSu Box2D wrapper ABI.
 *
 * The authentic DLL is loaded from Box2D.real.dll beside this proxy.  This
 * file contains no original-game code or data.  It is freestanding so the
 * proxy has no statically linked C runtime and resolves _snprintf only for
 * integer-only JSON formatting.  Float arguments are recorded as raw bits;
 * formatting them as decimal inside the game could perturb x87 state.
 */

typedef unsigned long DWORD;
typedef int BOOL;
typedef void *HANDLE;
typedef void *HMODULE;
typedef void *FARPROC;

#define WINAPI __stdcall
#define CDECL __cdecl
#define DLLIMPORT __declspec(dllimport)
#define DLLEXPORT __declspec(dllexport)
#define INVALID_HANDLE_VALUE ((HANDLE)(-1))
#define GENERIC_WRITE 0x40000000UL
#define FILE_SHARE_READ 0x00000001UL
#define CREATE_NEW 1UL
#define FILE_ATTRIBUTE_NORMAL 0x00000080UL
#define DLL_PROCESS_DETACH 0UL
#define DLL_PROCESS_ATTACH 1UL
#define ENTRY_CAPACITY 4096UL
#define ALL_EXPORTS_MASK 0x0000ffffUL

DLLIMPORT HANDLE WINAPI CreateFileA(const char *, DWORD, DWORD, void *, DWORD,
                                    DWORD, HANDLE);
DLLIMPORT BOOL WINAPI WriteFile(HANDLE, const void *, DWORD, DWORD *, void *);
DLLIMPORT BOOL WINAPI CloseHandle(HANDLE);
DLLIMPORT BOOL WINAPI FlushFileBuffers(HANDLE);
DLLIMPORT DWORD WINAPI GetModuleFileNameA(HMODULE, char *, DWORD);
DLLIMPORT HMODULE WINAPI LoadLibraryA(const char *);
DLLIMPORT FARPROC WINAPI GetProcAddress(HMODULE, const char *);

/* Required by 32-bit MSVC-compatible linking when floating point is used. */
int _fltused = 0;

typedef int(CDECL *snprintf_fn)(char *, unsigned int, const char *, ...);
typedef int(WINAPI *init_fn)(float, float, float, float, float, float);
typedef void(WINAPI *dispose_fn)(void);
typedef void *(WINAPI *box_fn)(float, float, float, float, float, float, float,
                               float);
typedef void *(WINAPI *circle_fn)(float, float, float, float, float, float);
typedef void(WINAPI *destroy_fn)(void *);
typedef int(WINAPI *contact_fn)(void **, void **);
typedef float(WINAPI *scalar_fn)(void *);
typedef void(WINAPI *get_v_fn)(void *, float *, float *);
typedef void(WINAPI *set_position_fn)(void *, float, float, float);
typedef void(WINAPI *set_user_data_fn)(void *, void *);
typedef void(WINAPI *set_v_fn)(void *, float, float);
typedef void(WINAPI *step_fn)(float, int);
typedef void(WINAPI *test_fn)(void *);

struct Api {
    box_fn create_box;
    circle_fn create_circle;
    box_fn create_triangle;
    destroy_fn destroy_body;
    dispose_fn dispose;
    contact_fn get_contact;
    scalar_fn get_r;
    get_v_fn get_v;
    scalar_fn get_x;
    scalar_fn get_y;
    init_fn init;
    set_position_fn set_position;
    set_user_data_fn set_user_data;
    set_v_fn set_v;
    step_fn step;
    test_fn test;
};

struct Entry {
    void *body;
    void *user;
    DWORD ordinal;
    BOOL active;
};

union FloatBits {
    float value;
    DWORD bits;
};

static HMODULE self_module;
static HMODULE real_module;
static HANDLE log_file = INVALID_HANDLE_VALUE;
static snprintf_fn format;
static struct Api api;
static struct Entry entries[ENTRY_CAPACITY];
static DWORD load_attempted;
static BOOL proxy_ready;
static DWORD sequence;
static DWORD world_number;
static DWORD step_number;
static DWORD contact_call;
static DWORD create_number;
static DWORD entry_count;

static DWORD string_length(const char *text) {
    DWORD length = 0;
    while (text[length])
        ++length;
    return length;
}

static void write_line(const char *line) {
    DWORD written = 0;
    if (log_file != INVALID_HANDLE_VALUE)
        WriteFile(log_file, line, string_length(line), &written, 0);
}

static DWORD float_bits(float value) {
    union FloatBits bits;
    bits.value = value;
    return bits.bits;
}

static unsigned int x87_control_word(void) {
    unsigned short word;
    __asm__ __volatile__("fnstcw %0" : "=m"(word));
    return word;
}

static BOOL sibling_path(char *path, DWORD capacity, const char *leaf) {
    DWORD length = GetModuleFileNameA(self_module, path, capacity);
    DWORD leaf_length = string_length(leaf);
    if (length == 0 || length >= capacity)
        return 0;
    while (length > 0 && path[length - 1] != '\\' && path[length - 1] != '/')
        --length;
    if (length + leaf_length >= capacity)
        return 0;
    while (*leaf)
        path[length++] = *leaf++;
    path[length] = 0;
    return 1;
}

static FARPROC resolve(const char *name, DWORD bit, DWORD *mask) {
    FARPROC result = GetProcAddress(real_module, name);
    if (result)
        *mask |= bit;
    return result;
}

static void ensure_loaded(void) {
    HMODULE runtime;
    DWORD export_mask = 0;
    char real_path[512];
    char log_path[512];
    char line[256];

    if (load_attempted)
        return;
    load_attempted = 1;

    runtime = LoadLibraryA("msvcrt.dll");
    if (runtime)
        format = (snprintf_fn)GetProcAddress(runtime, "_snprintf");
    if (!format)
        return;

    if (sibling_path(log_path, sizeof(log_path), "box2d-trace.jsonl"))
        log_file = CreateFileA(log_path, GENERIC_WRITE, FILE_SHARE_READ, 0,
                               CREATE_NEW, FILE_ATTRIBUTE_NORMAL, 0);

    /* Evidence is append-never.  If a trace already exists (or cannot be
     * created), fail closed instead of running an untraced second process. */
    if (log_file == INVALID_HANDLE_VALUE)
        return;

    if (sibling_path(real_path, sizeof(real_path), "Box2D.real.dll"))
        real_module = LoadLibraryA(real_path);

    if (real_module) {
        api.create_box = (box_fn)resolve("_b2d_create_box@32", 1UL << 0,
                                        &export_mask);
        api.create_circle = (circle_fn)resolve("_b2d_create_circle@24", 1UL << 1,
                                              &export_mask);
        api.create_triangle = (box_fn)resolve("_b2d_create_triangle@32", 1UL << 2,
                                             &export_mask);
        api.destroy_body = (destroy_fn)resolve("_b2d_destroy_body@4", 1UL << 3,
                                              &export_mask);
        api.dispose = (dispose_fn)resolve("_b2d_dispose@0", 1UL << 4,
                                         &export_mask);
        api.get_contact = (contact_fn)resolve("_b2d_get_contact@8", 1UL << 5,
                                             &export_mask);
        api.get_r = (scalar_fn)resolve("_b2d_get_r@4", 1UL << 6, &export_mask);
        api.get_v = (get_v_fn)resolve("_b2d_get_v@12", 1UL << 7, &export_mask);
        api.get_x = (scalar_fn)resolve("_b2d_get_x@4", 1UL << 8, &export_mask);
        api.get_y = (scalar_fn)resolve("_b2d_get_y@4", 1UL << 9, &export_mask);
        api.init = (init_fn)resolve("_b2d_init@24", 1UL << 10, &export_mask);
        api.set_position = (set_position_fn)resolve("_b2d_set_position@16",
                                                   1UL << 11, &export_mask);
        api.set_user_data = (set_user_data_fn)resolve("_b2d_set_user_data@8",
                                                     1UL << 12, &export_mask);
        api.set_v = (set_v_fn)resolve("_b2d_set_v@12", 1UL << 13, &export_mask);
        api.step = (step_fn)resolve("_b2d_step@8", 1UL << 14, &export_mask);
        api.test = (test_fn)resolve("_b2d_test@4", 1UL << 15, &export_mask);
    }
    proxy_ready = export_mask == ALL_EXPORTS_MASK;

    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"proxy_loaded\",\"schema\":1,"
               "\"real_loaded\":%s,\"export_mask\":\"%08lx\","
               "\"x87_cw\":\"%04x\",\"ok\":%s}\n",
               sequence++, real_module ? "true" : "false", export_mask,
               x87_control_word(),
               proxy_ready ? "true" : "false");
        write_line(line);
    }
}

static long find_body(void *body) {
    long index;
    for (index = (long)entry_count - 1; index >= 0; --index)
        if (entries[index].active && entries[index].body == body)
            return index;
    return -1;
}

static DWORD ordinal_for_body(void *body) {
    long index = find_body(body);
    return index >= 0 ? entries[index].ordinal : 0;
}

static DWORD user_for_body(void *body) {
    long index = find_body(body);
    return index >= 0 ? (DWORD)entries[index].user : 0;
}

#if defined(IRISU_TRACE_GETTERS)
static void log_scalar_get(const char *field, void *body, float value) {
    char line[288];
    if (!format)
        return;
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"get_scalar\",\"world\":%lu,"
           "\"step\":%lu,\"field\":\"%s\",\"body\":%lu,"
           "\"ordinal\":%lu,\"user\":%lu,\"value_f32\":\"%08lx\"}\n",
           sequence++, world_number, step_number, field, (DWORD)body,
           ordinal_for_body(body), user_for_body(body), float_bits(value));
    write_line(line);
}

static void log_velocity_get(void *body, float x, float y) {
    char line[304];
    if (!format)
        return;
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"get_v\",\"world\":%lu,"
           "\"step\":%lu,\"body\":%lu,\"ordinal\":%lu,\"user\":%lu,"
           "\"args_f32\":[\"%08lx\",\"%08lx\"]}\n",
           sequence++, world_number, step_number, (DWORD)body,
           ordinal_for_body(body), user_for_body(body), float_bits(x),
           float_bits(y));
    write_line(line);
}
#endif

static DWORD ordinal_for_user(void *user) {
    long index;
    if (!user)
        return 0;
    for (index = (long)entry_count - 1; index >= 0; --index)
        /* Keep historical mappings: the game destroys bodies while walking
         * the DLL's cached contact cursor, which can still return that user. */
        if (entries[index].user == user)
            return entries[index].ordinal;
    return 0;
}

static void remember(void *body) {
    char line[192];
    if (entry_count < ENTRY_CAPACITY) {
        entries[entry_count].body = body;
        entries[entry_count].user = 0;
        entries[entry_count].ordinal = create_number;
        entries[entry_count].active = 1;
        ++entry_count;
    } else if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"mapping_overflow\","
               "\"world\":%lu,\"capacity\":%lu}\n",
               sequence++, world_number, ENTRY_CAPACITY);
        write_line(line);
    }
}

static void log_create8(const char *shape, void *body, float a, float b,
                        float c, float d, float e, float f, float g, float h) {
    char line[512];
    if (!format)
        return;
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"create\",\"world\":%lu,"
           "\"step\":%lu,\"shape\":\"%s\",\"ordinal\":%lu,"
           "\"body\":%lu,\"args_f32\":[\"%08lx\",\"%08lx\","
           "\"%08lx\",\"%08lx\",\"%08lx\",\"%08lx\","
           "\"%08lx\",\"%08lx\"]}\n",
           sequence++, world_number, step_number, shape, create_number,
           (DWORD)body, float_bits(a), float_bits(b), float_bits(c),
           float_bits(d), float_bits(e), float_bits(f), float_bits(g),
           float_bits(h));
    write_line(line);
}

static void log_create6(const char *shape, void *body, float a, float b,
                        float c, float d, float e, float f) {
    char line[448];
    if (!format)
        return;
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"create\",\"world\":%lu,"
           "\"step\":%lu,\"shape\":\"%s\",\"ordinal\":%lu,"
           "\"body\":%lu,\"args_f32\":[\"%08lx\",\"%08lx\","
           "\"%08lx\",\"%08lx\",\"%08lx\",\"%08lx\"]}\n",
           sequence++, world_number, step_number, shape, create_number,
           (DWORD)body, float_bits(a), float_bits(b), float_bits(c),
           float_bits(d), float_bits(e), float_bits(f));
    write_line(line);
}

DLLEXPORT int WINAPI b2d_init(float min_x, float min_y, float max_x,
                              float max_y, float gravity_y,
                              float magnification) {
    int result;
    unsigned int control_before;
    unsigned int control_after;
    char line[384];
    ensure_loaded();
    ++world_number;
    step_number = 0;
    contact_call = 0;
    create_number = 0;
    entry_count = 0;
    control_before = x87_control_word();
    result = proxy_ready ? api.init(min_x, min_y, max_x, max_y, gravity_y,
                                    magnification) : 0;
    control_after = x87_control_word();
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"init\",\"world\":%lu,"
               "\"args_f32\":[\"%08lx\",\"%08lx\",\"%08lx\","
               "\"%08lx\",\"%08lx\",\"%08lx\"],\"result\":%d,"
               "\"x87_cw_before\":\"%04x\","
               "\"x87_cw_after\":\"%04x\"}\n",
               sequence++, world_number, float_bits(min_x), float_bits(min_y),
               float_bits(max_x), float_bits(max_y), float_bits(gravity_y),
               float_bits(magnification), result, control_before,
               control_after);
        write_line(line);
    }
    return result;
}

DLLEXPORT void WINAPI b2d_dispose(void) {
    char line[160];
    ensure_loaded();
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"dispose\",\"world\":%lu,"
               "\"step\":%lu}\n",
               sequence++, world_number, step_number);
        write_line(line);
    }
    if (api.dispose)
        api.dispose();
    if (log_file != INVALID_HANDLE_VALUE)
        FlushFileBuffers(log_file);
}

DLLEXPORT void *WINAPI b2d_create_box(float width, float height, float x,
                                      float y, float radians, float density,
                                      float friction, float restitution) {
    void *body;
    ensure_loaded();
    ++create_number;
    body = api.create_box ? api.create_box(width, height, x, y, radians,
                                           density, friction, restitution) : 0;
    remember(body);
    log_create8("box", body, width, height, x, y, radians, density, friction,
                restitution);
    return body;
}

DLLEXPORT void *WINAPI b2d_create_triangle(float width, float height, float x,
                                           float y, float radians,
                                           float density, float friction,
                                           float restitution) {
    void *body;
    ensure_loaded();
    ++create_number;
    body = api.create_triangle ? api.create_triangle(width, height, x, y,
                                                     radians, density, friction,
                                                     restitution) : 0;
    remember(body);
    log_create8("triangle", body, width, height, x, y, radians, density,
                friction, restitution);
    return body;
}

DLLEXPORT void *WINAPI b2d_create_circle(float radius, float x, float y,
                                         float density, float friction,
                                         float restitution) {
    void *body;
    ensure_loaded();
    ++create_number;
    body = api.create_circle ? api.create_circle(radius, x, y, density,
                                                 friction, restitution) : 0;
    remember(body);
    log_create6("circle", body, radius, x, y, density, friction, restitution);
    return body;
}

DLLEXPORT void WINAPI b2d_destroy_body(void *body) {
    char line[256];
    long index;
    ensure_loaded();
    index = find_body(body);
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"destroy\",\"world\":%lu,"
               "\"step\":%lu,\"body\":%lu,\"ordinal\":%lu,"
               "\"user\":%lu}\n",
               sequence++, world_number, step_number, (DWORD)body,
               index >= 0 ? entries[index].ordinal : 0,
               index >= 0 ? (DWORD)entries[index].user : 0);
        write_line(line);
    }
    if (api.destroy_body)
        api.destroy_body(body);
    if (index >= 0)
        entries[index].active = 0;
}

DLLEXPORT int WINAPI b2d_get_contact(void **a, void **b) {
    int result;
    void *user_a;
    void *user_b;
    char line[320];
    ensure_loaded();
    result = api.get_contact ? api.get_contact(a, b) : 0;
    ++contact_call;
    user_a = result && a ? *a : 0;
    user_b = result && b ? *b : 0;
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"contact\",\"world\":%lu,"
               "\"step\":%lu,\"call\":%lu,\"result\":%s,"
               "\"a_user\":%lu,\"b_user\":%lu,\"a_ordinal\":%lu,"
               "\"b_ordinal\":%lu}\n",
               sequence++, world_number, step_number, contact_call,
               result ? "true" : "false", (DWORD)user_a, (DWORD)user_b,
               ordinal_for_user(user_a), ordinal_for_user(user_b));
        write_line(line);
    }
    return result;
}

DLLEXPORT float WINAPI b2d_get_r(void *body) {
    float value;
    ensure_loaded();
    value = api.get_r ? api.get_r(body) : 0.0f;
#if defined(IRISU_TRACE_GETTERS)
    log_scalar_get("r", body, value);
#endif
    return value;
}

DLLEXPORT void WINAPI b2d_get_v(void *body, float *x, float *y) {
    ensure_loaded();
    if (api.get_v)
        api.get_v(body, x, y);
#if defined(IRISU_TRACE_GETTERS)
    log_velocity_get(body, x ? *x : 0.0f, y ? *y : 0.0f);
#endif
}

DLLEXPORT float WINAPI b2d_get_x(void *body) {
    float value;
    ensure_loaded();
    value = api.get_x ? api.get_x(body) : 0.0f;
#if defined(IRISU_TRACE_GETTERS)
    log_scalar_get("x", body, value);
#endif
    return value;
}

DLLEXPORT float WINAPI b2d_get_y(void *body) {
    float value;
    ensure_loaded();
    value = api.get_y ? api.get_y(body) : 0.0f;
#if defined(IRISU_TRACE_GETTERS)
    log_scalar_get("y", body, value);
#endif
    return value;
}

DLLEXPORT void WINAPI b2d_set_position(void *body, float x, float y,
                                       float radians) {
    char line[320];
    ensure_loaded();
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"set_position\",\"world\":%lu,"
               "\"step\":%lu,\"body\":%lu,\"ordinal\":%lu,"
               "\"user\":%lu,\"args_f32\":[\"%08lx\",\"%08lx\","
               "\"%08lx\"]}\n",
               sequence++, world_number, step_number, (DWORD)body,
               ordinal_for_body(body), user_for_body(body), float_bits(x),
               float_bits(y), float_bits(radians));
        write_line(line);
    }
    if (api.set_position)
        api.set_position(body, x, y, radians);
}

DLLEXPORT void WINAPI b2d_set_user_data(void *body, void *user) {
    char line[256];
    long index;
    ensure_loaded();
    index = find_body(body);
    if (index >= 0)
        entries[index].user = user;
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"set_user_data\","
               "\"world\":%lu,\"step\":%lu,\"body\":%lu,"
               "\"ordinal\":%lu,\"user\":%lu}\n",
               sequence++, world_number, step_number, (DWORD)body,
               index >= 0 ? entries[index].ordinal : 0, (DWORD)user);
        write_line(line);
    }
    if (api.set_user_data)
        api.set_user_data(body, user);
}

DLLEXPORT void WINAPI b2d_set_v(void *body, float x, float y) {
    char line[288];
    ensure_loaded();
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"set_v\",\"world\":%lu,"
               "\"step\":%lu,\"body\":%lu,\"ordinal\":%lu,"
               "\"user\":%lu,\"args_f32\":[\"%08lx\",\"%08lx\"]}\n",
               sequence++, world_number, step_number, (DWORD)body,
               ordinal_for_body(body), user_for_body(body), float_bits(x),
               float_bits(y));
        write_line(line);
    }
    if (api.set_v)
        api.set_v(body, x, y);
}

DLLEXPORT void WINAPI b2d_step(float dt, int iterations) {
    char line[224];
    ensure_loaded();
    ++step_number;
    contact_call = 0;
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"step\",\"world\":%lu,"
               "\"step\":%lu,\"dt_f32\":\"%08lx\","
               "\"iterations\":%d}\n",
               sequence++, world_number, step_number, float_bits(dt),
               iterations);
        write_line(line);
    }
    if (api.step)
        api.step(dt, iterations);
}

DLLEXPORT void WINAPI b2d_test(void *body) {
    ensure_loaded();
    if (api.test)
        api.test(body);
}

BOOL WINAPI DllMainCRTStartup(void *instance, DWORD reason, void *reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH)
        self_module = (HMODULE)instance;
    if (reason == DLL_PROCESS_DETACH && log_file != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(log_file);
        CloseHandle(log_file);
        log_file = INVALID_HANDLE_VALUE;
    }
    return 1;
}
