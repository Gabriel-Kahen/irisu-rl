/*
 * Behavioral probe for the shipped IriSu Syndrome v2.03 Box2D.dll.
 *
 * This is deliberately freestanding and dynamically resolves every wrapper
 * export.  The clean clone must never link to or distribute the original DLL.
 */

typedef unsigned long DWORD;
typedef int BOOL;
typedef void *HANDLE;
typedef void *HMODULE;
typedef void *FARPROC;

#define WINAPI __stdcall
#define CDECL __cdecl
#define DLLIMPORT __declspec(dllimport)
#define INVALID_HANDLE_VALUE ((HANDLE)(-1))
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
DLLIMPORT DWORD WINAPI GetLastError(void);
DLLIMPORT void WINAPI ExitProcess(unsigned int);

/* Required by 32-bit MSVC-compatible linking when floating point is used. */
int _fltused = 0;

typedef int(CDECL *snprintf_fn)(char *, unsigned int, const char *, ...);

typedef int(WINAPI *b2d_init_fn)(float, float, float, float, float, float);
typedef void(WINAPI *b2d_dispose_fn)(void);
typedef void *(WINAPI *b2d_create_box_fn)(float, float, float, float, float,
                                          float, float, float);
typedef void *(WINAPI *b2d_create_triangle_fn)(float, float, float, float, float,
                                               float, float, float);
typedef void *(WINAPI *b2d_create_circle_fn)(float, float, float, float, float,
                                             float);
typedef void(WINAPI *b2d_destroy_body_fn)(void *);
typedef void(WINAPI *b2d_step_fn)(float, int);
typedef int(WINAPI *b2d_get_contact_fn)(void **, void **);
typedef float(WINAPI *b2d_get_scalar_fn)(void *);
typedef void(WINAPI *b2d_get_v_fn)(void *, float *, float *);
typedef void(WINAPI *b2d_set_v_fn)(void *, float, float);
typedef void(WINAPI *b2d_set_user_data_fn)(void *, void *);
typedef void(WINAPI *b2d_set_position_fn)(void *, float, float, float);
typedef void(WINAPI *b2d_test_fn)(void *);

struct Api {
    b2d_init_fn init;
    b2d_dispose_fn dispose;
    b2d_create_box_fn create_box;
    b2d_create_triangle_fn create_triangle;
    b2d_create_circle_fn create_circle;
    b2d_destroy_body_fn destroy_body;
    b2d_step_fn step;
    b2d_get_contact_fn get_contact;
    b2d_get_scalar_fn get_x;
    b2d_get_scalar_fn get_y;
    b2d_get_scalar_fn get_r;
    b2d_get_v_fn get_v;
    b2d_set_v_fn set_v;
    b2d_set_user_data_fn set_user_data;
    b2d_set_position_fn set_position;
    b2d_test_fn test;
};

static HANDLE out_file;
static snprintf_fn format;
static unsigned long sequence;

static DWORD string_length(const char *s) {
    DWORD n = 0;
    while (s[n])
        ++n;
    return n;
}

static void write_text(const char *s) {
    DWORD written = 0;
    WriteFile(out_file, s, string_length(s), &written, 0);
}

static void write_error(const char *stage, DWORD code) {
    char line[256];
    if (format) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"error\",\"stage\":\"%s\","
               "\"win32_error\":%lu}\n",
               sequence++, stage, code);
        write_text(line);
    }
}

static FARPROC require_export(HMODULE dll, const char *logical,
                              const char *decorated) {
    FARPROC result = GetProcAddress(dll, decorated);
    char line[384];
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"export\",\"name\":\"%s\","
           "\"symbol\":\"%s\",\"resolved\":%s}\n",
           sequence++, logical, decorated, result ? "true" : "false");
    write_text(line);
    return result;
}

static int load_api(HMODULE dll, struct Api *api) {
#define LOAD(field, type, name, bytes)                                          \
    api->field = (type)require_export(dll, #name, "_" #name "@" #bytes);       \
    if (!api->field)                                                            \
        return 0
    LOAD(init, b2d_init_fn, b2d_init, 24);
    LOAD(dispose, b2d_dispose_fn, b2d_dispose, 0);
    LOAD(create_box, b2d_create_box_fn, b2d_create_box, 32);
    LOAD(create_triangle, b2d_create_triangle_fn, b2d_create_triangle, 32);
    LOAD(create_circle, b2d_create_circle_fn, b2d_create_circle, 24);
    LOAD(destroy_body, b2d_destroy_body_fn, b2d_destroy_body, 4);
    LOAD(step, b2d_step_fn, b2d_step, 8);
    LOAD(get_contact, b2d_get_contact_fn, b2d_get_contact, 8);
    LOAD(get_x, b2d_get_scalar_fn, b2d_get_x, 4);
    LOAD(get_y, b2d_get_scalar_fn, b2d_get_y, 4);
    LOAD(get_r, b2d_get_scalar_fn, b2d_get_r, 4);
    LOAD(get_v, b2d_get_v_fn, b2d_get_v, 12);
    LOAD(set_v, b2d_set_v_fn, b2d_set_v, 12);
    LOAD(set_user_data, b2d_set_user_data_fn, b2d_set_user_data, 8);
    LOAD(set_position, b2d_set_position_fn, b2d_set_position, 16);
    LOAD(test, b2d_test_fn, b2d_test, 4);
#undef LOAD
    return 1;
}

static void emit_lifecycle(const char *scenario, const char *operation,
                           int result) {
    char line[320];
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"lifecycle\","
           "\"scenario\":\"%s\",\"operation\":\"%s\","
           "\"result\":%d}\n",
           sequence++, scenario, operation, result);
    write_text(line);
}

static void emit_body(struct Api *api, const char *scenario, int tick,
                      unsigned long id, const char *shape, void *body) {
    union FloatBits {
        float value;
        DWORD bits;
    } x_bits, y_bits, r_bits, vx_bits, vy_bits;
    float x = api->get_x(body);
    float y = api->get_y(body);
    float r = api->get_r(body);
    float vx = 0.0f;
    float vy = 0.0f;
    char line[900];
    api->get_v(body, &vx, &vy);
    x_bits.value = x;
    y_bits.value = y;
    r_bits.value = r;
    vx_bits.value = vx;
    vy_bits.value = vy;
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"body\",\"scenario\":\"%s\","
           "\"tick\":%d,\"id\":%lu,\"shape\":\"%s\","
           "\"x\":%.9g,\"y\":%.9g,\"r\":%.9g,"
           "\"vx_world\":%.9g,\"vy_world\":%.9g,"
           "\"x_bits\":%lu,\"y_bits\":%lu,\"r_bits\":%lu,"
           "\"vx_bits\":%lu,\"vy_bits\":%lu}\n",
           sequence++, scenario, tick, id, shape, (double)x, (double)y,
           (double)r, (double)vx, (double)vy, x_bits.bits, y_bits.bits,
           r_bits.bits, vx_bits.bits, vy_bits.bits);
    write_text(line);
}

static int emit_contacts(struct Api *api, const char *scenario, int tick) {
    void *a;
    void *b;
    int index = 0;
    char line[384];
    while (api->get_contact(&a, &b)) {
        format(line, sizeof(line),
               "{\"seq\":%lu,\"type\":\"contact\","
               "\"scenario\":\"%s\",\"tick\":%d,\"index\":%d,"
               "\"user_a\":%lu,\"user_b\":%lu}\n",
               sequence++, scenario, tick, index++, (unsigned long)a,
               (unsigned long)b);
        write_text(line);
    }
    return index;
}

static void emit_contact_count(const char *scenario, int tick, int count) {
    char line[320];
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"contact_count\","
           "\"scenario\":\"%s\",\"tick\":%d,\"count\":%d}\n",
           sequence++, scenario, tick, count);
    write_text(line);
}

static int initialize(struct Api *api, const char *scenario, float min_x,
                      float min_y, float max_x, float max_y, float gravity_y,
                      float magnification) {
    int result = api->init(min_x, min_y, max_x, max_y, gravity_y,
                           magnification);
    char line[640];
    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"init\",\"scenario\":\"%s\","
           "\"min_x\":%.9g,\"min_y\":%.9g,\"max_x\":%.9g,"
           "\"max_y\":%.9g,\"gravity_y\":%.9g,"
           "\"magnification\":%.9g,\"result\":%d}\n",
           sequence++, scenario, (double)min_x, (double)min_y, (double)max_x,
           (double)max_y, (double)gravity_y, (double)magnification, result);
    write_text(line);
    return result;
}

static void scenario_transforms(struct Api *api) {
    const char *name = "transforms";
    void *box;
    void *circle;
    void *triangle;
    int contacts;
    if (!initialize(api, name, -1000.0f, -1000.0f, 1000.0f, 1000.0f, 0.0f,
                    100.0f))
        return;

    box = api->create_box(40.0f, 20.0f, 123.0f, 234.0f, 0.25f, 1.0f,
                          0.7f, 0.2f);
    circle = api->create_circle(12.0f, -45.0f, 67.0f, 1.0f, 0.4f, 0.3f);
    triangle = api->create_triangle(60.0f, 30.0f, -210.0f, -120.0f,
                                    -0.75f, 1.0f, 0.8f, 0.1f);
    api->set_user_data(box, (void *)101UL);
    api->set_user_data(circle, (void *)102UL);
    api->set_user_data(triangle, (void *)103UL);
    emit_body(api, name, 0, 101, "box", box);
    emit_body(api, name, 0, 102, "circle", circle);
    emit_body(api, name, 0, 103, "triangle", triangle);

    api->set_v(box, 250.0f, -500.0f);
    emit_body(api, name, 0, 101, "box_after_set_v", box);
    api->step(0.02f, 10);
    contacts = emit_contacts(api, name, 1);
    emit_contact_count(name, 1, contacts);
    emit_body(api, name, 1, 101, "box_after_step", box);

    api->set_position(box, 321.0f, 222.0f, -0.5f);
    emit_body(api, name, 1, 101, "box_after_set_position", box);
    api->set_v(box, 100.0f, 200.0f);
    api->test(box);
    emit_body(api, name, 1, 101, "box_after_test", box);
    api->step(0.02f, 10);
    emit_contacts(api, name, 2);
    emit_body(api, name, 2, 101, "box_after_test_step", box);

    api->destroy_body(triangle);
    emit_lifecycle(name, "destroy_triangle", 1);
    api->destroy_body(circle);
    emit_lifecycle(name, "destroy_circle", 1);
    api->destroy_body(box);
    emit_lifecycle(name, "destroy_box", 1);
    api->destroy_body(0);
    emit_lifecycle(name, "destroy_null", 1);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static int is_sample_tick(int tick) {
    return tick == 0 || tick == 1 || tick == 10 || tick == 25 || tick == 50 ||
           tick == 75 || tick == 100 || tick == 125 || tick == 150 ||
           tick == 200;
}

static void scenario_fall_and_contacts(struct Api *api) {
    const char *name = "fall_and_contacts";
    void *floor;
    void *box;
    void *circle;
    void *triangle;
    int tick;
    int count;
    if (!initialize(api, name, -1000.0f, -1000.0f, 1000.0f, 1000.0f,
                    100.0f, 100.0f))
        return;

    floor = api->create_box(800.0f, 20.0f, 0.0f, 300.0f, 0.0f, 0.0f, 1.0f,
                            0.0f);
    box = api->create_box(40.0f, 20.0f, -220.0f, 100.0f, 0.0f, 1.0f, 1.0f,
                          0.0f);
    circle = api->create_circle(12.0f, 0.0f, 100.0f, 1.0f, 1.0f, 0.0f);
    triangle = api->create_triangle(60.0f, 30.0f, 220.0f, 100.0f, 0.0f, 1.0f,
                                    1.0f, 0.0f);
    api->set_user_data(floor, (void *)9001UL);
    api->set_user_data(box, (void *)201UL);
    api->set_user_data(circle, (void *)202UL);
    api->set_user_data(triangle, (void *)203UL);

    emit_body(api, name, 0, 9001, "static_floor", floor);
    emit_body(api, name, 0, 201, "box", box);
    emit_body(api, name, 0, 202, "circle", circle);
    emit_body(api, name, 0, 203, "triangle", triangle);
    for (tick = 1; tick <= 200; ++tick) {
        api->step(0.02f, 10);
        count = emit_contacts(api, name, tick);
        if (count || is_sample_tick(tick))
            emit_contact_count(name, tick, count);
        if (is_sample_tick(tick)) {
            emit_body(api, name, tick, 201, "box", box);
            emit_body(api, name, tick, 202, "circle", circle);
            emit_body(api, name, tick, 203, "triangle", triangle);
        }
    }

    api->destroy_body(box);
    api->destroy_body(circle);
    api->destroy_body(triangle);
    api->destroy_body(floor);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static void scenario_restitution(struct Api *api) {
    const char *name = "restitution";
    void *wall;
    void *circle;
    int tick;
    int count;
    if (!initialize(api, name, -1000.0f, -1000.0f, 1000.0f, 1000.0f, 0.0f,
                    100.0f))
        return;
    wall = api->create_box(20.0f, 500.0f, 300.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                           1.0f);
    circle = api->create_circle(12.0f, 100.0f, 0.0f, 1.0f, 0.0f, 0.5f);
    api->set_user_data(wall, (void *)9002UL);
    api->set_user_data(circle, (void *)301UL);
    api->set_v(circle, 500.0f, 0.0f);
    emit_body(api, name, 0, 9002, "static_wall", wall);
    emit_body(api, name, 0, 301, "circle", circle);
    for (tick = 1; tick <= 80; ++tick) {
        api->step(0.02f, 10);
        count = emit_contacts(api, name, tick);
        if (count)
            emit_contact_count(name, tick, count);
        if (tick == 1 || tick == 10 || tick == 20 || tick == 30 || tick == 36 ||
            tick == 38 || tick == 40 || tick == 50 || tick == 60 || tick == 80)
            emit_body(api, name, tick, 301, "circle", circle);
    }
    api->destroy_body(circle);
    api->destroy_body(wall);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static void scenario_friction_response(struct Api *api) {
    const char *name = "friction_response";
    void *floor_zero;
    void *floor_half_a;
    void *floor_one;
    void *floor_half_b;
    void *box_zero;
    void *box_half_a;
    void *box_one;
    void *box_half_b;
    int tick;
    if (!initialize(api, name, -1200.0f, -1000.0f, 1200.0f, 1000.0f,
                    100.0f, 100.0f))
        return;

    floor_zero = api->create_box(180.0f, 20.0f, -600.0f, 300.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f);
    floor_half_a = api->create_box(180.0f, 20.0f, -200.0f, 300.0f, 0.0f,
                                   0.0f, 1.0f, 0.0f);
    floor_one = api->create_box(180.0f, 20.0f, 200.0f, 300.0f, 0.0f, 0.0f,
                                1.0f, 0.0f);
    floor_half_b = api->create_box(180.0f, 20.0f, 600.0f, 300.0f, 0.0f,
                                   0.0f, 0.25f, 0.0f);
    box_zero = api->create_box(20.0f, 20.0f, -640.0f, 280.5f, 0.0f, 1.0f,
                               0.0f, 0.0f);
    box_half_a = api->create_box(20.0f, 20.0f, -240.0f, 280.5f, 0.0f, 1.0f,
                                 0.25f, 0.0f);
    box_one = api->create_box(20.0f, 20.0f, 160.0f, 280.5f, 0.0f, 1.0f, 1.0f,
                              0.0f);
    box_half_b = api->create_box(20.0f, 20.0f, 560.0f, 280.5f, 0.0f, 1.0f,
                                 1.0f, 0.0f);
    api->set_user_data(floor_zero, (void *)9101UL);
    api->set_user_data(floor_half_a, (void *)9102UL);
    api->set_user_data(floor_one, (void *)9103UL);
    api->set_user_data(floor_half_b, (void *)9104UL);
    api->set_user_data(box_zero, (void *)401UL);
    api->set_user_data(box_half_a, (void *)402UL);
    api->set_user_data(box_one, (void *)403UL);
    api->set_user_data(box_half_b, (void *)404UL);
    api->set_v(box_zero, 100.0f, 0.0f);
    api->set_v(box_half_a, 100.0f, 0.0f);
    api->set_v(box_one, 100.0f, 0.0f);
    api->set_v(box_half_b, 100.0f, 0.0f);
    for (tick = 0; tick <= 20; ++tick) {
        if (tick) {
            api->step(0.02f, 10);
            if (tick == 1 || tick == 20)
                emit_contact_count(name, tick, emit_contacts(api, name, tick));
        }
        emit_body(api, name, tick, 401, "mu_0", box_zero);
        emit_body(api, name, tick, 402, "mu_sqrt_0.25", box_half_a);
        emit_body(api, name, tick, 403, "mu_1", box_one);
        emit_body(api, name, tick, 404, "mu_sqrt_0.25_symmetric", box_half_b);
    }
    api->destroy_body(box_zero);
    api->destroy_body(box_half_a);
    api->destroy_body(box_one);
    api->destroy_body(box_half_b);
    api->destroy_body(floor_zero);
    api->destroy_body(floor_half_a);
    api->destroy_body(floor_one);
    api->destroy_body(floor_half_b);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static void scenario_sleep_timing(struct Api *api) {
    const char *name = "sleep_timing";
    void *floor;
    void *probe23;
    void *probe24;
    void *probe25;
    void *probe26;
    void *control;
    int tick;
    if (!initialize(api, name, -1000.0f, -1000.0f, 1000.0f, 1000.0f,
                    100.0f, 100.0f))
        return;
    floor = api->create_box(1100.0f, 20.0f, 0.0f, 300.0f, 0.0f, 0.0f, 1.0f,
                            0.0f);
    probe23 = api->create_box(20.0f, 20.0f, -400.0f, 280.5f, 0.0f, 1.0f,
                              0.0f, 0.0f);
    probe24 = api->create_box(20.0f, 20.0f, -200.0f, 280.5f, 0.0f, 1.0f,
                              0.0f, 0.0f);
    probe25 = api->create_box(20.0f, 20.0f, 0.0f, 280.5f, 0.0f, 1.0f, 0.0f,
                              0.0f);
    probe26 = api->create_box(20.0f, 20.0f, 200.0f, 280.5f, 0.0f, 1.0f,
                              0.0f, 0.0f);
    control = api->create_box(20.0f, 20.0f, 400.0f, 280.5f, 0.0f, 1.0f, 0.0f,
                              0.0f);
    api->set_user_data(floor, (void *)9201UL);
    api->set_user_data(probe23, (void *)501UL);
    api->set_user_data(probe24, (void *)502UL);
    api->set_user_data(probe25, (void *)503UL);
    api->set_user_data(probe26, (void *)504UL);
    api->set_user_data(control, (void *)505UL);
    for (tick = 1; tick <= 35; ++tick) {
        api->step(0.02f, 10);
        if (tick == 1)
            emit_contact_count(name, tick, emit_contacts(api, name, tick));
        if (tick == 23) {
            api->set_v(probe23, 100.0f, 0.0f);
            emit_body(api, name, tick, 501, "set_v_after_tick_23", probe23);
        }
        if (tick == 24) {
            api->set_v(probe24, 100.0f, 0.0f);
            emit_body(api, name, tick, 502, "set_v_after_tick_24", probe24);
        }
        if (tick == 25) {
            api->set_v(probe25, 100.0f, 0.0f);
            emit_body(api, name, tick, 503, "set_v_after_tick_25", probe25);
        }
        if (tick == 26) {
            api->set_v(probe26, 100.0f, 0.0f);
            emit_body(api, name, tick, 504, "set_v_after_tick_26", probe26);
        }
        if (tick >= 22 && tick <= 28) {
            emit_body(api, name, tick, 501, "probe_23", probe23);
            emit_body(api, name, tick, 502, "probe_24", probe24);
            emit_body(api, name, tick, 503, "probe_25", probe25);
            emit_body(api, name, tick, 504, "probe_26", probe26);
        }
        emit_body(api, name, tick, 505, "control", control);
    }
    api->destroy_body(probe23);
    api->destroy_body(probe24);
    api->destroy_body(probe25);
    api->destroy_body(probe26);
    api->destroy_body(control);
    api->destroy_body(floor);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static void scenario_triangle_orientation(struct Api *api) {
    const char *name = "triangle_orientation";
    void *triangle;
    void *inside_left;
    void *inside_bottom;
    void *outside_right;
    void *outside_top;
    void *left_touch;
    void *left_miss;
    void *bottom_touch;
    void *bottom_miss;
    int count;
    if (!initialize(api, name, -500.0f, -500.0f, 500.0f, 500.0f, 0.0f,
                    100.0f))
        return;
    triangle = api->create_triangle(100.0f, 60.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f);
    inside_left = api->create_circle(2.0f, -35.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    inside_bottom = api->create_circle(2.0f, 0.0f, 25.0f, 1.0f, 0.0f, 0.0f);
    outside_right = api->create_circle(2.0f, 35.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    outside_top = api->create_circle(2.0f, 0.0f, -25.0f, 1.0f, 0.0f, 0.0f);
    left_touch = api->create_circle(2.0f, -51.0f, -15.0f, 1.0f, 0.0f, 0.0f);
    left_miss = api->create_circle(2.0f, -53.0f, 15.0f, 1.0f, 0.0f, 0.0f);
    bottom_touch = api->create_circle(2.0f, -20.0f, 31.0f, 1.0f, 0.0f, 0.0f);
    bottom_miss = api->create_circle(2.0f, 20.0f, 33.0f, 1.0f, 0.0f, 0.0f);
    api->set_user_data(triangle, (void *)9301UL);
    api->set_user_data(inside_left, (void *)601UL);
    api->set_user_data(inside_bottom, (void *)602UL);
    api->set_user_data(outside_right, (void *)603UL);
    api->set_user_data(outside_top, (void *)604UL);
    api->set_user_data(left_touch, (void *)605UL);
    api->set_user_data(left_miss, (void *)606UL);
    api->set_user_data(bottom_touch, (void *)607UL);
    api->set_user_data(bottom_miss, (void *)608UL);
    api->step(0.0f, 10);
    count = emit_contacts(api, name, 0);
    emit_contact_count(name, 0, count);
    api->destroy_body(inside_left);
    api->destroy_body(inside_bottom);
    api->destroy_body(outside_right);
    api->destroy_body(outside_top);
    api->destroy_body(left_touch);
    api->destroy_body(left_miss);
    api->destroy_body(bottom_touch);
    api->destroy_body(bottom_miss);
    api->destroy_body(triangle);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static void scenario_dimension_skin(struct Api *api) {
    const char *name = "dimension_skin";
    void *floor;
    void *box20;
    void *box40;
    void *circle10;
    void *circle20;
    int tick;
    int count;
    if (!initialize(api, name, -1000.0f, -1000.0f, 1000.0f, 1000.0f,
                    100.0f, 100.0f))
        return;
    floor = api->create_box(1000.0f, 20.0f, 0.0f, 300.0f, 0.0f, 0.0f, 0.0f,
                            0.0f);
    box20 = api->create_box(20.0f, 20.0f, -300.0f, 100.0f, 0.0f, 1.0f, 0.0f,
                            0.0f);
    box40 = api->create_box(20.0f, 40.0f, -100.0f, 100.0f, 0.0f, 1.0f, 0.0f,
                            0.0f);
    circle10 = api->create_circle(10.0f, 100.0f, 100.0f, 1.0f, 0.0f, 0.0f);
    circle20 = api->create_circle(20.0f, 300.0f, 100.0f, 1.0f, 0.0f, 0.0f);
    api->set_user_data(floor, (void *)9401UL);
    api->set_user_data(box20, (void *)801UL);
    api->set_user_data(box40, (void *)802UL);
    api->set_user_data(circle10, (void *)803UL);
    api->set_user_data(circle20, (void *)804UL);
    for (tick = 1; tick <= 200; ++tick) {
        api->step(0.02f, 10);
        count = emit_contacts(api, name, tick);
        if (count)
            emit_contact_count(name, tick, count);
        if (tick == 1 || tick == 75 || tick == 90 || tick == 95 ||
            tick == 100 || tick == 125 || tick == 200) {
            emit_body(api, name, tick, 801, "box_height_20", box20);
            emit_body(api, name, tick, 802, "box_height_40", box40);
            emit_body(api, name, tick, 803, "circle_radius_10", circle10);
            emit_body(api, name, tick, 804, "circle_radius_20", circle20);
        }
    }
    api->destroy_body(box20);
    api->destroy_body(box40);
    api->destroy_body(circle10);
    api->destroy_body(circle20);
    api->destroy_body(floor);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static void gravity_trial(struct Api *api, const char *name, float gravity,
                          float magnification, unsigned long id) {
    void *box;
    if (!initialize(api, name, -1000.0f, -1000.0f, 1000.0f, 1000.0f,
                    gravity, magnification))
        return;
    box = api->create_box(20.0f, 20.0f, 0.0f, 100.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
    api->set_user_data(box, (void *)id);
    emit_body(api, name, 0, id, "box", box);
    api->step(0.02f, 10);
    emit_body(api, name, 1, id, "box", box);
    api->step(0.02f, 10);
    emit_body(api, name, 2, id, "box", box);
    api->destroy_body(box);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static void scenario_contact_ordering(struct Api *api) {
    const char *name = "contact_ordering";
    void *floor;
    void *box1;
    void *box2;
    void *box3;
    void *box4;
    void *box5;
    void *box6;
    int count;
    if (!initialize(api, name, -1000.0f, -1000.0f, 1000.0f, 1000.0f,
                    100.0f, 100.0f))
        return;
    floor = api->create_box(1400.0f, 20.0f, 0.0f, 300.0f, 0.0f, 0.0f, 0.0f,
                            0.0f);
    box1 = api->create_box(20.0f, 20.0f, -500.0f, 280.5f, 0.0f, 1.0f, 0.0f,
                           0.0f);
    box2 = api->create_box(20.0f, 20.0f, -300.0f, 280.5f, 0.0f, 1.0f, 0.0f,
                           0.0f);
    box3 = api->create_box(20.0f, 20.0f, -100.0f, 280.5f, 0.0f, 1.0f, 0.0f,
                           0.0f);
    api->set_user_data(floor, (void *)9501UL);
    api->set_user_data(box1, (void *)701UL);
    api->set_user_data(box2, (void *)702UL);
    api->set_user_data(box3, (void *)703UL);
    api->step(0.02f, 10);
    count = emit_contacts(api, name, 1);
    emit_contact_count(name, 1, count);
    box4 = api->create_box(20.0f, 20.0f, 100.0f, 280.5f, 0.0f, 1.0f, 0.0f,
                           0.0f);
    api->set_user_data(box4, (void *)704UL);
    api->step(0.02f, 10);
    count = emit_contacts(api, name, 2);
    emit_contact_count(name, 2, count);
    box5 = api->create_box(20.0f, 20.0f, 300.0f, 280.5f, 0.0f, 1.0f, 0.0f,
                           0.0f);
    box6 = api->create_box(20.0f, 20.0f, 500.0f, 280.5f, 0.0f, 1.0f, 0.0f,
                           0.0f);
    api->set_user_data(box5, (void *)705UL);
    api->set_user_data(box6, (void *)706UL);
    api->step(0.02f, 10);
    count = emit_contacts(api, name, 3);
    emit_contact_count(name, 3, count);
    api->destroy_body(box1);
    api->destroy_body(box2);
    api->destroy_body(box3);
    api->destroy_body(box4);
    api->destroy_body(box5);
    api->destroy_body(box6);
    api->destroy_body(floor);
    api->dispose();
    emit_lifecycle(name, "dispose", 1);
}

static int run_probe(void) {
    HMODULE runtime;
    HMODULE dll;
    struct Api api;
    char line[384];

    out_file = CreateFileA("box2d-probe.jsonl", GENERIC_WRITE, 0, 0,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
    if (out_file == INVALID_HANDLE_VALUE)
        return 10;

    runtime = LoadLibraryA("msvcrt.dll");
    if (!runtime)
        return 11;
    format = (snprintf_fn)GetProcAddress(runtime, "_snprintf");
    if (!format)
        return 12;

    format(line, sizeof(line),
           "{\"seq\":%lu,\"type\":\"meta\",\"schema\":2,"
           "\"probe\":\"irisu-box2d-oracle\",\"architecture\":\"x86\","
           "\"dll_filename\":\"Box2D.dll\"}\n",
           sequence++);
    write_text(line);

    dll = LoadLibraryA("Box2D.dll");
    if (!dll) {
        write_error("LoadLibraryA(Box2D.dll)", GetLastError());
        CloseHandle(out_file);
        return 20;
    }
    if (!load_api(dll, &api)) {
        write_error("GetProcAddress", GetLastError());
        FreeLibrary(dll);
        CloseHandle(out_file);
        return 21;
    }

    scenario_transforms(&api);
    scenario_fall_and_contacts(&api);
    scenario_restitution(&api);
    scenario_friction_response(&api);
    scenario_sleep_timing(&api);
    scenario_triangle_orientation(&api);
    scenario_dimension_skin(&api);
    gravity_trial(&api, "gravity_g100_m100", 100.0f, 100.0f, 901UL);
    gravity_trial(&api, "gravity_g250_m100", 250.0f, 100.0f, 902UL);
    gravity_trial(&api, "gravity_g100_m50", 100.0f, 50.0f, 903UL);
    gravity_trial(&api, "gravity_gminus250_m100", -250.0f, 100.0f, 904UL);
    scenario_contact_ordering(&api);

    FreeLibrary(dll);
    emit_lifecycle("probe", "FreeLibrary", 1);
    CloseHandle(out_file);
    FreeLibrary(runtime);
    return 0;
}

void WINAPI mainCRTStartup(void) { ExitProcess((unsigned int)run_probe()); }
