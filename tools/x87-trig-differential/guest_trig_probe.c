#include <stdint.h>

static int syscall3(int number, int a, int b, int c) {
  int result;
  __asm__ volatile("int $0x80" : "=a"(result) : "a"(number), "b"(a), "c"(b), "d"(c) : "memory");
  return result;
}

static int open_read(const char *path) { return syscall3(5, (int)path, 0, 0); }
static int open_write(const char *path) { return syscall3(5, (int)path, 0x241, 0644); }
static int read_bytes(int fd, void *data, int size) { return syscall3(3, fd, (int)data, size); }
static int write_bytes(int fd, const void *data, int size) { return syscall3(4, fd, (int)data, size); }

static void put16(uint8_t **out, uint16_t value) {
  *(*out)++ = value;
  *(*out)++ = value >> 8;
}

static void put32(uint8_t **out, uint32_t value) {
  put16(out, value);
  put16(out, value >> 16);
}

static uint32_t bits(float value) {
  union { float f; uint32_t u; } cast = {value};
  return cast.u;
}

static int probe_main(int argc, char **argv) {
  if (argc != 3) return 2;
  int in = open_read(argv[1]);
  int out = open_write(argv[2]);
  if (in < 0 || out < 0) return 3;
  const uint16_t control = 0x027f;
  __asm__ volatile("fldcw %0" : : "m"(control));

  uint32_t input_bits;
  while (read_bytes(in, &input_bits, 4) == 4) {
    union { uint32_t u; float f; } input = {input_bits};
    float sine = 0, cosine = 0, pair_sine = 0, pair_cosine = 0, remaining = 0;
    uint16_t sine_status, cosine_status, pair_status;
    uint8_t has_pair = 0;
    __asm__ volatile(
        "fnclex\n\tflds %[in]\n\tfsin\n\tfnstsw %%ax\n\tmovw %%ax,%[sw]\n\tfstps %[out]"
        : [out] "=m"(sine), [sw] "=m"(sine_status) : [in] "m"(input.f) : "ax", "cc", "st");
    __asm__ volatile(
        "fnclex\n\tflds %[in]\n\tfcos\n\tfnstsw %%ax\n\tmovw %%ax,%[sw]\n\tfstps %[out]"
        : [out] "=m"(cosine), [sw] "=m"(cosine_status) : [in] "m"(input.f) : "ax", "cc", "st");
    __asm__ volatile(
        "fnclex\n\tflds %[in]\n\tfsincos\n\tfnstsw %%ax\n\tmovw %%ax,%[sw]\n\t"
        "testw $0x0400,%%ax\n\tjnz 1f\n\tfstps %[cos]\n\tfstps %[sin]\n\tmovb $1,%[pair]\n\tjmp 2f\n"
        "1:\n\tfstps %[remaining]\n2:"
        : [sin] "=m"(pair_sine), [cos] "=m"(pair_cosine),
          [remaining] "=m"(remaining), [sw] "=m"(pair_status), [pair] "+m"(has_pair)
        : [in] "m"(input.f) : "ax", "cc", "memory", "st");
    if (!has_pair) pair_sine = pair_cosine = 0;

    uint8_t record[31], *cursor = record;
    put32(&cursor, input_bits);
    put32(&cursor, bits(sine)); put16(&cursor, sine_status & 0x047f);
    put32(&cursor, bits(cosine)); put16(&cursor, cosine_status & 0x047f);
    put32(&cursor, bits(pair_sine)); put32(&cursor, bits(pair_cosine));
    put32(&cursor, bits(remaining)); put16(&cursor, pair_status & 0x047f);
    *cursor++ = has_pair;
    if (write_bytes(out, record, cursor - record) != cursor - record) return 4;
  }
  return 0;
}

__attribute__((naked, noreturn)) void _start(void) {
  __asm__ volatile(
      "movl %esp,%eax\n\tpushl %eax\n\tcall probe_start\n\tud2");
}

__attribute__((noreturn)) void probe_start(uint32_t *stack) {
  int status = probe_main(stack[0], (char **)(stack + 1));
  syscall3(1, status, 0, 0);
  __builtin_unreachable();
}
