#include <stdint.h>

#include "exact_host_image.inc"

enum { EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI };

static uint32_t registers[8];
static uint32_t eip;
static uint32_t zero_flag;
static uint32_t fpu_bits;
static uint32_t executed;

static void fail(void) { __builtin_trap(); }

static uint8_t load8(uint32_t address) {
  if (address >= sizeof(guest)) fail();
  return guest[address];
}

static uint16_t load16(uint32_t address) {
  return (uint16_t)(load8(address) | ((uint16_t)load8(address + 1u) << 8));
}

static uint32_t load32(uint32_t address) {
  return (uint32_t)load8(address) | ((uint32_t)load8(address + 1u) << 8) |
         ((uint32_t)load8(address + 2u) << 16) |
         ((uint32_t)load8(address + 3u) << 24);
}

static void store32(uint32_t address, uint32_t value) {
  if (address > sizeof(guest) - 4u) fail();
  guest[address] = (uint8_t)value;
  guest[address + 1u] = (uint8_t)(value >> 8);
  guest[address + 2u] = (uint8_t)(value >> 16);
  guest[address + 3u] = (uint8_t)(value >> 24);
}

static uint32_t decode_ea(uint8_t modrm) {
  uint32_t mod = modrm >> 6;
  uint32_t base = modrm & 7u;
  uint32_t address = 0;
  if (mod == 3u) fail();
  if (base == 4u) {
    uint8_t sib = load8(eip++);
    uint32_t scale = sib >> 6;
    uint32_t index = (sib >> 3) & 7u;
    base = sib & 7u;
    if (base == 5u && mod == 0u)
      address = load32(eip), eip += 4u;
    else
      address = registers[base];
    if (index != 4u) address += registers[index] << scale;
  } else if (base == 5u && mod == 0u) {
    address = load32(eip);
    eip += 4u;
  } else {
    address = registers[base];
  }
  if (mod == 1u) address += (uint32_t)(int32_t)(int8_t)load8(eip++);
  if (mod == 2u) address += load32(eip), eip += 4u;
  return address;
}

static uint32_t execute(uint32_t entry, uint32_t stack) {
  eip = entry;
  registers[ESP] = stack;
  zero_flag = 0;
  fpu_bits = 0;
  executed = 0;
  for (;;) {
    uint8_t opcode = load8(eip++);
    ++executed;
    if (opcode == 0x8b) {
      uint8_t modrm = load8(eip++);
      uint32_t destination = (modrm >> 3) & 7u;
      registers[destination] = load32(decode_ea(modrm));
    } else if (opcode == 0x85) {
      uint8_t modrm = load8(eip++);
      if ((modrm >> 6) != 3u) fail();
      zero_flag = (registers[modrm & 7u] & registers[(modrm >> 3) & 7u]) == 0;
    } else if (opcode == 0x39) {
      uint8_t modrm = load8(eip++);
      uint32_t right = registers[(modrm >> 3) & 7u];
      zero_flag = load32(decode_ea(modrm)) == right;
    } else if (opcode == 0x74 || opcode == 0x75) {
      int32_t displacement = (int8_t)load8(eip++);
      if ((opcode == 0x74 && zero_flag) || (opcode == 0x75 && !zero_flag))
        eip += (uint32_t)displacement;
    } else if (opcode == 0xd9) {
      uint8_t modrm = load8(eip++);
      uint32_t operation = (modrm >> 3) & 7u;
      uint32_t address = decode_ea(modrm);
      if (operation == 0u)
        fpu_bits = load32(address);
      else if (operation == 3u)
        store32(address, fpu_bits);
      else
        fail();
    } else if (opcode == 0xc2) {
      uint32_t result = load32(registers[ESP]);
      registers[ESP] += 4u + load16(eip);
      return result;
    } else {
      fail();
    }
  }
}

static uint32_t run_case(uint32_t owned) {
  uint32_t scratch = EXACT_HOST_IMAGE_SIZE;
  uint32_t handle = scratch + 0x100u;
  uint32_t body = scratch + 0x200u;
  uint32_t stack = scratch + 0x800u;
  uint32_t world = 0x12345678u;
  store32(handle, world);
  store32(body + 0x4cu, owned ? world : world + 1u);
  store32(body + 0x34u, 0xdeadbeefu);
  store32(stack, 0xfeedfaceu);
  store32(stack + 4u, handle);
  store32(stack + 8u, body);
  if (execute(EXACT_HOST_WORLD_TEST_ENTRY, stack) != 0xfeedfaceu) fail();
  return load32(body + 0x34u);
}

__attribute__((export_name("smoke_world_test"))) uint32_t smoke_world_test(void) {
  uint32_t total = 0;
  uint32_t owned = run_case(1);
  total += executed;
  uint32_t foreign = run_case(0);
  total += executed;
  executed = total;
  if (foreign != 0xdeadbeefu) fail();
  return owned;
}

__attribute__((export_name("smoke_instruction_count"))) uint32_t
smoke_instruction_count(void) {
  return executed;
}

__attribute__((export_name("host_image_size"))) uint32_t host_image_size(void) {
  return EXACT_HOST_IMAGE_SIZE;
}
