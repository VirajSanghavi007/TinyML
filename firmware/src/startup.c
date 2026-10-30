// Minimal Cortex-M4F startup (RA4M1 / nRF52840): vector table, .data/.bss init, FPU on, C++ ctors, main().
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern void __libc_init_array(void);
extern int main(void);

void Reset_Handler(void);
static void Default_Handler(void) {
  for (;;) {
  }
}
void NMI_Handler(void) __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void) __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void) __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void) __attribute__((weak, alias("Default_Handler")));

__attribute__((section(".isr_vector"), used)) const void* const g_vectors[16] = {
    &_estack,          Reset_Handler,    NMI_Handler,      HardFault_Handler,
    MemManage_Handler, BusFault_Handler, UsageFault_Handler, 0,
    0,                 0,                0,                SVC_Handler,
    0,                 0,                PendSV_Handler,   SysTick_Handler,
};

#define STACK_PAINT 0xDEADBEEFu
#define SCB_CPACR (*(volatile uint32_t*)0xE000ED88u)

void Reset_Handler(void) {
  SCB_CPACR |= (0xFu << 20);  // CP10/CP11 full access: FPU on (kernels are built -mfloat-abi=hard)
  __asm volatile("dsb\n isb");
  for (uint32_t *src = &_sidata, *dst = &_sdata; dst < &_edata;) *dst++ = *src++;
  for (uint32_t* dst = &_sbss; dst < &_ebss;) *dst++ = 0;
  // Paint the free RAM above .bss (stack lives at the top) so StackHighWaterBytes() can measure it.
  uint32_t sp;
  __asm volatile("mov %0, sp" : "=r"(sp));
  for (uint32_t* p = &_ebss; p < (uint32_t*)(sp - 64);) *p++ = STACK_PAINT;
  __libc_init_array();
  main();
  for (;;) {
  }
}

// newlib-nano stubs we actually reach: no heap (TFLM never mallocs), no files.
void* _sbrk(int incr) {
  (void)incr;
  return (void*)-1;
}

// -nostartfiles drops crti/crtn; __libc_init_array still calls these.
void _init(void) {}
void _fini(void) {}

// Bytes of stack ever used: scan up from the end of .bss to the first overwritten paint word.
uint32_t StackHighWaterBytes(void) {
  uint32_t* p = &_ebss;
  while (p < &_estack && *p == STACK_PAINT) ++p;
  return (uint32_t)((uint8_t*)&_estack - (uint8_t*)p);
}
uint32_t StaticRamBytes(void) { return (uint32_t)((uint8_t*)&_ebss - (uint8_t*)&_sdata); }
