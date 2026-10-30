#include "uart.h"

#include <stdint.h>

// RA4M1 SCI2 registers (hardware manual, SCI chapter). 8-bit registers at 0x40070040.
#define SCI2_BASE 0x40070040u
#define SCI_SCR (*(volatile uint8_t*)(SCI2_BASE + 0x02))
#define SCI_TDR (*(volatile uint8_t*)(SCI2_BASE + 0x03))
#define SCI_SSR (*(volatile uint8_t*)(SCI2_BASE + 0x04))
#define SCR_TE (1u << 5)
#define SSR_TDRE (1u << 7)
// Module stop control: SCI2 is MSTPCRB bit 29 (1 = stopped after reset).
#define MSTPCRB (*(volatile uint32_t*)0x40047000u)

void UartInit(void) {
  // Enough for Renode. Real silicon also needs PRCR unlock, pin mux (P301/P302 PSEL) and BRR for the baud rate.
  MSTPCRB &= ~(1u << 29);
  SCI_SCR = SCR_TE;
}

static void PutChar(char c) {
  while ((SCI_SSR & SSR_TDRE) == 0) {
  }
  SCI_TDR = (uint8_t)c;
}

void UartWrite(const char* s) {
  for (; *s; ++s) {
    if (*s == '\n') PutChar('\r');
    PutChar(*s);
  }
}

// RA4M1 option-setting memory (OFS0/OFS1, security MPU) lives at 0x400 in code flash.
// All 0xFF = watchdogs off, no MPU regions, defaults: the erased state.
__attribute__((section(".option_bytes"), used)) const uint32_t g_option_bytes[20] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
};

