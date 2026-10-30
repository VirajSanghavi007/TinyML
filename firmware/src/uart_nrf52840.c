#include <stdint.h>

#include "uart.h"

// nRF52840 UARTE0 (EasyDMA). On the Nano 33 BLE this is Serial1: TX on P1.03 (header pin TX/D1).
#define UARTE0_BASE 0x40002000u
#define REG(off) (*(volatile uint32_t*)(UARTE0_BASE + (off)))
#define TASKS_STARTTX REG(0x008)
#define TASKS_STOPTX REG(0x00C)
#define EVENTS_ENDTX REG(0x120)
#define ENABLE REG(0x500)
#define PSEL_TXD REG(0x50C)
#define BAUDRATE REG(0x524)
#define TXD_PTR REG(0x544)
#define TXD_MAXCNT REG(0x548)
#define ENABLE_UARTE 8u
#define BAUD_115200 0x01D7E000u
#define PIN_P1_03 ((1u << 5) | 3u)

// EasyDMA can only read from RAM, so strings in flash are staged through this buffer.
static char tx_buf[128];

void UartInit(void) {
  PSEL_TXD = PIN_P1_03;
  BAUDRATE = BAUD_115200;
  ENABLE = ENABLE_UARTE;
}

static void Send(uint32_t n) {
  EVENTS_ENDTX = 0;
  TXD_PTR = (uint32_t)tx_buf;
  TXD_MAXCNT = n;
  TASKS_STARTTX = 1;
  while (EVENTS_ENDTX == 0) {
  }
  TASKS_STOPTX = 1;
}

void UartWrite(const char* s) {
  uint32_t n = 0;
  for (; *s; ++s) {
    if (*s == '\n') {
      if (n + 2 > sizeof(tx_buf)) Send(n), n = 0;
      tx_buf[n++] = '\r';
    } else if (n + 1 > sizeof(tx_buf)) {
      Send(n), n = 0;
    }
    tx_buf[n++] = *s;
  }
  if (n) Send(n);
}
