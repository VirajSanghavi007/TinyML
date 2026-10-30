/* Nordic nRF52840 as on the Arduino Nano 33 BLE: 1 MB flash, 256 KB RAM.
   Bare-metal image linked at 0x0 (Renode, or SWD flashing which replaces the Arduino/mbed
   bootloader that normally occupies 0x0-0x10000). */
STACK_SIZE = DEFINED(STACK_SIZE) ? STACK_SIZE : 0x2000;

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 1024K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 256K
}

INCLUDE common_sections.ld
