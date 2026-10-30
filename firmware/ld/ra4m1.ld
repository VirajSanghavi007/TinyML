/* Renesas RA4M1 (R7FA4M1AB) as on the Arduino Uno R4 Minima: 256 KB code flash, 32 KB SRAM.
   Bare-metal image linked at 0x0 (Renode, or SWD flashing which replaces the Arduino bootloader
   that normally occupies 0x0-0x4000 and owns the option bytes). */
STACK_SIZE   = DEFINED(STACK_SIZE) ? STACK_SIZE : 0x800;

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 32K
}

INCLUDE common_sections.ld
