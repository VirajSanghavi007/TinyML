/* Shared section layout; included by the per-board scripts after MEMORY. */
ENTRY(Reset_Handler)

SECTIONS
{
  .isr_vector 0x00000000 :
  {
    KEEP(*(.isr_vector))
  } > FLASH

  /* RA4M1 option-setting memory must hold exactly these bytes at 0x400..0x44F, never code.
     Empty (and discarded) on boards without it. */
  .option_bytes 0x00000400 :
  {
    KEEP(*(.option_bytes))
  } > FLASH

  .text :
  {
    *(.text*)
    *(.rodata*)
    KEEP(*(.init))
    KEEP(*(.fini))
    . = ALIGN(4);
    __preinit_array_start = .; KEEP(*(.preinit_array)) __preinit_array_end = .;
    __init_array_start = .;    KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array)) __init_array_end = .;
    __fini_array_start = .;    KEEP(*(.fini_array)) __fini_array_end = .;
  } > FLASH

  .ARM.exidx : { *(.ARM.exidx* .gnu.linkonce.armexidx.*) } > FLASH

  _sidata = LOADADDR(.data);
  .data : ALIGN(4)
  {
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM AT > FLASH

  .bss (NOLOAD) : ALIGN(4)
  {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > RAM

  /* Stack at the top of RAM; the link fails if .data + .bss (incl. tensor_arena) + stack > 32 KB. */
  _estack = ORIGIN(RAM) + LENGTH(RAM);
  ASSERT(_ebss + STACK_SIZE <= _estack, "RAM overflow: .data + .bss (tensor_arena) + stack exceed RAM")
}
