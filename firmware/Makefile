# Bare-metal TFLM firmware, Cortex-M4F boards. Build inside WSL/Linux:
#   make BOARD=nano33ble ARENA_KB=30      (Arduino Nano 33 BLE, nRF52840, 1 MB / 256 KB)
#   make BOARD=unor4 ARENA_KB=30          (Arduino Uno R4 Minima, RA4M1, 256 KB / 32 KB)
TFLM ?= $(HOME)/tflite-micro
BOARD ?= nano33ble
ARENA_KB ?= 30
BUILD ?= build/$(BOARD)_arena$(ARENA_KB)

ifeq ($(BOARD),nano33ble)
  LDSCRIPT := ld/nrf52840.ld
  UART_SRC := uart_nrf52840
  RAM_KB := 256
else ifeq ($(BOARD),unor4)
  LDSCRIPT := ld/ra4m1.ld
  UART_SRC := uart_ra4m1
  RAM_KB := 32
else
  $(error BOARD must be nano33ble or unor4)
endif
TFLM_GEN := $(TFLM)/gen/cortex_m_generic_cortex-m4+fp_release_with_logs_cmsis_nn_gcc
TFLM_LIB := $(TFLM_GEN)/lib/libtensorflow-microlite.a
DL := $(TFLM)/tensorflow/lite/micro/tools/make/downloads

CC := arm-none-eabi-gcc
CXX := arm-none-eabi-g++
SIZE := arm-none-eabi-size

# Must match how libtensorflow-microlite.a was built (cortex_m_generic, cortex-m4+fp, cmsis_nn).
ARCH := -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -mlittle-endian
DEFS := -DTF_LITE_MCU_DEBUG_LOG -DCMSIS_NN -DKERNELS_OPTIMIZED_FOR_SPEED -DNDEBUG -DTF_LITE_DISABLE_X86_NEON \
        -DARENA_SIZE='($(ARENA_KB)*1024)' -DBOARD_RAM_BYTES='($(RAM_KB)*1024)'
INC := -Isrc -I$(BUILD) -I$(TFLM) -I$(DL)/flatbuffers/include -I$(DL)/gemmlowp
COMMON := $(ARCH) $(DEFS) $(INC) -Os -ffunction-sections -fdata-sections -fno-unwind-tables \
          -fno-asynchronous-unwind-tables -funsigned-char -Wall -Wextra -Wno-unused-parameter
CFLAGS := $(COMMON) -std=c17
CXXFLAGS := $(COMMON) -std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics
LDFLAGS ?= $(ARCH) -L ld -T $(LDSCRIPT) $(EXTRA_LDFLAGS) -nostartfiles --specs=nano.specs --specs=nosys.specs \
           -Wl,--gc-sections -Wl,-Map=$(BUILD)/firmware.map -Wl,--print-memory-usage

OBJS := $(BUILD)/startup.o $(BUILD)/$(UART_SRC).o $(BUILD)/main.o $(BUILD)/model.o $(BUILD)/test_images.o

all: $(BUILD)/firmware.elf

$(BUILD):
	mkdir -p $@

# Step 3: model.tflite -> model.cc with xxd -i (renamed to g_model, const, alignas(16)).
$(BUILD)/model.cc: model/model.tflite make_model_cc.sh | $(BUILD)
	bash make_model_cc.sh $< $@

$(BUILD)/test_images.cc: model/test_images.cc model/model_ops.h | $(BUILD)
	cp model/test_images.cc model/test_images.h model/model_ops.h $(BUILD)/

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/%.o: src/%.cc $(BUILD)/test_images.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD)/%.o: $(BUILD)/%.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/firmware.elf: $(OBJS) $(TFLM_LIB) $(LDSCRIPT) ld/common_sections.ld
	$(CXX) $(LDFLAGS) $(OBJS) $(TFLM_LIB) -lm -o $@
	$(SIZE) $@

# Arena probe for tinyml/tflm_cost.py: no model linked in, the host loads one at 0x80000 (Nano 33 BLE only).
PROBE_BUILD := build/probe_$(BOARD)
PROBE_OBJS := $(PROBE_BUILD)/startup.o $(PROBE_BUILD)/$(UART_SRC).o $(PROBE_BUILD)/probe.o
PROBE_FLAGS := -Isrc/probe

probe: $(PROBE_BUILD)/probe.elf

$(PROBE_BUILD):
	mkdir -p $@
$(PROBE_BUILD)/%.o: src/%.c | $(PROBE_BUILD)
	$(CC) $(CFLAGS) -c $< -o $@
$(PROBE_BUILD)/%.o: src/%.cc src/probe/model_ops.h | $(PROBE_BUILD)
	$(CXX) $(CXXFLAGS) $(PROBE_FLAGS) -U ARENA_SIZE -DARENA_SIZE="(200*1024)" -c $< -o $@
$(PROBE_BUILD)/probe.elf: $(PROBE_OBJS) $(TFLM_LIB) $(LDSCRIPT) ld/common_sections.ld
	$(CXX) $(ARCH) -L ld -T $(LDSCRIPT) -nostartfiles --specs=nano.specs --specs=nosys.specs 	  -Wl,--gc-sections -Wl,--print-memory-usage $(PROBE_OBJS) $(TFLM_LIB) -lm -o $@
	$(SIZE) $@

clean:
	rm -rf build

.PHONY: all clean probe
