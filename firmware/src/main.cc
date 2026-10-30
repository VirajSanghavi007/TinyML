// CIFAR-10 int8 TinyNet on Cortex-M4F boards (Arduino Nano 33 BLE / Uno R4 Minima).
// Adapted from tflite-micro's hello_world example: model include, op resolver, arena size, and
// input handling changed for this model.
#include <stdint.h>
#include <string.h>

#include "model.h"
#include "model_ops.h"
#include "test_images.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "uart.h"

extern "C" uint32_t StackHighWaterBytes(void);
extern "C" uint32_t StaticRamBytes(void);

#ifndef ARENA_SIZE
#define ARENA_SIZE (30 * 1024)
#endif

namespace {
// Op resolver: model_ops.h is generated from the model's op list (tinyml/tflm_cost.py), so it
// registers exactly the ops the model uses. A missing one fails AllocateTensors().
using OpResolver = ModelOpResolver;

alignas(16) uint8_t tensor_arena[ARENA_SIZE];

const char* const kClasses[10] = {"airplane", "automobile", "bird", "cat", "deer",
                                  "dog", "frog", "horse", "ship", "truck"};

int ArgMax(const int8_t* v, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (v[i] > v[best]) best = i;
  }
  return best;
}

// Per-allocation breakdown of the arena, then the interpreter is destroyed and the arena reused.
TfLiteStatus PrintArenaBreakdown(const tflite::Model* model, OpResolver& resolver) {
  tflite::RecordingMicroAllocator* allocator =
      tflite::RecordingMicroAllocator::Create(tensor_arena, ARENA_SIZE);
  tflite::RecordingMicroInterpreter interpreter(model, resolver, allocator);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  MicroPrintf("--- arena breakdown (RecordingMicroAllocator) ---");
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus Run() {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("schema mismatch: model %d vs runtime %d", (int)model->version(), TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }
  OpResolver resolver;
  TF_LITE_ENSURE_STATUS(RegisterModelOps(resolver));
  MicroPrintf("model: %u bytes in flash at %p", g_model_len, g_model);
  MicroPrintf("arena: %d bytes reserved", ARENA_SIZE);

  TF_LITE_ENSURE_STATUS(PrintArenaBreakdown(model, resolver));

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, ARENA_SIZE);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed (arena too small or op missing)");
    return kTfLiteError;
  }
  MicroPrintf("arena_used_bytes: %u", (unsigned)interpreter.arena_used_bytes());

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  MicroPrintf("input:  type=%d bytes=%u scale=%d/1000 zp=%d", input->type, (unsigned)input->bytes,
              (int)(input->params.scale * 1000), (int)input->params.zero_point);
  MicroPrintf("output: type=%d bytes=%u", output->type, (unsigned)output->bytes);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 || input->bytes != 32 * 32 * 3) {
    MicroPrintf("unexpected tensor types/shape");
    return kTfLiteError;
  }

  int correct = 0, host_match = 0;
  for (int i = 0; i < kNumTestImages; ++i) {
    // Input handling: images are already int8 NHWC (pixel - 128, input scale 1, zero point 0).
    // On a real camera pipeline this is the only per-pixel work: out[i] = (int8_t)(rgb[i] - 128).
    memcpy(input->data.int8, kTestImages[i], input->bytes);
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    const int8_t* out = output->data.int8;
    int pred = ArgMax(out, 10);
    bool exact = memcmp(out, kHostOutputs[i], 10) == 0;
    correct += pred == kTestLabels[i];
    host_match += exact;
    MicroPrintf("img %d: pred=%s (p=%d/256) label=%s %s host_bitexact=%s", i, kClasses[pred], out[pred] + 128,
                kClasses[kTestLabels[i]], pred == kTestLabels[i] ? "OK" : "MISS", exact ? "yes" : "no");
  }
  MicroPrintf("correct %d/%d, bit-exact vs host interpreter %d/%d", correct, kNumTestImages, host_match,
              kNumTestImages);
  MicroPrintf("arena_used_bytes: %u", (unsigned)interpreter.arena_used_bytes());
  MicroPrintf("RAM: static (.data+.bss incl. arena) %u B, stack high-water %u B, total %u of %u B",
              (unsigned)StaticRamBytes(), (unsigned)StackHighWaterBytes(),
              (unsigned)(StaticRamBytes() + StackHighWaterBytes()), (unsigned)BOARD_RAM_BYTES);
  MicroPrintf("DONE");
  return kTfLiteOk;
}
}  // namespace

int main() {
  UartInit();
  RegisterDebugLogCallback(UartWrite);
  if (Run() != kTfLiteOk) MicroPrintf("FAILED");
  for (;;) {
  }
}
