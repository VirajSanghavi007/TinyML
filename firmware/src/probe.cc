// Arena probe: measures what TFLM really needs for a .tflite on the target (Cortex-M4F, CMSIS-NN).
// The model is not linked in: the host (tinyml/tflm_cost.py) loads it into flash at PROBE_MODEL_ADDR,
// so one ELF measures any number of candidate models. Prints one line:
//   PROBE ok=1 arena=<arena_used_bytes> persistent=<persistent buffer bytes>
#include <stdint.h>

#include "model_ops.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "uart.h"

#ifndef PROBE_MODEL_ADDR
#define PROBE_MODEL_ADDR 0x00080000u
#endif
#ifndef ARENA_SIZE
#define ARENA_SIZE (200 * 1024)
#endif

namespace {
alignas(16) uint8_t tensor_arena[ARENA_SIZE];

void Probe() {
  const tflite::Model* model = tflite::GetModel(reinterpret_cast<const void*>(PROBE_MODEL_ADDR));
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("PROBE ok=0 reason=schema");
    return;
  }
  ModelOpResolver resolver;
  if (RegisterModelOps(resolver) != kTfLiteOk) {
    MicroPrintf("PROBE ok=0 reason=resolver");
    return;
  }
  size_t arena = 0;
  {
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, ARENA_SIZE);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
      MicroPrintf("PROBE ok=0 reason=allocate");
      return;
    }
    arena = interpreter.arena_used_bytes();
  }
  tflite::RecordingMicroAllocator* allocator = tflite::RecordingMicroAllocator::Create(tensor_arena, ARENA_SIZE);
  tflite::RecordingMicroInterpreter rec(model, resolver, allocator);
  if (rec.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("PROBE ok=0 reason=allocate_recording");
    return;
  }
  const auto pers = rec.GetMicroAllocator().GetRecordedAllocation(
      tflite::RecordedAllocationType::kPersistentBufferData);
  MicroPrintf("PROBE ok=1 arena=%u persistent=%u", (unsigned)arena, (unsigned)pers.used_bytes);
}
}  // namespace

int main() {
  UartInit();
  RegisterDebugLogCallback(UartWrite);
  Probe();
  MicroPrintf("DONE");
  for (;;) {
  }
}
