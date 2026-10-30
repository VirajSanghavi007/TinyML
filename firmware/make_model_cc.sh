#!/usr/bin/env bash
# model.tflite -> model.cc via `xxd -i`, then: rename the array to g_model, make it const, alignas(16).
# Usage: make_model_cc.sh <model.tflite> <out model.cc>
set -euo pipefail
in="$1"
out="$2"
tmp="$(mktemp)"
# xxd names the array after the path; run it from the file's dir so the name is just model_tflite.
(cd "$(dirname "$in")" && xxd -i "$(basename "$in")") > "$tmp"
{
  echo '#include "model.h"'
  echo
  sed -e 's/^unsigned char model_tflite\[\]/alignas(16) const unsigned char g_model[]/' \
      -e 's/^unsigned int model_tflite_len/const unsigned int g_model_len/' "$tmp"
} > "$out"
rm -f "$tmp"
grep -q '^alignas(16) const unsigned char g_model\[\]' "$out" || { echo "rename failed" >&2; exit 1; }
grep -q '^const unsigned int g_model_len' "$out" || { echo "rename failed" >&2; exit 1; }
