// Usage:
//     #include "bf16.h"
//
//     Set env_name/binding.c obs to PrecisionTensor
//
//     bf16* observations;
//     observations[0] = f32_to_bf16(some_float);                // scalar
//
//     // Convert eight floats in one call:
//     store_f32x8_as_bf16(&observations[i], values);             // 8 values
//
//     // Reverse if you ever need to read back as float:
//     float f = bf16_to_f32(observations[0]);
//
#include <stdint.h>
#include <string.h>

typedef uint16_t bf16;

static inline bf16 f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_to_f32(bf16 b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline void store_f32x8_as_bf16(bf16* dst, const float* src) {
    for (int i = 0; i < 8; i++) {
        dst[i] = f32_to_bf16(src[i]);
    }
}
