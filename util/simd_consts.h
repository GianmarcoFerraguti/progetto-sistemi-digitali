#ifndef SIMD_CONSTS
#define SIMD_CONSTS
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif // _WIN32s
#define AVX_DATA_LANE 16
#endif