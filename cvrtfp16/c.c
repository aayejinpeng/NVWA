#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// 将 float 转换为半精度 (FP16)
static inline uint16_t float_to_fp16_one(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000u;      // sign位到 half 的位置
    int32_t exp  = (int32_t)((x >> 23) & 0xFFu);
    uint32_t mant = x & 0x007FFFFFu;

    if (exp == 255) {
        // Inf 或 NaN
        if (mant == 0) {
            // Infinity
            return (uint16_t)(sign | 0x7C00u);
        } else {
            // NaN -> 转成 qNaN (保留高位mantissa)
            uint16_t nan_mant = (uint16_t)(mant >> 13);
            // Ensure at least one mantissa bit set for a NaN
            if (nan_mant == 0) nan_mant = 1;
            return (uint16_t)(sign | 0x7C00u | nan_mant);
        }
    }

    // 计算目标 exponent (未偏置)
    int32_t new_exp = exp - 127 + 15;

    if (new_exp >= 31) {
        // 溢出 -> 转为 Infinity (也可以选择 saturate to max finite)
        return (uint16_t)(sign | 0x7C00u);
    } else if (new_exp <= 0) {
        // 可能是 subnormal 或者太小变为 0
        if (new_exp < -10) {
            // 绝对太小，四舍五入为 0
            return (uint16_t)sign;
        }
        // 转为 subnormal：先把隐含的1加回到 mantissa (对于归一化数)
        uint32_t mant_with_hidden = mant | 0x00800000u; // 24-bit (1.mant)
        // 需要右移的位数：1 - new_exp （因为 new_exp ≤ 0）
        int shift = 1 - new_exp; // 在 [1..11]
        // 将 mantissa 右移 (带舍入位)
        // 我们要得到最终 half 的 10-bit mantissa，所以先右移 (13 + shift) 位，保留低位作舍入判定
        uint32_t shifted = mant_with_hidden >> (13 + shift); // tentative half mantissa
        uint32_t remainder = mant_with_hidden & ((1u << (13 + shift)) - 1u);

        // round-to-nearest-even: compare remainder with half
        uint32_t half = 1u << (12 + shift); // the halfway point
        if (remainder > half || (remainder == half && (shifted & 1u))) {
            shifted += 1u;
        }

        return (uint16_t)(sign | (uint16_t)shifted);
    } else {
        // 归一化数，普通情况
        uint16_t half_exp = (uint16_t)(new_exp & 0x1Fu);
        // 取上位 10 位 mantissa (从 23 位中 >>13)
        uint32_t half_mant = mant >> 13;
        // 剩下的低13位用于舍入判定
        uint32_t rem = mant & 0x1FFFu;
        uint32_t half_point = 0x1000u; // 1 << 12 (halfway)

        // round-to-nearest-even
        if (rem > half_point || (rem == half_point && (half_mant & 1u))) {
            half_mant += 1u;
            if (half_mant == 0x400u) {
                // 进位导致 mantissa 溢出 -> 指数加 1
                half_mant = 0;
                half_exp += 1u;
                if (half_exp >= 31u) {
                    // 溢出到 inf
                    return (uint16_t)(sign | 0x7C00u);
                }
            }
        }

        return (uint16_t)(sign | (half_exp << 10) | (half_mant & 0x3FFu));
    }
}
//TODO: Update the function definitions to reflect the new OPSNAME
void cvrtfp16(float* input, void* output, int m,int n)
{
    for(int i = 0 ; i < m; i++)
        for(int j = 0; j < n; j++)
            ((uint16_t*)output)[i*n + j] = float_to_fp16_one(input[i*n + j]);
}
