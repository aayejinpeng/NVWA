#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
# include <riscv_vector.h>
//TODO: Update the function definitions to reflect the new OPSNAME
inline vfloat32m4_t vec_exp(vfloat32m4_t x, size_t vl) {
    // x = ln2 * a + b, 其中 b ∈ [0, ±ln2]
    // eˣ = 2ᵃ * eᵇ
    // 常数定义
    const float NEG_LN2 = -0.69314718056f;
    const float INV_LN2 = 1.44269504089f;
    const int32_t MAX_A = 127; // 防止指数溢出
    const int32_t MIN_A = -126; // 防止指数下溢

    // 计算 a = round(x / ln2)
    vfloat32m4_t af = __riscv_vfmul_vf_f32m4(x, INV_LN2, vl);
    vfloat32m4_t r = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl); // 2²³ + 2²²，用于取整
    vfloat32m4_t a = __riscv_vfadd_vv_f32m4(af, r, vl);
    a = __riscv_vfsub_vv_f32m4(a, r, vl);
    vint32m4_t a_int = __riscv_vfcvt_x_f_v_i32m4(a, vl);
    // 处理 a 的边界情况
    vbool8_t mask_max = __riscv_vmsgt_vx_i32m4_b8(a_int, MAX_A, vl);    // res[i] = op1[i] > op2
    vbool8_t mask_min = __riscv_vmslt_vx_i32m4_b8(a_int, MIN_A, vl);    // res[i] = op1[i] < op2

    // 计算 2ᵃ
    vint32m4_t biased_exponent = __riscv_vadd_vx_i32m4(a_int, 127, vl); // 加上偏置127
    biased_exponent = __riscv_vsll_vx_i32m4(biased_exponent, 23, vl); // 左移23位
    vfloat32m4_t a2 = __riscv_vreinterpret_v_i32m4_f32m4(biased_exponent); // 视为浮点数

    // 计算 b = x - a * ln2
    // res = a2 * a3 + a1
    vfloat32m4_t b = __riscv_vfmacc_vf_f32m4(x, NEG_LN2, a, vl);

    // 计算 eᵇ
    // 泰勒展开多项式系数
    vfloat32m4_t c0 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.166666666667f, vl);
    vfloat32m4_t c4 = __riscv_vfmv_v_f_f32m4(0.041666666667f, vl);
    vfloat32m4_t c5 = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c6 = __riscv_vfmv_v_f_f32m4(0.001388888889f, vl);
    vfloat32m4_t  p;
    // exp(b) ≈ c₀ + b * (c₁ + b * (c₂ + b * (c₃ + b * (c₄ + b * (c₅ + b * c₆)))))
    // res = a1 + a2 * a3
    p = __riscv_vfmacc_vv_f32m4(c5, c6, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c4,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c3,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c2,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c1,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c0,  p, b, vl);
    
    // 计算 2ᵃ * eᵇ
    p = __riscv_vfmul_vv_f32m4(a2, p, vl);

    // 处理边界情况
    // mask[i] ? op2[i] : op1[i]
    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(INFINITY, vl), mask_max, vl);
    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(0.0f, vl), mask_min, vl);
    return p;
}


void silu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
{
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < exhidden_dim; s++) {
            float* row_input  = &input[b * exhidden_dim * hidden_dim + s * hidden_dim];
            float* row_output = &output[b * exhidden_dim * hidden_dim + s * hidden_dim];
            size_t avl, vl;
            for (int i = 0, avl = hidden_dim; avl > 0; i += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t x = __riscv_vle32_v_f32m4(&row_input[i], vl);
                vfloat32m4_t exp_neg_x = vec_exp(__riscv_vfneg_v_f32m4(x, vl), vl);
                vfloat32m4_t silu = __riscv_vfdiv_vv_f32m4(x, __riscv_vfadd_vf_f32m4(exp_neg_x, 1.0f, vl), vl);
                __riscv_vse32_v_f32m4(&row_output[i], silu, vl);
            }
        }
    }
}
#else
//TODO: Update the function definitions to reflect the new OPSNAME
#include <immintrin.h>
#include <math.h>

// Sigmoid 9 次多项式
static inline __m256 _mm256_sigmoid_ps(__m256 x) {
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 x1 = _mm256_mul_ps(_mm256_set1_ps(0.25f), x);        // x/4
    __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
    __m256 x5 = _mm256_mul_ps(x3, _mm256_mul_ps(x, x));
    __m256 x7 = _mm256_mul_ps(x5, _mm256_mul_ps(x, x));
    __m256 x9 = _mm256_mul_ps(x7, _mm256_mul_ps(x, x));

    __m256 term3 = _mm256_mul_ps(_mm256_set1_ps(-1.0f/48.0f), x3);
    __m256 term5 = _mm256_mul_ps(_mm256_set1_ps(1.0f/480.0f), x5);
    __m256 term7 = _mm256_mul_ps(_mm256_set1_ps(-17.0f/80640.0f), x7);
    __m256 term9 = _mm256_mul_ps(_mm256_set1_ps(31.0f/1451520.0f), x9);

    __m256 res = _mm256_add_ps(half, x1);
    res = _mm256_add_ps(res, term3);
    res = _mm256_add_ps(res, term5);
    res = _mm256_add_ps(res, term7);
    res = _mm256_add_ps(res, term9);
    return res;
}

// SIMD SiLU TODO:需要分段函数！分段的话，就很难silu了，但是！可以在硬件里silu！也就是求exp！
void silu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim) {
    int total = batch * exhidden_dim * hidden_dim;
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 s = _mm256_sigmoid_ps(x);
        __m256 y = _mm256_mul_ps(x, s);
        _mm256_storeu_ps(&output[i], y);
    }
    for (; i < total; i++) {
        float x = input[i];
        float s = 0.5f + 0.25f*x - 1.0f/48.0f*x*x*x + 1.0f/480.0f*x*x*x*x*x
                  - 17.0f/80640.0f*x*x*x*x*x*x*x + 31.0f/1451520.0f*x*x*x*x*x*x*x*x*x;
        output[i] = x * s;
    }
}



#endif