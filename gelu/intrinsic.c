#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
#include <riscv_vector.h>

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

inline vfloat32m4_t vec_tanh(vfloat32m4_t x, size_t vl) {
    const float THRESHOLD = 12.0f; // 阈值
    // 计算 |x|
    vfloat32m4_t abs_x = __riscv_vfabs_v_f32m4(x, vl);
    // 初始化结果向量
    vfloat32m4_t result = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    // 对于 |x| > THRESHOLD，tanh(x) ≈ sign(x)
    vbool8_t mask_large = __riscv_vmfgt_vf_f32m4_b8(abs_x, THRESHOLD, vl);
    // 用b的sign给a赋值为±1
    vfloat32m4_t sign_x = __riscv_vfsgnj_vv_f32m4(__riscv_vfmv_v_f_f32m4(1.0f, vl), x, vl);
    // mask[i] ? op2[i] : op1[i]
    result = __riscv_vmerge_vvm_f32m4(result, sign_x, mask_large, vl);

    // 对于 |x| <= THRESHOLD，使用(exp(2x) - 1) / (exp(2x) + 1)计算tanh(x)
    vbool8_t mask_small = __riscv_vmfle_vf_f32m4_b8(abs_x, THRESHOLD, vl);
    if (__riscv_vfirst_m_b8(mask_small, vl) >= 0) {
        // mask[i] ? op2[i] : op1[i]
        vfloat32m4_t x_small = __riscv_vmerge_vvm_f32m4( __riscv_vfmv_v_f_f32m4(0.0f, vl), x, mask_small, vl);
        vfloat32m4_t two_x = __riscv_vfmul_vf_f32m4(x_small, 2.0f, vl);
        vfloat32m4_t exp_2x = vec_exp(two_x, vl);
        vfloat32m4_t numerator = __riscv_vfsub_vf_f32m4(exp_2x, 1.0f, vl);
        vfloat32m4_t denominator = __riscv_vfadd_vf_f32m4(exp_2x, 1.0f, vl);

        vfloat32m4_t tanh_x = __riscv_vfdiv_vv_f32m4(numerator, denominator, vl);
        result = __riscv_vmerge_vvm_f32m4(result, tanh_x, mask_small, vl);
    }

    return result;
}

// inline vfloat32m4_t vec_tanh(vfloat32m4_t x, size_t vl) {

//     vfloat32m4_t two_x = __riscv_vfmul_vf_f32m4(x, 2.0f, vl);
//     vfloat32m4_t exp_2x = vec_exp(two_x, vl);
//     vfloat32m4_t numerator = __riscv_vfsub_vf_f32m4(exp_2x, 1.0f, vl);
//     vfloat32m4_t denominator = __riscv_vfadd_vf_f32m4(exp_2x, 1.0f, vl);
//     vfloat32m4_t tanh_x = __riscv_vfdiv_vv_f32m4(numerator, denominator, vl);
//     return tanh_x;
// }


void GeLu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
{
    static const float GELU_COEF_A     = 0.044715f;
    static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
    size_t avl, vl;
    size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < exhidden_dim; s++) {
            float* row_input  = &input[b * exhidden_dim * hidden_dim + s * hidden_dim];
            float* row_output = &output[b * exhidden_dim * hidden_dim + s * hidden_dim];
            // 向量化计算 GELU
            // 0.5 * x * (1 + tanh[ √(2/π) * (x + 0.044715 * x³)])
            for (int i = 0, avl = hidden_dim; avl > 0; i += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t x = __riscv_vle32_v_f32m4(&row_input[i], vl);
                vfloat32m4_t x_cube = __riscv_vfmul_vv_f32m4(x, __riscv_vfmul_vv_f32m4(x, x, vl), vl);
                vfloat32m4_t inner = __riscv_vfmacc_vf_f32m4(x, GELU_COEF_A, x_cube, vl);
                inner = __riscv_vfmul_vf_f32m4(inner, SQRT_2_OVER_PI, vl);
                vfloat32m4_t tanh_inner = vec_tanh(inner, vl);
                vfloat32m4_t one_plus_tanh = __riscv_vfadd_vf_f32m4(tanh_inner, 1.0f, vl);
                vfloat32m4_t gelu = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(x, 0.5f, vl), one_plus_tanh, vl);
                __riscv_vse32_v_f32m4(&row_output[i], gelu, vl);
            }
        }
    }

}
#else
//TODO: Update the function definitions to reflect the new OPSNAME
void GeLu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
{
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < exhidden_dim; s++) {
            for (int h = 0; h < hidden_dim; h++) {
                float x = input[b * exhidden_dim * hidden_dim + s * hidden_dim + h];
                output[b * exhidden_dim * hidden_dim + s * hidden_dim + h] = 0.5 * x * (1.0 + tanh(sqrtf(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
            }
        }
    }
}




#endif