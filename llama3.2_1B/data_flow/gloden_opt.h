#include <stdio.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

void __gloden_f16_matmul(uint16_t* A, uint16_t* B, float* output, int M, int N,int K)
{
    
    _Float16* f16A = (_Float16*)(A);
    _Float16* f16B = (_Float16*)(B);
    float* C = (float*)(output);
    for(int i=0;i<M;i++)
    for(int j=0;j<N;j++)
    {
        float acc = 0.0f;
        for(int k=0;k<K;k++)
        {
            float a_val = (float)f16A[i*M+k];
            float b_val = (float)f16B[j*N+k];
            acc += a_val * b_val;
        }
        C[i*M+j] = acc;
    }
}

static inline vfloat32m4_t __gloden_vec_exp(vfloat32m4_t x, size_t vl) {
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

static inline vfloat32m4_t __gloden_vec_tanh(vfloat32m4_t x, size_t vl) {
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
        vfloat32m4_t exp_2x = __gloden_vec_exp(two_x, vl);
        vfloat32m4_t numerator = __riscv_vfsub_vf_f32m4(exp_2x, 1.0f, vl);
        vfloat32m4_t denominator = __riscv_vfadd_vf_f32m4(exp_2x, 1.0f, vl);

        vfloat32m4_t tanh_x = __riscv_vfdiv_vv_f32m4(numerator, denominator, vl);
        result = __riscv_vmerge_vvm_f32m4(result, tanh_x, mask_small, vl);
    }

    return result;
}


void __gloden_GeLu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
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
                vfloat32m4_t tanh_inner = __gloden_vec_tanh(inner, vl);
                vfloat32m4_t one_plus_tanh = __riscv_vfadd_vf_f32m4(tanh_inner, 1.0f, vl);
                vfloat32m4_t gelu = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(x, 0.5f, vl), one_plus_tanh, vl);
                __riscv_vse32_v_f32m4(&row_output[i], gelu, vl);
            }
        }
    }

}

void __gloden_Q_matmul_I8I8I32(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int)(A[m*K + k]) * (int)(B[n*K + k]);
            }
            ((int32_t *)output)[m*N + n] = sum;  // Updated to use scale[m]
        }
    }
}

void __gloden_pertoken_pertensor_scale(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    int32_t *output_i = (int32_t *)output;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = output_i[m*N + n];
            output[m*N + n] = sum * A_scale[m] * B_scale[0];  // Updated to use scale[m]
        }
    }
}

void __gloden_Q_matmul_I8I8I32_pertoken_pertensor(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    __gloden_Q_matmul_I8I8I32(A, B, A_scale, B_scale, output, M, N, K);
    __gloden_pertoken_pertensor_scale(A, B, A_scale, B_scale, output, M, N, K);
}

void __gloden_RMSnorm(float* input, float* output, float* per_channle_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    assert(batch > 0 && seq_len > 0 && hidden_dim > 0);
    assert(hidden_dim % (16*4) == 0); // 512 = 32 * 16 m4
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");
    
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            // 计算平方和
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t vec_2 = __riscv_vfmul_vv_f32m4(vec, vec, vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec_2, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float rms = 1.0 / sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            // 归一化并缩放
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t per_channle_scale_vec = __riscv_vle32_v_f32m4(&per_channle_scale[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, per_channle_scale_vec, vl);
                __riscv_vse32_v_f32m4(&output[b * seq_len * hidden_dim + j * hidden_dim + h], scaled_vec, vl);
            }

        }
    }

}

static inline vfloat32m4_t __gloden_vec_sin_small(vfloat32m4_t x, size_t vl) 
{
    // 计算 sin(x) 的泰勒展开，适用于小角度 x
    // sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9! - x¹¹/11!
    // sin(x) ≈ x * (c₁ + x² * (c₃ + x² * (c₅ + x² * (c₇ + x² * (c₉ + x² * c₁₁)))))
    vfloat32m4_t x2 = __riscv_vfmul_vv_f32m4(x, x, vl); // x²
    vfloat32m4_t c1  = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c3  = __riscv_vfmv_v_f_f32m4(-0.166666666667f, vl);
    vfloat32m4_t c5  = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c7  = __riscv_vfmv_v_f_f32m4(-0.0001984126984f, vl);
    vfloat32m4_t c9  = __riscv_vfmv_v_f_f32m4(0.000002755731922f, vl);
    vfloat32m4_t c11 = __riscv_vfmv_v_f_f32m4(-0.000000025052108f, vl);
    vfloat32m4_t result;
    // 计算并累加每一项
    // res = a1 + a2 * a3
    result = __riscv_vfmacc_vv_f32m4(c9,     c11, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c7,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c5,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c3,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c1,  result, x2, vl);
    result = __riscv_vfmul_vv_f32m4(result, x, vl); // 最后乘以 x
    return result;
}


static inline vfloat32m4_t __gloden_vec_sin(vfloat32m4_t x, size_t vl) 
{
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    vfloat32m4_t pi_vec = __riscv_vfmv_v_f_f32m4(PI, vl);
    vfloat32m4_t round = __riscv_vfdiv_vv_f32m4(new_rad, pi_vec, vl);
    vfloat32m4_t magic = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl); // 2²³ + 2²²，用于取整
    round = __riscv_vfadd_vv_f32m4(round, magic, vl);
    round = __riscv_vfsub_vv_f32m4(round, magic, vl);
    new_rad = __riscv_vfsub_vv_f32m4(new_rad, __riscv_vfmul_vv_f32m4(round, pi_vec, vl), vl);
    new_rad = __riscv_vfsub_vv_f32m4(new_rad, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    // 计算 sin(new_rad)
    vfloat32m4_t sin_result = __gloden_vec_sin_small(new_rad, vl);
    // 根据 round 的奇偶性调整符号
    vuint32m4_t round_int = __riscv_vfcvt_xu_f_v_u32m4(round, vl);
    vuint32m4_t round_odd_int = __riscv_vand_vx_u32m4(round_int, 1, vl);
    round_odd_int = __riscv_vsll_vx_u32m4(round_odd_int, 31, vl); // 将最低位移到符号位
    vfloat32m4_t sign = __riscv_vreinterpret_v_u32m4_f32m4(round_odd_int);
    sin_result = __riscv_vfsgnjn_vv_f32m4(sin_result, sign, vl);
    return sin_result;
}

static inline vfloat32m4_t __gloden_vec_cos(vfloat32m4_t x, size_t vl) 
{
    // 将 x 映射到 [-π, π]
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    return __gloden_vec_sin(new_rad, vl);
}

//TODO: Update the function definitions to reflect the new OPSNAME
void __gloden_rope(float* input, float* output, float* rope_theta, int pos, int batch, int n_head, int seq_len, int head_dim)
{
    // 获取最大向量长度
    const int half_dim = head_dim / 2;

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < n_head; i++) {
            for (int j = 0; j < seq_len; j++) {
                int pos_ = j + pos;

                // 计算当前token的起始位置
                int input_offset = b * n_head * seq_len * head_dim +
                                  i * seq_len * head_dim +
                                  j * head_dim;
                int output_offset = b * n_head * seq_len * head_dim +
                                   i * seq_len * head_dim +
                                   j * head_dim;
                size_t avl, vl;
                // 处理每两个元素（实部和虚部）
                for (int k = 0, avl = half_dim; avl > 0; k += vl, avl -= vl) {
                    vl = __riscv_vsetvl_e32m4(avl);

                    // 加载rope_theta参数
                    vfloat32m4_t theta_vec = __riscv_vle32_v_f32m4(rope_theta + k, vl);

                    // 计算角度：angle = pos_ * inv_freq
                    vfloat32m4_t angle_vec = __riscv_vfmul_vf_f32m4(theta_vec, pos_, vl);

                    // 计算sin和cos
                    vfloat32m4_t sin_vec, cos_vec;
                    sin_vec = __gloden_vec_sin(angle_vec, vl);
                    cos_vec = __gloden_vec_cos(angle_vec, vl);

                    // 加载输入数据（实部和虚部）
                    vfloat32m4_t real_in = __riscv_vlse32_v_f32m4(input + input_offset + 2 * k, 2*sizeof(float), vl);
                    vfloat32m4_t imag_in = __riscv_vlse32_v_f32m4(input + input_offset + 2 * k + 1, 2*sizeof(float), vl);

                    // 计算旋转：real_out = real*cos - imag*sin
                    // res = a1 * a2 - a3
                    vfloat32m4_t real_out = __riscv_vfmsub_vv_f32m4(real_in, cos_vec,
                                                                   __riscv_vfmul_vv_f32m4(imag_in, sin_vec, vl), vl);

                    // 计算旋转：imag_out = real*sin + imag*cos
                    // res = a2 * a3 + a1
                    vfloat32m4_t imag_out = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(real_in, sin_vec, vl),
                                                                   imag_in, cos_vec, vl);

                    // 存储结果
                    __riscv_vse32_v_f32m4(output + output_offset + k, real_out, vl);
                    __riscv_vse32_v_f32m4(output + output_offset + half_dim + k, imag_out, vl);
                }
            }
        }
    }
}

void __gloden_silu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
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
                vfloat32m4_t exp_neg_x = __gloden_vec_exp(__riscv_vfneg_v_f32m4(x, vl), vl);
                vfloat32m4_t silu = __riscv_vfdiv_vv_f32m4(x, __riscv_vfadd_vf_f32m4(exp_neg_x, 1.0f, vl), vl);
                __riscv_vse32_v_f32m4(&row_output[i], silu, vl);
            }
        }
    }
}

void __gloden_smoothquantO1_stage1_getscale(float* A, float* scale, int M,int K)
{
    assert(K%(64*4) == 0);
    assert(M%16 == 0);
    assert(M <= 1024);
    assert(K <= 32768);
    
    for (int i = 0; i < M; i++) {
        float* row_A = &A[i * K];

        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(K);
        vl = vl_0;
        vfloat32m4_t tmp = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        for (int j = 0, avl = K; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(&row_A[j], vl);
            vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
            tmp = __riscv_vfmax_vv_f32m4(tmp, vfabs, vl);
        }
        vfloat32m1_t tmp_m1_max = __riscv_vfmv_v_f_f32m1(0.0f, vl_0);
        tmp_m1_max = __riscv_vfredmax_vs_f32m4_f32m1(tmp, tmp_m1_max, vl_0);

        float token_max = __riscv_vfmv_f_s_f32m1_f32(tmp_m1_max);

        const float d = token_max / (127.0f);
        scale[i] = d;

    }
}

void __gloden_smoothquantO1_stage2_quant(float* A, int8_t* output,float* scale, int M,int K)
{
    for (int i = 0; i < M; i++) {
        float* row_A = &A[i * K];
        int8_t* output_row = &output[i * K];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(K);
        vl = vl_0;
        float id = 1.0f / scale[i];
        for (int j = 0, avl = K; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row_A[j], vl);
            vfloat32m4_t x0  = __riscv_vfmul_vf_f32m4(v_x, id, vl);
            vint16m2_t   vi  = __riscv_vfncvt_x_f_w_i16m2(x0, vl);//默认舍入模式为round to nearest, ties to even
            vint8m1_t    vs  = __riscv_vncvt_x_x_w_i8m1(vi, vl);
            __riscv_vse8_v_i8m1(&output_row[j], vs, vl);
        }
    }

}


void __gloden_smoothquantO1(float* A, int8_t* output,float* scale, int M,int K)
{
    __gloden_smoothquantO1_stage1_getscale(A, scale, M, K);
    __gloden_smoothquantO1_stage2_quant(A, output, scale, M, K);
}

void __gloden_softmax(float* x, float* y, void* bitmask_ptr, int M, int N) 
{
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");
    for (int i = 0; i < M; i++) {
        float* row_x = &x[i * N];
        float* row_y = &y[i * N];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(N);
        // 第一步：计算最大值
        float max_val = row_x[0];
        vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(max_val, vl_0);
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_x[j], vl);
            vbool8_t mask = __riscv_vlm_v_b8((uint8_t*)(bitmask_ptr + (i * N + j)/8), vl);
            // mask[i] ? op2[i] : op1[i]
            vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-INFINITY, vl), vec, mask, vl);
            max_vec = __riscv_vfmax_vv_f32m4(max_vec, vec, vl);
        }
        max_val = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(max_vec, __riscv_vfmv_v_f_f32m1(-INFINITY, vl_0), vl_0));
        // 第二步：计算指数和求和
        float sum_exp = 0.0f;
        vfloat32m4_t sumexp_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        // 向量化计算指数和求和
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_x[j], vl);
            vec = __riscv_vfsub_vf_f32m4(vec, max_val, vl); // 减去最大值

            // 应用掩码
            vbool8_t mask = __riscv_vlm_v_b8((uint8_t*)(bitmask_ptr + (i * N + j)/8), vl);
            // vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-INFINITY, vl), vec, mask, vl);
            vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-90, vl), vec, mask, vl);

            // 计算 exp(x)
            vfloat32m4_t exp_vec = __gloden_vec_exp(vec, vl);
            
            // 存储结果
            __riscv_vse32_v_f32m4(&row_y[j], exp_vec, vl);
            
            // 向量求和
            sumexp_vec = __riscv_vfadd_vv_f32m4(sumexp_vec, exp_vec, vl);
        }
        sum_exp = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumexp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
        
        // 第三步：归一化
        vfloat32m4_t inv_sum_exp_vec = __riscv_vfmv_v_f_f32m4(1.0f / sum_exp, vl_0);

        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_y[j], vl);
            vfloat32m4_t normalized = __riscv_vfmul_vv_f32m4(vec, inv_sum_exp_vec, vl);
            __riscv_vse32_v_f32m4(&row_y[j], normalized, vl);
        }
    }
  
}