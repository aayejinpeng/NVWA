#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
#include <riscv_vector.h>

static inline vfloat32m4_t vec_sin_small(vfloat32m4_t x, size_t vl) 
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


static inline vfloat32m4_t vec_sin(vfloat32m4_t x, size_t vl) 
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
    vfloat32m4_t sin_result = vec_sin_small(new_rad, vl);
    // 根据 round 的奇偶性调整符号
    vuint32m4_t round_int = __riscv_vfcvt_xu_f_v_u32m4(round, vl);
    vuint32m4_t round_odd_int = __riscv_vand_vx_u32m4(round_int, 1, vl);
    round_odd_int = __riscv_vsll_vx_u32m4(round_odd_int, 31, vl); // 将最低位移到符号位
    vfloat32m4_t sign = __riscv_vreinterpret_v_u32m4_f32m4(round_odd_int);
    sin_result = __riscv_vfsgnjn_vv_f32m4(sin_result, sign, vl);
    return sin_result;
}

static inline vfloat32m4_t vec_cos(vfloat32m4_t x, size_t vl) 
{
    // 将 x 映射到 [-π, π]
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    return vec_sin(new_rad, vl);
}

//TODO: Update the function definitions to reflect the new OPSNAME
void rope(float* input, float* output, float* rope_theta, int pos, int batch, int n_head, int seq_len, int head_dim)
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
                    sin_vec = vec_sin(angle_vec, vl);
                    cos_vec = vec_cos(angle_vec, vl);

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
#else
//TODO: Update the function definitions to reflect the new OPSNAME
void rope(float* input, float* output, float* rope_theta, int pos, int batch, int n_head, int seq_len, int head_dim)
{
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < n_head; i++) {
            for (int j = 0; j < seq_len; j++) {
                for (int k = 0; k < head_dim; k+=2) {
                    // 计算每个位置和维度的旋转角度
                    int pos_ = j + pos;
                    int dim_ = k / 2;

                    float inv_freq = rope_theta[dim_]; // 使用 rope_theta 计算 inv_freq
                    float angle = pos_ * inv_freq;
                    // printf("Position: %d, Dimension: %d, Angle: %f\n", pos_, dim_, angle);

                    // 计算旋转后的值
                    float sin_val = sinf(angle);
                    float cos_val = cosf(angle);

                    int idx = b * n_head * seq_len * head_dim + i * seq_len * head_dim + j * head_dim + k;
                    float real = input[idx];
                    float imag = input[idx + 1];

                    int out_idx = b * n_head * seq_len * head_dim + i * seq_len * head_dim + j * head_dim + k/2;
                    int out_idx_imag = b * n_head * seq_len * head_dim + i * seq_len * head_dim + j * head_dim + k/2 + head_dim/2;

                    output[out_idx] = real * cos_val - imag * sin_val;     // 实部
                    output[out_idx_imag] = real * sin_val + imag * cos_val; // 虚部

                }
            }
        }
    }
}


#endif