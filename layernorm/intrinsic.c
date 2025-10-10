#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
#include <riscv_vector.h>
#include <assert.h>
void LayerNorm(float* input, float* output, float* per_channle_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");

    assert(batch > 0 && seq_len > 0 && hidden_dim > 0);
    assert(hidden_dim % (16*4) == 0); // 512 = 32 * 16 m4
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            //第一步求和
            float sum = 0.0;
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float mean = sum / hidden_dim;
            //第二步求方差
            float variance = 0.0;
            vfloat32m4_t mean_vec = __riscv_vfmv_v_f_f32m4(mean, vl_0);
            vfloat32m4_t var_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t diff = __riscv_vfsub_vv_f32m4(vec, mean_vec, vl);
                var_vec = __riscv_vfadd_vv_f32m4(var_vec, __riscv_vfmul_vv_f32m4(diff, diff, vl), vl);
            }
            variance = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(var_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            variance /= hidden_dim;
            float invstd = 1.0 / sqrt(variance + rms_epsilon);

            //第三步归一化
            vfloat32m4_t invstd_vec = __riscv_vfmv_v_f_f32m4(invstd, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t normed = __riscv_vfmul_vv_f32m4(__riscv_vfsub_vv_f32m4(vec, mean_vec, vl), invstd_vec, vl);
                vfloat32m4_t scale = __riscv_vle32_v_f32m4(&per_channle_scale[h], vl);
                vfloat32m4_t out_vec = __riscv_vfmul_vv_f32m4(normed, scale, vl);
                __riscv_vse32_v_f32m4(&output[b * seq_len * hidden_dim + j * hidden_dim + h], out_vec, vl);
            }
        }
    }
}
#else

void LayerNorm(float* input, float* output, float* per_channle_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            for (int h = 0; h < hidden_dim; h++) {
                sum += input[b * seq_len * hidden_dim + j * hidden_dim + h];
            }
            float mean = sum / hidden_dim;
            float variance = 0.0;
            for (int h = 0; h < hidden_dim; h++) {
                variance += (input[b * seq_len * hidden_dim + j * hidden_dim + h] - mean) * 
                             (input[b * seq_len * hidden_dim + j * hidden_dim + h] - mean);
            }
            variance /= hidden_dim;
            float invstd = 1.0 / sqrt(variance + rms_epsilon);
            for (int h = 0; h < hidden_dim; h++) {
                output[b * seq_len * hidden_dim + j * hidden_dim + h] = 
                    (input[b * seq_len * hidden_dim + j * hidden_dim + h] - mean) * invstd * per_channle_scale[h];
            }
        }
    }
}


#endif