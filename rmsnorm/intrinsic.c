#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
#include <riscv_vector.h>
#include <assert.h>
void RMSnorm(float* input, float* output, float* per_channle_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
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

#else
//TODO: Update the function definitions to reflect the new OPSNAME
void RMSnorm(float* input, float* output, float* per_channle_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            for (int h = 0; h < hidden_dim; h++) {
                sum += input[b * seq_len * hidden_dim + j * hidden_dim + h] * input[b * seq_len * hidden_dim + j * hidden_dim + h];
            }
            float rms = sqrt(sum / hidden_dim + rms_epsilon);
            for (int h = 0; h < hidden_dim; h++) {
                output[b * seq_len * hidden_dim + j * hidden_dim + h] = input[b * seq_len * hidden_dim + j * hidden_dim + h] / rms * per_channle_scale[h];
            }
        }
    }
}


#endif