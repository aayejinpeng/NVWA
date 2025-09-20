#include <math.h>
#include <stdio.h>
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

                    int out_idx = b * n_head * seq_len * head_dim + i * seq_len * head_dim + j * head_dim + k/2;// 实部
                    int out_idx_imag = b * n_head * seq_len * head_dim + i * seq_len * head_dim + j * head_dim + k/2 + head_dim/2;// 虚部

                    output[out_idx] = real * cos_val - imag * sin_val;     
                    output[out_idx_imag] = real * sin_val + imag * cos_val; 

                }
            }
        }
    }
}

