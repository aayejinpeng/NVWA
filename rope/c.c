#include <math.h>
#include <stdio.h>
//TODO: Update the function definitions to reflect the new OPSNAME
void rope(float* x, float* y, int batch, int seq_len, int n_head, int head_dim, int basefreq)
{
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < n_head; j++) {
                for (int k = 0; k < head_dim; k+=2) {
                    // 计算每个位置和维度的旋转角度
                    int pos = i;
                    int dim = k;

                    float inv_freq = 1.0f / powf(basefreq, (float)dim / head_dim);
                    float angle = pos * inv_freq;
                    // printf("Position: %d, Dimension: %d, Angle: %f\n", pos, dim, angle);

                    // 计算旋转后的值
                    float sin_val = sinf(angle);
                    float cos_val = cosf(angle);

                    int idx = ((b * seq_len + i) * n_head + j) * head_dim + k;
                    float x1 = x[idx];
                    float x2 = x[idx + 1];
                    y[idx] = x1 * cos_val - x2 * sin_val;
                    y[idx + 1] = x1 * sin_val + x2 * cos_val;

                }
            }
        }
    }
}

