#include <math.h>
#include <stdio.h>
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

