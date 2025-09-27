#include <math.h>
#include <stdio.h>
//TODO: Update the function definitions to reflect the new OPSNAME
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

