#include <math.h>
#include <stdio.h>
#include <math.h>
//TODO: Update the function definitions to reflect the new OPSNAME
void silu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
{
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < exhidden_dim; s++) {
            for (int h = 0; h < hidden_dim; h++) {
                int i = b * exhidden_dim * hidden_dim + s * hidden_dim + h;
                float x = input[i];
                output[i] = 1/(1 + expf(-x)) * x;  // SiLU激活函数
            }
        }
    }
}

