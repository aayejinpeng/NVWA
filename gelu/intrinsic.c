#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
//TODO: Update the function definitions to reflect the new OPSNAME
void GeLu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
{
    printf("RISC-V specific implementation WORK IN PROGRESS\n");
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