#include <math.h>
#include <stdio.h>
#include <stdint.h>
#if defined(__riscv)// RISC-V 架构
//TODO: Update the function definitions to reflect the new OPSNAME
void w8a8matmul_smoothquant03(int8_t* A, int8_t* B,float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    printf("RISC-V specific implementation WORK IN PROGRESS\n");
}
#else
//TODO: Update the function definitions to reflect the new OPSNAME
void w8a8matmul_smoothquant03(int8_t* A, int8_t* B,float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += (A[m*K + k]) * (B[n*K + k]);
            }
            output[m*N + n] = sum * A_scale[0] * B_scale[0];
        }
    }
}



#endif