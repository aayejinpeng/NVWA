#include <math.h>
#include <stdio.h>
#include <stdint.h>
//TODO: Update the function definitions to reflect the new OPSNAME
void w8a8matmul_smoothquant03(int8_t* A, int8_t* B,float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int)(A[m*K + k]) * (int)(B[n*K + k]);
            }
            output[m*N + n] = sum * A_scale[0] * B_scale[0];
        }
    }
}

