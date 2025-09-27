#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
//TODO: Update the function definitions to reflect the new OPSNAME
void w8a8matmul_smoothquant01(float* A, int8_t* B, float* B_scale, float* output, int M, int N,int K)
{

    //先对A进行abs_max的量化
    float* q_A = (float*)malloc(M*K*sizeof(float));
    float* token_max = (float*)malloc(M*sizeof(float));
    float* scale = (float*)malloc(M*sizeof(float));
    for(int i = 0; i < M; i++){
        token_max[i] = 0;
        for(int j = 0; j < K; j++){
            if(fabs(A[i*K + j]) > token_max[i]){
                token_max[i] = fabs(A[i*K + j]);
            }
        }
    }
    
    for(int i = 0; i < M; i++){
        if (token_max[i] < 1e-5) {
            token_max[i] = 1e-5; // 防止除以零
        }
        scale[i] = token_max[i] / 127.0f;
    }

    for (int i = 0; i < M*K; i++) {
        q_A[i] = roundf(A[i] / scale[i / K]);
    }

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int)(q_A[m*K + k]) * (int)(B[n*K + k]);
            }
            output[m*N + n] = sum * scale[m] * B_scale[0];  // Updated to use scale[m]
        }
    }
}

