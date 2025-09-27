#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
//TODO: Update the function definitions to reflect the new OPSNAME
void w8a8matmul_smoothquant02(float* A, int8_t* B, float* B_scale, float* output, int M, int N,int K)
{

    //先对A进行abs_max的量化
    float* q_A = (float*)malloc(M*K*sizeof(float));
    float max = 0;
    for(int i = 0; i < M*K; i++){
        if(fabs(A[i]) > max){
            max = fabs(A[i]);
        }
    }
    if (max < 1e-5) {
        max = 1e-5; // 防止除以零
    }
    float scale = max / 127.0f;

    for (int i = 0; i < M*K; i++) {
        q_A[i] = roundf(A[i] / scale);
        // printf("%f ", q_A[i]);
        // if(i % K == K-1){
        //     printf("\n");
        // }
    }

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int)(q_A[m*K + k]) * (int)(B[n*K + k]);
            }
            output[m*N + n] = sum * scale * B_scale[0];
        }
    }
}

