#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
//TODO: Update the function definitions to reflect the new OPSNAME
void Q_matmul_I8I8I32(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int)(A[m*K + k]) * (int)(B[n*K + k]);
            }
            ((int32_t *)output)[m*N + n] = sum;  // Updated to use scale[m]
        }
    }
}

void pertoken_pertensor_scale(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    int32_t *output_i = (int32_t *)output;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = output_i[m*N + n];
            output[m*N + n] = sum * A_scale[m] * B_scale[0];  // Updated to use scale[m]
        }
    }
}

void Q_matmul_I8I8I32_pertoken_pertensor(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    Q_matmul_I8I8I32(A, B, A_scale, B_scale, output, M, N, K);
    pertoken_pertensor_scale(A, B, A_scale, B_scale, output, M, N, K);
}

