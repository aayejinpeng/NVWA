#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(__riscv)// RISC-V 架构
#include <assert.h>
#include <riscv_vector.h>


void f16_matmul(uint16_t* A, uint16_t* B, float* output, int M, int N,int K)
{
    
    _Float16* f16A = (_Float16*)(A);
    _Float16* f16B = (_Float16*)(B);
    float* C = (float*)(output);
    for(int i=0;i<M;i++)
    for(int j=0;j<N;j++)
    {
        float acc = 0.0f;
        for(int k=0;k<K;k++)
        {
            float a_val = (float)f16A[i*M+k];
            float b_val = (float)f16B[j*N+k];
            acc += a_val * b_val;
        }
        C[i*M+j] = acc;
    }
}

#else
//TODO: Update the function definitions to reflect the new OPSNAME
float fp16_to_fp32(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t f_sgn = ((uint32_t)h & 0x8000u) << 16;
    uint32_t f_exp, f_sig;

    if (h_exp == 0x0000u) {
        // subnormal or zero
        if (h_sig == 0)
            return *(float*)&f_sgn;
        else {
            // Normalize the subnormal number
            h_sig <<= 1;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = ((127 - 15 - h_exp) << 23);
            f_sig = ((uint32_t)(h_sig & 0x03FFu)) << 13;
            uint32_t f = f_sgn | f_exp | f_sig;
            return *(float*)&f;
        }
    } else if (h_exp == 0x7C00u) {
        // inf or NaN
        f_exp = 0xFFu << 23;
        f_sig = ((uint32_t)h_sig) << 13;
        uint32_t f = f_sgn | f_exp | f_sig;
        return *(float*)&f;
    } else {
        // normalized number
        f_exp = ((uint32_t)(h_exp >> 10) - 15 + 127) << 23;
        f_sig = ((uint32_t)h_sig) << 13;
        uint32_t f = f_sgn | f_exp | f_sig;
        return *(float*)&f;
    }
}

void f16_matmul(uint16_t* A, uint16_t* B, float* output, int M, int N,int K)
{
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += fp16_to_fp32(A[i * K + k]) * fp16_to_fp32(B[j*K + k]);
            }
            output[i * N + j] = sum;
        }
    }
}





#endif