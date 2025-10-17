#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
//TODO: Update the function definitions to reflect the new OPSNAME
void smoothquantO1_stage1_getscale(float* A, float* scale, int M,int K)
{
    float token_max = 0.0;
    for(int i = 0; i < M; i++){
        token_max = 0;
        for(int j = 0; j < K; j++){
            if(fabs(A[i*K + j]) > token_max){
                token_max = fabs(A[i*K + j]);
            }
        }
        scale[i] = token_max / 127.0f;
    }
    
}

void smoothquantO1_stage2_quant(float* A, int8_t* output,float* scale, int M,int K)
{
    for (int i = 0; i < M*K; i++) {
        output[i] = roundf(A[i] / scale[i / K]);
    }
}

void smoothquantO1(float* A, int8_t* output,float* scale, int M,int K)
{
    smoothquantO1_stage1_getscale(A, scale, M, K);
    smoothquantO1_stage2_quant(A, output, scale, M, K);
}

