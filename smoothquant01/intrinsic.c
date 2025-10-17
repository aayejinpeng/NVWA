#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(__riscv)// RISC-V 架构
#include <assert.h>
#include <riscv_vector.h>
//TODO: Update the function definitions to reflect the new OPSNAME

void smoothquantO1_stage1_getscale(float* A, float* scale, int M,int K)
{
    assert(K%(64*4) == 0);
    assert(M%16 == 0);
    assert(M <= 1024);
    assert(K <= 32768);
    
    for (int i = 0; i < M; i++) {
        float* row_A = &A[i * K];

        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(K);
        vl = vl_0;
        vfloat32m4_t tmp = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        for (int j = 0, avl = K; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(&row_A[j], vl);
            vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
            tmp = __riscv_vfmax_vv_f32m4(tmp, vfabs, vl);
        }
        vfloat32m1_t tmp_m1_max = __riscv_vfmv_v_f_f32m1(0.0f, vl_0);
        tmp_m1_max = __riscv_vfredmax_vs_f32m4_f32m1(tmp, tmp_m1_max, vl_0);

        float token_max = __riscv_vfmv_f_s_f32m1_f32(tmp_m1_max);

        const float d = token_max / (127.0f);
        scale[i] = d;

    }
}

void smoothquantO1_stage2_quant(float* A, int8_t* output,float* scale, int M,int K)
{
    for (int i = 0; i < M; i++) {
        float* row_A = &A[i * K];
        int8_t* output_row = &output[i * K];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(K);
        vl = vl_0;
        float id = 1.0f / scale[i];
        for (int j = 0, avl = K; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row_A[j], vl);
            vfloat32m4_t x0  = __riscv_vfmul_vf_f32m4(v_x, id, vl);
            vint16m2_t   vi  = __riscv_vfncvt_x_f_w_i16m2(x0, vl);//默认舍入模式为round to nearest, ties to even
            vint8m1_t    vs  = __riscv_vncvt_x_x_w_i8m1(vi, vl);
            __riscv_vse8_v_i8m1(&output_row[j], vs, vl);
        }
    }

}


void smoothquantO1(float* A, int8_t* output,float* scale, int M,int K)
{
    smoothquantO1_stage1_getscale(A, scale, M, K);
    smoothquantO1_stage2_quant(A, output, scale, M, K);
}
#else
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



#endif