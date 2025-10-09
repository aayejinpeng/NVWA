#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(__riscv)// RISC-V 架构
#include <assert.h>
#include <riscv_vector.h>
//TODO: Update the function definitions to reflect the new OPSNAME
float w8a8matmul_smoothquant01_scalebuf[1024] __attribute__((aligned(32)));
int8_t w8a8matmul_smoothquant01_qA[32768*1024] __attribute__((aligned(32)));
float tmp_f[128] __attribute__((aligned(32)));
void smoothquant01(float* A, int8_t* B, float* B_scale, float* output, int M, int N,int K)
{

    assert(K%(64*4) == 0);
    assert(N%16 == 0);
    assert(M%16 == 0);
    assert(M <= 1024);
    assert(N <= 1024);
    assert(K <= 32768);
    
    //先对A进行abs_max的量化
    //量化激活
    for (int i = 0; i < M; i++) {
        float* row_A = &A[i * K];
        int8_t* q_row_A = &w8a8matmul_smoothquant01_qA[i * K];

        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(N);
        vl = vl_0;
        // 第一步，求per_token的最大值
        vfloat32m4_t tmp = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        for (int j = 0, avl = K; avl > 0; j += vl, avl -= vl) {
                // printf("j=%d, avl=%d\n", j, avl);
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(&row_A[j], vl);
                // printf("v_x\n");
                // //输出v_x
                // __riscv_vse32_v_f32m4(&tmp_f, v_x, vl);
                // for(int ii = 0; ii < vl; ii++){
                //     printf("%f ", tmp_f[ii]);
                // }
                // printf("\n");
            vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
                // //输出vfabs
                // printf("vfabs\n");
                // __riscv_vse32_v_f32m4(&tmp_f, vfabs, vl);
                // for(int ii = 0; ii < vl; ii++){
                //     printf("%f ", tmp_f[ii]);
                // }
                // printf("\n");
            tmp = __riscv_vfmax_vv_f32m4(tmp, vfabs, vl);
                // //输出tmp
                // printf("tmp\n");
                // __riscv_vse32_v_f32m4(&tmp_f, tmp, vl);
                // for(int ii = 0; ii < vl; ii++){
                //     printf("%f ", tmp_f[ii]);
                // }
                // printf("\n");
        }
        vfloat32m1_t tmp_m1_max = __riscv_vfmv_v_f_f32m1(0.0f, vl_0);
            // //输出tmp_m1_max
            // printf("tmp_m1_max before\n"); 
            // __riscv_vse32_v_f32m1(&tmp_f, tmp_m1_max, vl_0);
            // for(int ii = 0; ii < vl_0; ii++){
            //     printf("%f ", tmp_f[ii]);
            // }
            // printf("\n");
        tmp_m1_max = __riscv_vfredmax_vs_f32m4_f32m1(tmp, tmp_m1_max, vl_0);
            // //输出tmp_m1_max
            // printf("tmp_m1_max after\n");
            // __riscv_vse32_v_f32m1(&tmp_f, tmp_m1_max, vl_0);
            // for(int ii = 0; ii < vl_0; ii++){
            //     printf("%f ", tmp_f[ii]);
            // }
            // printf("\n");

        float token_max = __riscv_vfmv_f_s_f32m1_f32(tmp_m1_max);

        // 第二步，计算scale
        const float d = token_max / (127.0f);
        const float id = d ? 1.0f / d : 0.0f;
        w8a8matmul_smoothquant01_scalebuf[i] = d;
            // printf("scale[%d]=%f\n", i, d);
        // 第三步，量化
        for (int j = 0, avl = K; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row_A[j], vl);
            vfloat32m4_t x0  = __riscv_vfmul_vf_f32m4(v_x, id, vl);
            vint16m2_t   vi  = __riscv_vfncvt_x_f_w_i16m2(x0, vl);//默认舍入模式为round to nearest, ties to even
            vint8m1_t    vs  = __riscv_vncvt_x_x_w_i8m1(vi, vl);
            __riscv_vse8_v_i8m1(&q_row_A[j], vs, vl);
        }
    }

}

void w8a8matmul_with_scale(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K)
{
    assert(K%(64*4) == 0);
    assert(N%16 == 0);
    assert(M%16 == 0);
    assert(M <= 1024);
    assert(N <= 1024);
    assert(K <= 32768);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int)(A[m*K + k]) * (int)(B[n*K + k]);
            }
            output[m*N + n] = sum * A_scale[m] * B_scale[0];
        }
    }
}


void w8a8matmul_smoothquant01(float* A, int8_t* B, float* B_scale, float* output, int M, int N,int K)
{
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");
    assert(K%(64*4) == 0);
    assert(N%16 == 0);
    assert(M%16 == 0);
    assert(M <= 1024);
    assert(N <= 1024);
    assert(K <= 32768);
    


    smoothquant01(A, B, B_scale, output, M, N, K);

        // //printf w8a8matmul_smoothquant01_scalebuf
        // for(int i = 0; i < M; i++){
        //     printf("%f ", w8a8matmul_smoothquant01_scalebuf[i]);
        // }
        // printf("\n");

        // // printf w8a8matmul_smoothquant01_qA
        // for(int i = 0; i < M; i++){
        //     for(int j = 0; j < K; j++){
        //         printf("%d ", w8a8matmul_smoothquant01_qA[i*K + j]);
        //     }
        //     printf("\n");
        // }
    w8a8matmul_with_scale((int8_t*)w8a8matmul_smoothquant01_qA, B, w8a8matmul_smoothquant01_scalebuf, B_scale, output, M, N, K);
}
#else
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



#endif