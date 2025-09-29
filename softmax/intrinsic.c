#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
#include <riscv_vector.h>

inline vfloat32m4_t softmax_exp(vfloat32m4_t x, size_t vl) {
    // 泰勒展开多项式系数
    vfloat32m4_t c0 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.166666666667f, vl);
    vfloat32m4_t c4 = __riscv_vfmv_v_f_f32m4(0.041666666667f, vl);
    vfloat32m4_t c5 = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t  p;
    // exp(b) ≈ c₀ + b * (c₁ + b * (c₂ + b * (c₃ + b * (c₄ + b * c₅))))
    // res = a1 + a2 * a3
    p = __riscv_vfmacc_vv_f32m4(c4, c5, x, vl);
    p = __riscv_vfmacc_vv_f32m4(c3,  p, x, vl);
    p = __riscv_vfmacc_vv_f32m4(c2,  p, x, vl);
    p = __riscv_vfmacc_vv_f32m4(c1,  p, x, vl);
    p = __riscv_vfmacc_vv_f32m4(c0,  p, x, vl);
    
    return p;
}

void softmax(float* x, float* y, int M, int N) {
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");

    // 获取vl
    size_t vl = __riscv_vsetvlmax_e32m4();
    
    for (int i = 0; i < M; i++) {
        float* row_x = &x[i * N];
        float* row_y = &y[i * N];
        
        // 第一步：计算最大值
        float max_val = row_x[0];
        int j = 1;
        
        // 向量化求最大值
        for (; j + vl <= N; j += vl) {
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_x[j], vl);
            vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(max_val, vl);
            
            vbool8_t mask = __riscv_vmfgt_vv_f32m4_b8(vec, max_vec, vl);
            max_val = __riscv_vfmv_f_s_f32m4_f32(__riscv_vfmerge_vfm_f32m4(max_vec, row_x[j], mask, vl));
        }
        
        // 处理剩余标量元素
        for (; j < N; j++) {
            if (row_x[j] > max_val) max_val = row_x[j];
        }
        
        // 第二步：计算指数和求和
        float sum = 0.0f;
        j = 0;
        
        // 向量化计算指数和求和
        for (; j + vl <= N; j += vl) {
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_x[j], vl);
            vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(max_val, vl);
            
            // 计算 x - max_val
            vfloat32m4_t shifted = __riscv_vfsub_vv_f32m4(vec, max_vec, vl);
            
            // 计算 exp(x - max_val)
            vfloat32m4_t exp_vec;

            // 使用自定义的向量化exp函数
            exp_vec = softmax_exp(shifted, vl);
            
            // 存储结果并累加求和
            __riscv_vse32_v_f32m4(&row_y[j], exp_vec, vl);
            
            // 向量求和
            sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(exp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl));
        }
        
        // 处理剩余标量元素
        for (; j < N; j++) {
            row_y[j] = expf(row_x[j] - max_val);
            sum += row_y[j];
        }
        
        // 第三步：归一化
        j = 0;
        vfloat32m4_t inv_sum_vec = __riscv_vfmv_v_f_f32m4(1.0f / sum, vl);
        
        // 向量化归一化
        for (; j + vl <= N; j += vl) {
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_y[j], vl);
            vfloat32m4_t normalized = __riscv_vfmul_vv_f32m4(vec, inv_sum_vec, vl);
            __riscv_vse32_v_f32m4(&row_y[j], normalized, vl);
        }
        
        // 处理剩余标量元素
        for (; j < N; j++) {
            row_y[j] /= sum;
        }
    }
  
}

#else
// fallback 普通 C 实现
void softmax(float* x, float* y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float max_val = x[i * N];
        for (int j = 1; j < N; j++) {
            if (x[i*N + j] > max_val) max_val = x[i*N + j];
        }

        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            y[i*N + j] = expf(x[i*N + j] - max_val);
            sum += y[i*N + j];
        }
        for (int j = 0; j < N; j++) {
            y[i*N + j] /= sum;
        }
    }
}
#endif