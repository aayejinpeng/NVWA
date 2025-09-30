#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
#include <riscv_vector.h>

inline vfloat32m4_t vec_exp(vfloat32m4_t x, size_t vl) {
    // 常数定义
    const float neg_ln2 = -0.69314718056f;
    const float inv_ln2 = 1.44269504089f;
    const float log2e = 1.44269504089f;


    // 计算 nr = round(x / ln2)
    vfloat32m4_t nf = __riscv_vfmul_vf_f32m4(x, inv_ln2, vl);
    vfloat32m4_t r = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl); // 2²³ + 2²²，用于取整
    vfloat32m4_t nr = __riscv_vfadd_vv_f32m4(nf, r, vl);
    nr = __riscv_vfsub_vv_f32m4(nr, r, vl);
    vint32m4_t nr_int = __riscv_vfcvt_x_f_v_i32m4(nr, vl);

    // 计算 2ⁿʳ
    vint32m4_t biased_exponent = __riscv_vadd_vx_i32m4(nr_int, 127, vl); // 加上偏置127
    biased_exponent = __riscv_vsll_vx_i32m4(biased_exponent, 23, vl); // 左移23位
    vfloat32m4_t a2 = __riscv_vreinterpret_v_i32m4_f32m4(biased_exponent); // 视为浮点数

    // 计算 b = x - nr * ln2
    // res = a2 * a3 + a1
    vfloat32m4_t b = __riscv_vfmacc_vf_f32m4(x, neg_ln2, nr, vl);

    // 计算 eᵇ
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
    p = __riscv_vfmacc_vv_f32m4(c4, c5, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c3,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c2,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c1,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c0,  p, b, vl);
    
    // 计算 2ⁿʳ*eᵇ
    p = __riscv_vfmul_vv_f32m4(a2, p, vl);
    return p;
}

void softmax(float* x, float* y, int M, int N) {
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");
    float max_row[M];
    for (int i = 0; i < M; i++) {
        float* row_x = &x[i * N];
        float* row_y = &y[i * N];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(N);
        // 第二步：计算指数和求和
        float sum_exp = 0.0f;
        vfloat32m4_t sumexp_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        // 向量化计算指数和求和
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_x[j], vl);
            
            // 计算 exp(x)
            vfloat32m4_t exp_vec = vec_exp(vec, vl);
            
            // 存储结果
            __riscv_vse32_v_f32m4(&row_y[j], exp_vec, vl);
            
            // 向量求和
            sumexp_vec = __riscv_vfadd_vv_f32m4(sumexp_vec, exp_vec, vl);
        }
        sum_exp = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumexp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
        
        // 第三步：归一化
        vfloat32m4_t inv_sum_exp_vec = __riscv_vfmv_v_f_f32m4(1.0f / sum_exp, vl_0);

        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_y[j], vl);
            vfloat32m4_t normalized = __riscv_vfmul_vv_f32m4(vec, inv_sum_exp_vec, vl);
            __riscv_vse32_v_f32m4(&row_y[j], normalized, vl);
        }
    }
  
}

#else
// fallback 普通 C 实现
void softmax(float* x, float* y, int M, int N) {
    /*
    x: 输入矩阵，大小为 M x N
    y: 输出矩阵，大小为 M x N
    */
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