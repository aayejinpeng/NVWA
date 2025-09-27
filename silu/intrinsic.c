#include <math.h>
#include <stdio.h>

#if defined(__riscv)// RISC-V 架构
//TODO: Update the function definitions to reflect the new OPSNAME
void silu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
{
    printf("RISC-V specific implementation WORK IN PROGRESS\n");
}
#else
//TODO: Update the function definitions to reflect the new OPSNAME
#include <immintrin.h>
#include <math.h>

// Sigmoid 9 次多项式
static inline __m256 _mm256_sigmoid_ps(__m256 x) {
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 x1 = _mm256_mul_ps(_mm256_set1_ps(0.25f), x);        // x/4
    __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
    __m256 x5 = _mm256_mul_ps(x3, _mm256_mul_ps(x, x));
    __m256 x7 = _mm256_mul_ps(x5, _mm256_mul_ps(x, x));
    __m256 x9 = _mm256_mul_ps(x7, _mm256_mul_ps(x, x));

    __m256 term3 = _mm256_mul_ps(_mm256_set1_ps(-1.0f/48.0f), x3);
    __m256 term5 = _mm256_mul_ps(_mm256_set1_ps(1.0f/480.0f), x5);
    __m256 term7 = _mm256_mul_ps(_mm256_set1_ps(-17.0f/80640.0f), x7);
    __m256 term9 = _mm256_mul_ps(_mm256_set1_ps(31.0f/1451520.0f), x9);

    __m256 res = _mm256_add_ps(half, x1);
    res = _mm256_add_ps(res, term3);
    res = _mm256_add_ps(res, term5);
    res = _mm256_add_ps(res, term7);
    res = _mm256_add_ps(res, term9);
    return res;
}

// SIMD SiLU TODO:需要分段函数！分段的话，就很难silu了，但是！可以在硬件里silu！也就是求exp！
void silu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim) {
    int total = batch * exhidden_dim * hidden_dim;
    int i = 0;
    for (; i + 8 <= total; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 s = _mm256_sigmoid_ps(x);
        __m256 y = _mm256_mul_ps(x, s);
        _mm256_storeu_ps(&output[i], y);
    }
    for (; i < total; i++) {
        float x = input[i];
        float s = 0.5f + 0.25f*x - 1.0f/48.0f*x*x*x + 1.0f/480.0f*x*x*x*x*x
                  - 17.0f/80640.0f*x*x*x*x*x*x*x + 31.0f/1451520.0f*x*x*x*x*x*x*x*x*x;
        output[i] = x * s;
    }
}



#endif