#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// Softmax 函数
void softmax(float* x, float* y, void* bitmask_ptr, int M, int N) {
    char *bitmask = malloc(M * N * sizeof(char));

    for (int i = 0; i < M * N; i++) {
        bitmask[i] = (((uint32_t*)bitmask_ptr)[i / 32] & (1U << (i % 32))) != 0;
    }

    for (int i = 0; i < M; i++) {
        float max_val = x[i * N];
        for (int j = 1; j < N; j++) {
            if (x[i*N + j] > max_val && bitmask[i * N + j]) max_val = x[i*N + j];
        }

        float sum = 0.0;
        for (int j = 0; j < N; j++) {
            y[i*N + j] = bitmask[i * N + j] ? exp(x[i*N + j] - max_val) : 0;
            sum += y[i*N + j];
        }
        for (int j = 0; j < N; j++) {
            y[i*N + j] /= sum;
        }
    }

}

