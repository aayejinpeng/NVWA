#include <math.h>

#if defined(__riscv)// RISC-V 架构
//TODO: Update the function definitions to reflect the new OPSNAME
void rope(float* x, float* y, int M, int N) {

}
#else
//TODO: Update the function definitions to reflect the new OPSNAME
void rope(float* x, float* y, int M, int N) {
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