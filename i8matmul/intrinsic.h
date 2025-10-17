void Q_matmul_I8I8I32(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K);
void pertoken_pertensor_scale(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K);
void Q_matmul_I8I8I32_pertoken_pertensor(int8_t* A, int8_t* B, float* A_scale, float* B_scale, float* output, int M, int N,int K);