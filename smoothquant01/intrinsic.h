void smoothquantO1_stage1_getscale(float* A, float* scale, int M,int K);
void smoothquantO1_stage2_quant(float* A, int8_t* output,float* scale, int M,int K);
void smoothquantO1(float* A, int8_t* output,float* scale, int M,int K);