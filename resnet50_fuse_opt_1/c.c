#include <math.h>
#include <stdio.h>
#include <stdint.h>
//TODO: Update the function definitions to reflect the new OPSNAME
void fuse_shift_scale_resadd_relu(int32_t * input, int8_t * output, uint64_t stride_input, uint64_t stride_output,uint64_t shift_scale, uint64_t dim_I,int8_t *residual)
{
    // Implementation goes here
    for(int i = 0;i<dim_I;i++)
    {
        int32_t* row_input = (void*)input + i*stride_input;
        int8_t*  row_output = (void*)output + i*stride_output;
        int8_t*  row_residual = (void*)residual + i*stride_output;
        for(int j = 0;j<64;j++)
        {
            int32_t val = ((row_input[j] + (1 << (shift_scale - 1))) >> shift_scale) + row_residual[j];
            if(val < 0) val = 0;
            if(val > 127) val = 127;
            row_output[j] = (int8_t)(val);
        }
    }


}

