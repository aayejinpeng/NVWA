#include <math.h>
#include <stdio.h>
#include <stdint.h>

#if defined(__riscv)// RISC-V 架构
#include <riscv_vector.h>

void fuse_shift_scale_resadd_relu(int32_t * input, int8_t * output, uint64_t stride_input, uint64_t stride_output,uint64_t shift_scale, uint64_t dim_I,int8_t *residual)
{
    // Implementation goes here
    for(int i = 0;i<dim_I;i++)
    {
        int32_t* row_input = (void*)input + i*stride_input;
        int8_t*  row_output = (void*)output + i*stride_output;
        int8_t*  row_residual = (void*)residual + i*stride_output;
        int j = 0;
        while (j < 64) {
            size_t vl = __riscv_vsetvl_e32m4(64 - j);

            // 1. load int32
            vint32m4_t vec_input = __riscv_vle32_v_i32m4(&row_input[j], vl);
            // 2. right shift
            vint8m1_t vec_shifted = __riscv_vnclip_wx_i8m1(__riscv_vnclip_wx_i16m2(vec_input,0,vl), shift_scale, vl);
            // 3. load residual
            vint8m1_t vec_residual = __riscv_vle8_v_i8m1(&row_residual[j], vl);
            // 4. add
            vint8m1_t vec_added = __riscv_vadd_vv_i8m1(vec_shifted, vec_residual, vl);
            // 5. relu clamp to [0,127]
            vec_added = __riscv_vmax_vx_i8m1(vec_added, 0, vl);
            vec_added = __riscv_vmin_vx_i8m1(vec_added, 127, vl);
            // 6. store output
            __riscv_vse8_v_i8m1(&row_output[j], vec_added, vl);

            j += vl;
        }
    }


}

void fuse_shift_scale_resadd_relu_dim_j_64(int32_t * input, int8_t * output, uint64_t stride_input, uint64_t stride_output,uint64_t shift_scale, uint64_t dim_I,int8_t *residual)
{
    // Implementation goes here
    for(int i = 0;i<dim_I;i++)
    {
        int32_t* row_input = (void*)input + i*stride_input;
        int8_t*  row_output = (void*)output + i*stride_output;
        int8_t*  row_residual = (void*)residual + i*stride_output;

        int vl = 64;
        // 1. load int32
        vint32m4_t vec_input = __riscv_vle32_v_i32m4(row_input, vl);
        // 2. right shift
        vint8m1_t vec_shifted = __riscv_vnclip_wx_i8m1(__riscv_vnclip_wx_i16m2(vec_input,shift_scale,vl), 0, vl);
        // 3. load residual
        vint8m1_t vec_residual = __riscv_vle8_v_i8m1(row_residual, vl);
        // 4. add
        vint8m1_t vec_added = __riscv_vadd_vv_i8m1(vec_shifted, vec_residual, vl);
        // 5. relu clamp to [0,127]
        vec_added = __riscv_vmax_vx_i8m1(vec_added, 0, vl);
        vec_added = __riscv_vmin_vx_i8m1(vec_added, 127, vl);
        // 6. store output
        __riscv_vse8_v_i8m1(row_output, vec_added, vl);
    }
}

#else
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

#endif