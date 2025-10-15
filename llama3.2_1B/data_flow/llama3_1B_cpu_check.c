
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
// #include "easy_test_data.h"
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
// #include "cuteMarcoinstHelper.h"
#include <riscv_vector.h>
#include <assert.h>
#include "gloden_opt.h"

#define CUTEDataTypeI8I8I32     0     //I8 * I8 * I32
#define CUTEDataTypeF16F16F32   1     //FP16 * FP16 * FP32
#define CUTEDataTypeBF16BF16F32 2     //BF16 * BF16 * FP32
#define CUTEDataTypeTF32TF32F32 3     //TF32 * TF32 * FP32
#define CUTEDataTypeI8U8I32     4     //I8 * UI8 * I32
#define CUTEDataTypeU8I8I32     5     //U8 * I8 * I32
#define CUTEDataTypeU8U8I32     6     //U8 * U8 * I32
#define CUTEDataTypee4m3F32     7

#define TaskTypeTensorZeroLoad 0

#define LAYEROPT 2048
#define FUSEOPT 1024
#define NO_ACTIVATION 0
#define DEQUANT 1
#define ROPE 2
#define CVRT_TO_BF16 3
#define SOFTMAX 4
#define RMSNORM 5
#define RESADD 6
#define SILU 7
#define HADAMARD_PRODUCT 8
#define PER_TOKEN_QUANT 9                   //for smoothquant do quant
#define KVSCALE 10
#define MASKED_SOFTMAX 11
#define QUANTSTAGE1 12                      //for smoothquant max abs
#define FUSE_DEQUANT_ROPE_BF16CVRT              (FUSEOPT + 1)   //dequant+rope+bf16cvrt for proj_q,proj_k to score
#define FUSE_DEQUANT_BF16CVRT                   (FUSEOPT + 2)   //dequant+bf16cvrt for proj_v to attention
#define FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT    (FUSEOPT + 3)   //softmax+bf16cvrt+KVSCALE for score to attention
// #define FUSE_DEQUANT_RESADD_RMSNORM_QUANT       (FUSEOPT + 4)   //dequant+resadd+rmsnorm+quant for proj_o to ffn_gate,ffn_up
#define FUSE_DEQUANT_SILU                       (FUSEOPT + 5)   //dequant+silu for ffn_gate to ffn_up hadamard product
#define FUSE_DEQUANT_HADAMARD_QUANTSTAGE1       (FUSEOPT + 6)   //dequant+hadamard+get abs max for smoothquant ffn_up to ffn_down
#define FUSE_DEQUANT_RESADD                     (FUSEOPT + 7)   //dequant+resadd for ffn_down to output

#define SCALE_TYPE_NONE 0
#define SCALE_TYPE_PERTOKEN_A_PERTENSOR_B 1

#include <stdint.h> 
#define Tensor_M 64

#define SEQ_LEN 128
#define RMS_EPSILON 9.999999747378752e-06
#define EMBEDING_DIMENSION 2048
#define KEY_DIMENSION 64
#define VALUE_DIMENSION 64
#define N_HEAD_Q 32
#define N_HEAD_KV 8
#define MAX_CTX_LEN 8192
#define FFN_DIMENSION 8192
#define SQRT_KEY_DIMENSION 8.0
#define INV_SQRT_KEY_DIMENSION 0.125

// static inline float fast_sqrt(float x) {
//     float result;
//     __asm__ volatile ("fsqrt.s %0, %1" : "=f"(result) : "f"(x));
//     return result;
// }
// typedef __int16_t _Float16;

static float rope_theta[KEY_DIMENSION/2] __attribute__((aligned(64))) = {1.0000e+00, 6.6360e-01, 4.4037e-01, 2.9223e-01, 1.9392e-01, 1.2869e-01,
        8.5397e-02, 5.6670e-02, 3.7606e-02, 2.4955e-02, 1.6560e-02, 1.0990e-02,
        7.2927e-03, 4.8394e-03, 3.2114e-03, 1.6846e-03, 7.7941e-04, 2.8119e-04,
        8.8651e-05, 5.2790e-05, 3.1436e-05, 1.8720e-05, 1.1147e-05, 6.6380e-06,
        3.9528e-06, 2.3539e-06, 1.4017e-06, 8.3469e-07, 4.9704e-07, 2.9598e-07,
        1.7625e-07, 1.0496e-07};

// #define SOFTAMAX_MASKED_0 1

static int8_t bitmask_ptr[SEQ_LEN][SEQ_LEN] __attribute__((aligned(64))) = {0};

static float identity[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float attn_norm_weight[EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static int8_t proj_q_weight[N_HEAD_Q][KEY_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_q_scale[1] = {0};
static int8_t proj_k_weight[N_HEAD_KV][KEY_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_k_scale[1] = {0};
static int8_t proj_v_weight[N_HEAD_KV][VALUE_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_v_scale[1] = {0};
static int8_t proj_o_weight[EMBEDING_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_o_scale[1] = {0};
static float  ffn_norm_weight[EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static int8_t ffn_gate_weight[FFN_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  ffn_gate_scale[1] = {0};
static int8_t ffn_up_weight[FFN_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  ffn_up_scale[1] = {0};
static int8_t ffn_down_weight[EMBEDING_DIMENSION][FFN_DIMENSION] __attribute__((aligned(64))) = {0};
static float  ffn_down_scale[1] = {0};

static int8_t hidden_states_buf_q8_after_pre_rmsnorm[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  hidden_states_buf_q8_after_pre_rmsnorm_scale[SEQ_LEN] = {0};

static _Float16  proj_q_buf_q16[SEQ_LEN][N_HEAD_Q][KEY_DIMENSION] __attribute__((aligned(64))) = {0};
static _Float16  proj_k_buf_q16[SEQ_LEN][N_HEAD_KV][KEY_DIMENSION] __attribute__((aligned(64))) = {0};
static _Float16  proj_v_buf_q16[SEQ_LEN][N_HEAD_KV][VALUE_DIMENSION] __attribute__((aligned(64))) = {0};

static _Float16  scores_buf_q16[N_HEAD_Q][SEQ_LEN][SEQ_LEN] __attribute__((aligned(64))) = {0};

static int8_t attn_buf_q8[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  attn_buf_q8_scale[SEQ_LEN] = {0};

static float*  proj_o_buf_f32 = identity;

static int8_t proj_o_buf_after_RMSNORM_q8[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_o_buf_after_RMSNORM_q8_scale[SEQ_LEN] = {0};

static float ffn_gate_buf_f32[SEQ_LEN][FFN_DIMENSION] __attribute__((aligned(64))) = {0};

static float* ffn_up_buf_f32 = ffn_gate_buf_f32;

static int8_t ffn_up_buf_q8[SEQ_LEN][FFN_DIMENSION] __attribute__((aligned(64))) = {0};
static float ffn_up_buf_q8_scale[SEQ_LEN] __attribute__((aligned(64))) = {0};

static float* hidden_states_output= identity;

static float gloden_identity[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static int8_t gloden_hidden_states_buf_q8_after_pre_rmsnorm[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  gloden_hidden_states_buf_q8_after_pre_rmsnorm_scale[SEQ_LEN] = {0};
static _Float16  gloden_proj_q_buf_q16[SEQ_LEN][N_HEAD_Q][KEY_DIMENSION] __attribute__((aligned(64))) = {0};
static _Float16  gloden_proj_k_buf_q16[SEQ_LEN][N_HEAD_KV][KEY_DIMENSION] __attribute__((aligned(64))) = {0};
static _Float16  gloden_proj_v_buf_q16[SEQ_LEN][N_HEAD_KV][VALUE_DIMENSION] __attribute__((aligned(64))) = {0};
static _Float16  gloden_scores_buf_q16[N_HEAD_Q][SEQ_LEN][SEQ_LEN] __attribute__((aligned(64))) = {0};
static int8_t gloden_attn_buf_q8[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  gloden_attn_buf_q8_scale[SEQ_LEN] = {0};
static int8_t gloden_proj_o_buf_after_RMSNORM_q8[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  gloden_proj_o_buf_after_RMSNORM_q8_scale[SEQ_LEN] = {0};
static float gloden_ffn_gate_buf_f32[SEQ_LEN][FFN_DIMENSION] __attribute__((aligned(64))) = {0};
static int8_t gloden_ffn_up_buf_q8[SEQ_LEN][FFN_DIMENSION] __attribute__((aligned(64))) = {0};
static float gloden_ffn_up_buf_q8_scale[SEQ_LEN] __attribute__((aligned(64))) = {0};
static float* gloden_hidden_states_output= gloden_identity;
static float*  gloden_proj_o_buf_f32 = gloden_identity;
static float* gloden_ffn_up_buf_f32 = gloden_ffn_gate_buf_f32;
static float gloden_TCM_buffer[1024*1024*2] __attribute__((aligned(64))) = {0};
static float tmpbuffer0[2048*8192] __attribute__((aligned(64))); 
static float tmpbuffer1[2048*8192] __attribute__((aligned(64))); 
static float tmpbuffer2[2048*8192] __attribute__((aligned(64))); 


#include <math.h>

char *activation_name(int after_ops) {
  switch (after_ops) {
    case NO_ACTIVATION:
      return "NO_ACTIVATION";
    case FUSE_DEQUANT_ROPE_BF16CVRT:
      return "FUSE_DEQUANT_ROPE_BF16CVRT";
    case FUSE_DEQUANT_BF16CVRT:
        return "FUSE_DEQUANT_BF16CVRT";
    case FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT:
        return "FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT";
    case FUSE_DEQUANT_SILU:
        return "FUSE_DEQUANT_SILU";
    case FUSE_DEQUANT_HADAMARD_QUANTSTAGE1:
        return "FUSE_DEQUANT_HADAMARD_QUANTSTAGE1";
    case FUSE_DEQUANT_RESADD:
        return "FUSE_DEQUANT_RESADD";
    case PER_TOKEN_QUANT:
        return "PER_TOKEN_QUANT";
    case QUANTSTAGE1:
        return "QUANTSTAGE1";
    default:
      return "UNKNOWN";
  }
}


inline vfloat32m4_t vec_sin_small(vfloat32m4_t x, size_t vl) {
    // 计算 sin(x) 的泰勒展开，适用于小角度 x
    // sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9! - x¹¹/11!
    // sin(x) ≈ x * (c₁ + x² * (c₃ + x² * (c₅ + x² * (c₇ + x² * (c₉ + x² * c₁₁)))))
    vfloat32m4_t x2 = __riscv_vfmul_vv_f32m4(x, x, vl); // x²
    vfloat32m4_t c1  = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c3  = __riscv_vfmv_v_f_f32m4(-0.166666666667f, vl);
    vfloat32m4_t c5  = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c7  = __riscv_vfmv_v_f_f32m4(-0.0001984126984f, vl);
    vfloat32m4_t c9  = __riscv_vfmv_v_f_f32m4(0.000002755731922f, vl);
    vfloat32m4_t c11 = __riscv_vfmv_v_f_f32m4(-0.000000025052108f, vl);
    vfloat32m4_t result;
    // 计算并累加每一项
    // res = a1 + a2 * a3
    result = __riscv_vfmacc_vv_f32m4(c9,     c11, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c7,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c5,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c3,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c1,  result, x2, vl);
    result = __riscv_vfmul_vv_f32m4(result, x, vl); // 最后乘以 x
    return result;
}


inline vfloat32m4_t vec_sin(vfloat32m4_t x, size_t vl) {
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    vfloat32m4_t pi_vec = __riscv_vfmv_v_f_f32m4(PI, vl);
    vfloat32m4_t round = __riscv_vfdiv_vv_f32m4(new_rad, pi_vec, vl);
    vfloat32m4_t magic = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl); // 2²³ + 2²²，用于取整
    round = __riscv_vfadd_vv_f32m4(round, magic, vl);
    round = __riscv_vfsub_vv_f32m4(round, magic, vl);
    new_rad = __riscv_vfsub_vv_f32m4(new_rad, __riscv_vfmul_vv_f32m4(round, pi_vec, vl), vl);
    new_rad = __riscv_vfsub_vv_f32m4(new_rad, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    // 计算 sin(new_rad)
    vfloat32m4_t sin_result = vec_sin_small(new_rad, vl);
    // 根据 round 的奇偶性调整符号
    vuint32m4_t round_int = __riscv_vfcvt_xu_f_v_u32m4(round, vl);
    vuint32m4_t round_odd_int = __riscv_vand_vx_u32m4(round_int, 1, vl);
    round_odd_int = __riscv_vsll_vx_u32m4(round_odd_int, 31, vl); // 将最低位移到符号位
    vfloat32m4_t sign = __riscv_vreinterpret_v_u32m4_f32m4(round_odd_int);
    sin_result = __riscv_vfsgnjn_vv_f32m4(sin_result, sign, vl);
    return sin_result;
}

inline vfloat32m4_t vec_cos(vfloat32m4_t x, size_t vl) {
    // 将 x 映射到 [-π, π]
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    return vec_sin(new_rad, vl);
}


void fuse_ops_DEQUANT_ROPE_BF16CVRT(void * input,void *output,void * input_scale,void *weight_scale,int dim_i,int dim_j,uint64_t input_stride,uint64_t output_stride, void* output_scale)
{
    //stage 1: dequant input(I32) with scale to f32
    //stage 2: rope
    //stage 3: cvrt to fp16

    //input scale  is per token scale
    //weight scale is per tensor scale
    float_t * input_scale_f32 = (float_t *)input_scale;
    float_t * weight_scale_f32 = (float_t *)weight_scale;

    int head_dim = dim_j;
    int seq_len = dim_i;
    int pos = 0;//TODO:prefill only

    const int half_dim = head_dim / 2;


    for (int j = 0; j < seq_len; j++) {
        int pos_ = j + pos;

        // 计算当前token的起始位置
        int input_offset = j * input_stride;
        int output_offset = j * output_stride;
        int32_t* input_row = (int32_t*)(input + input_offset);
        _Float16* output_row = (_Float16*)(output + output_offset);
        size_t avl, vl;
        float_t scale = input_scale_f32[j] * weight_scale_f32[0];
        // 处理每两个元素（实部和虚部）
        vl = __riscv_vsetvl_e32m4(avl);
        for (int k = 0, avl = half_dim, vl = 0; avl > 0; k += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);

            // 加载rope_theta参数
            vfloat32m4_t theta_vec = __riscv_vle32_v_f32m4(rope_theta + k, vl);

            // 计算角度：angle = pos_ * inv_freq
            vfloat32m4_t angle_vec = __riscv_vfmul_vf_f32m4(theta_vec, pos_, vl);

            // 计算sin和cos
            vfloat32m4_t sin_vec, cos_vec;
            sin_vec = vec_sin(angle_vec, vl);
            cos_vec = vec_cos(angle_vec, vl);

            // 加载输入数据（实部和虚部）
            vint32m4_t real_in = __riscv_vlse32_v_i32m4(input_row + 2 * k, 2*sizeof(float), vl);//虽然慢，但是能够接受
            vint32m4_t imag_in = __riscv_vlse32_v_i32m4(input_row + 2 * k + 1, 2*sizeof(float), vl);//虽然慢，但是能够接受

            // 将I32转换为F32并应用缩放
            vfloat32m4_t real_in_f32 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(real_in, vl), scale, vl);
            vfloat32m4_t imag_in_f32 = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(imag_in, vl), scale, vl);

            // 计算旋转：real_out = real*cos - imag*sin
            // res = a1 * a2 - a3
            vfloat32m4_t real_out = __riscv_vfmsub_vv_f32m4(real_in_f32, cos_vec,__riscv_vfmul_vv_f32m4(imag_in_f32, sin_vec, vl), vl);

            // 计算旋转：imag_out = real*sin + imag*cos
            // res = a2 * a3 + a1
            vfloat32m4_t imag_out = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(real_in_f32, sin_vec, vl),imag_in_f32, cos_vec, vl);

            // 存储结果

            // 转化为fp16再存储
            vfloat16m2_t real_out_fp16 = __riscv_vfncvt_f_f_w_f16m2(real_out, vl);
            vfloat16m2_t imag_out_fp16 = __riscv_vfncvt_f_f_w_f16m2(imag_out, vl);
            __riscv_vse16_v_f16m2(output_row + k, real_out_fp16, vl);
            __riscv_vse16_v_f16m2(output_row + half_dim + k, imag_out_fp16, vl);
        }
    }
}

void fuse_ops_DEQUANT_BF16CVRT(void * input,void *output,void * input_scale,void *weight_scale,int dim_i,int dim_j,uint64_t input_stride,uint64_t output_stride, void* output_scale)
{
    //stage 1: dequant input(I32) with scale to f32
    //stage 2: cvrt to fp16

    //input scale  is per token scale
    //weight scale is per tensor scale
    int32_t* input_i32 = (int32_t*)input;
    _Float16* output_f16 = (_Float16*)output;
    float_t * input_scale_f32 = (float_t *)input_scale;
    float_t * weight_scale_f32 = (float_t *)weight_scale;

    int seq_len = dim_i;
    int headdim = dim_j;


    for (int j = 0; j < seq_len; j++) {
        size_t avl, vl;
        float_t scale = input_scale_f32[j] * weight_scale_f32[0];
        // 计算当前token的起始位置
        int input_offset = j * input_stride;
        int output_offset = j * output_stride;
        int32_t* input_row = (int32_t*)(input + input_offset);
        _Float16* output_row = (_Float16*)(output + output_offset);

        vl = __riscv_vsetvl_e32m4(avl);
        for (int k = 0, avl = headdim, vl = 0; avl > 0; k += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(input_row + k, vl);

            vfloat32m4_t deq_done = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale, vl);

            vfloat16m2_t deq_done_fp16 = __riscv_vfncvt_f_f_w_f16m2(deq_done, vl);
            __riscv_vse16_v_f16m2(output_row + k, deq_done_fp16, vl);
        }
    }
}



inline vfloat32m4_t vec_exp(vfloat32m4_t x, size_t vl) {
    // x = ln2 * a + b, 其中 b ∈ [0, ±ln2]
    // eˣ = 2ᵃ * eᵇ
    // 常数定义
    const float NEG_LN2 = -0.69314718056f;
    const float INV_LN2 = 1.44269504089f;
    const int32_t MAX_A = 127; // 防止指数溢出
    const int32_t MIN_A = -126; // 防止指数下溢

    // 计算 a = round(x / ln2)
    vfloat32m4_t af = __riscv_vfmul_vf_f32m4(x, INV_LN2, vl);
    vfloat32m4_t r = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl); // 2²³ + 2²²，用于取整
    vfloat32m4_t a = __riscv_vfadd_vv_f32m4(af, r, vl);
    a = __riscv_vfsub_vv_f32m4(a, r, vl);
    vint32m4_t a_int = __riscv_vfcvt_x_f_v_i32m4(a, vl);
    // 处理 a 的边界情况
    vbool8_t mask_max = __riscv_vmsgt_vx_i32m4_b8(a_int, MAX_A, vl);    // res[i] = op1[i] > op2
    vbool8_t mask_min = __riscv_vmslt_vx_i32m4_b8(a_int, MIN_A, vl);    // res[i] = op1[i] < op2

    // 计算 2ᵃ
    vint32m4_t biased_exponent = __riscv_vadd_vx_i32m4(a_int, 127, vl); // 加上偏置127
    biased_exponent = __riscv_vsll_vx_i32m4(biased_exponent, 23, vl); // 左移23位
    vfloat32m4_t a2 = __riscv_vreinterpret_v_i32m4_f32m4(biased_exponent); // 视为浮点数

    // 计算 b = x - a * ln2
    // res = a2 * a3 + a1
    vfloat32m4_t b = __riscv_vfmacc_vf_f32m4(x, NEG_LN2, a, vl);

    // 计算 eᵇ
    // 泰勒展开多项式系数
    vfloat32m4_t c0 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.166666666667f, vl);
    vfloat32m4_t c4 = __riscv_vfmv_v_f_f32m4(0.041666666667f, vl);
    vfloat32m4_t c5 = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c6 = __riscv_vfmv_v_f_f32m4(0.001388888889f, vl);
    vfloat32m4_t  p;
    // exp(b) ≈ c₀ + b * (c₁ + b * (c₂ + b * (c₃ + b * (c₄ + b * (c₅ + b * c₆)))))
    // res = a1 + a2 * a3
    p = __riscv_vfmacc_vv_f32m4(c5, c6, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c4,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c3,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c2,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c1,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c0,  p, b, vl);
    
    // 计算 2ᵃ * eᵇ
    p = __riscv_vfmul_vv_f32m4(a2, p, vl);

    // 处理边界情况
    // mask[i] ? op2[i] : op1[i]
    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(INFINITY, vl), mask_max, vl);
    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(0.0f, vl), mask_min, vl);
    return p;
}

inline void softmax_cvrtfp16(void* x, void* y, void* bitmask_ptr, int M, int N,uint64_t input_stride,uint64_t output_stride) {
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");
    for (int i = 0; i < M; i++) {
        int input_offset = i * input_stride;
        int output_offset = i * output_stride;
        float_t* input_row_f32 = (float_t*)(x + input_offset);
        _Float16* output_row_f16 = (_Float16*)(y + output_offset);

        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(N);
        // 第一步：计算最大值
        float max_val = input_row_f32[0];
        vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(max_val, vl_0);
        vl = __riscv_vsetvl_e32m4(avl);
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input_row_f32[j], vl);
            vbool8_t mask = __riscv_vlm_v_b8((uint8_t*)(bitmask_ptr + (i * N + j)/8), vl);
            // mask[i] ? op2[i] : op1[i]
            vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-INFINITY, vl), vec, mask, vl);
            max_vec = __riscv_vfmax_vv_f32m4(max_vec, vec, vl);
        }
        max_val = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(max_vec, __riscv_vfmv_v_f_f32m1(-INFINITY, vl_0), vl_0));
        // 第二步：计算指数和求和
        float sum_exp = 0.0f;
        vfloat32m4_t sumexp_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        // 向量化计算指数和求和
        vl = __riscv_vsetvl_e32m4(avl);
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input_row_f32[j], vl);
            vec = __riscv_vfsub_vf_f32m4(vec, max_val, vl); // 减去最大值

            // 应用掩码
            vbool8_t mask = __riscv_vlm_v_b8((uint8_t*)(bitmask_ptr + (i * N + j)/8), vl);
            // vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-INFINITY, vl), vec, mask, vl);
            vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-90, vl), vec, mask, vl);

            // 计算 exp(x)
            vfloat32m4_t exp_vec = vec_exp(vec, vl);
            
            // 存储结果
            __riscv_vse32_v_f32m4(&input_row_f32[j], exp_vec, vl);
            
            // 向量求和
            sumexp_vec = __riscv_vfadd_vv_f32m4(sumexp_vec, exp_vec, vl);
        }
        sum_exp = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumexp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
        
        // 第三步：归一化
        vfloat32m4_t inv_sum_exp_vec = __riscv_vfmv_v_f_f32m4(1.0f / sum_exp, vl_0);
        vl = __riscv_vsetvl_e32m4(avl);
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input_row_f32[j], vl);
            vfloat32m4_t normalized = __riscv_vfmul_vv_f32m4(vec, inv_sum_exp_vec, vl);
            vfloat16m2_t normalized_fp16 = __riscv_vfncvt_f_f_w_f16m2(normalized, vl);
            __riscv_vse16_v_f16m2((_Float16*)&output_row_f16[j], normalized_fp16, vl);
        }
    }
  
}


void fuse_ops_MASKED_SOFTMAX_KVSCALE_BF16CVRT(void * input,void *output,void * input_scale,void *weight_scale,int dim_i,int dim_j,uint64_t input_stride,uint64_t output_stride, void* output_scale)
{
    //stage 1: dequant input(I32) with scale to f32,multi INV_SQRT_KEY_DIMENSION
    //stage 2: do masked softmax
    //stage 3: cvrt to fp16


    int32_t* input_i32 = (int32_t*)input;
    _Float16* output_f16 = (_Float16*)output;
    float_t* softmabuf_f32 = (float_t*)input;

    int seq_len = dim_i;
    // int headdim = dim_j;


    for (int j = 0; j < seq_len; j++) {
        size_t avl, vl;
        float_t scale = INV_SQRT_KEY_DIMENSION;

        // 计算当前token的起始位置
        int input_offset = j * input_stride;
        int output_offset = j * output_stride;
        int32_t* input_row = (int32_t*)(input + input_offset);
        _Float16* output_row = (_Float16*)(output + output_offset);
        float_t* softmabuf_f32_row = (float_t*)input_row;
        vl = __riscv_vsetvl_e32m4(avl);
        for (int k = 0, avl = seq_len, vl = 0; avl > 0; k += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(input_row + k, vl);

            vfloat32m4_t deq_done = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale, vl);

            //store to float32_t buffer for softmax
            __riscv_vse32_v_f32m4(softmabuf_f32_row + k, deq_done, vl);
        }
    }
    softmax_cvrtfp16(softmabuf_f32, output_f16, bitmask_ptr, seq_len, seq_len,input_stride,output_stride);

}


void fuse_ops_DEQUANT_RESADD(void * input,void *output,void * input_scale,void *weight_scale,int dim_i,int dim_j,uint64_t input_stride,uint64_t output_stride, void* output_scale)
{
    //stage 1: dequant input(I32) with scale to f32
    //stage 2: add residual

    //input scale  is per token scale
    //weight scale is per tensor scale
    int32_t* input_i32 = (int32_t*)input;
    float_t* output_f32 = (float_t*)output;
    float_t * input_scale_f32 = (float_t *)input_scale;
    float_t * weight_scale_f32 = (float_t *)weight_scale;

    int head_dim = dim_j;
    int batch = 1;
    int n_head = 1;
    int seq_len = dim_i;
    int headdim = dim_j;


    for (int j = 0; j < seq_len; j++) {
        size_t avl, vl;
        vl = __riscv_vsetvl_e32m4(avl);
        float_t scale = input_scale_f32[j] * weight_scale_f32[0];
        int input_offset = j * input_stride;
        int output_offset = j * output_stride;
        int32_t* input_row = (int32_t*)(input + input_offset);
        float_t* output_row = (float_t*)(output + output_offset);

        for (int k = 0, avl = headdim, vl = 0; avl > 0; k += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(input_row + k, vl);
            vfloat32m4_t res_vec = __riscv_vle32_v_f32m4(output_row + k, vl);

            vfloat32m4_t deq_done = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale, vl);
            vfloat32m4_t result = __riscv_vfadd_vv_f32m4(deq_done, res_vec, vl);

            __riscv_vse32_v_f32m4(output_row + k, result, vl);
        }
    }

}

void silu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
{
    // printf("RISC-V specific implementation WORK IN PROGRESS\n");

}

void fuse_ops_DEQUANT_SILU(void * input,void *output,void * input_scale,void *weight_scale,int dim_i,int dim_j,uint64_t input_stride,uint64_t output_stride, void* output_scale)
{
    //stage 1: dequant input(I32) with scale to f32
    //stage 2: do SILU

    //input scale  is per token scale
    //weight scale is per tensor scale
    int32_t* input_i32 = (int32_t*)input;
    float_t* output_f32 = (float_t*)output;
    float_t * input_scale_f32 = (float_t *)input_scale;
    float_t * weight_scale_f32 = (float_t *)weight_scale;

    for (int s = 0; s < dim_i; s++) {
        int input_offset = s * input_stride;
        int output_offset = s * output_stride;
        int32_t* input_row = (int32_t*)(input + input_offset);
        float_t* output_row = (float_t*)(output + output_offset);
        size_t avl, vl;
        vl = __riscv_vsetvl_e32m4(avl);
        float_t scale = input_scale_f32[s] * weight_scale_f32[0];
        for (int i = 0, avl = dim_j; avl > 0; i += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(&input_row[i], vl);
            vfloat32m4_t x = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale, vl);
            vfloat32m4_t exp_neg_x = vec_exp(__riscv_vfneg_v_f32m4(x, vl), vl);
            vfloat32m4_t silu = __riscv_vfdiv_vv_f32m4(x, __riscv_vfadd_vf_f32m4(exp_neg_x, 1.0f, vl), vl);//div性能很差啊
            __riscv_vse32_v_f32m4(&output_row[i], silu, vl);
        }
    }
}

void fuse_ops_DEQUANT_HADAMARD_QUANTSTAGE1(void * input,void *output,void * input_scale,void *weight_scale,int dim_i,int dim_j,uint64_t input_stride,uint64_t output_stride, void* output_max)
{
    //stage 1: dequant input(I32) with scale to f32
    //stage 2: do HADAMARD product
    //stage 3: quant stage1

    //input scale  is per token scale
    //weight scale is per tensor scale
    int32_t* input_i32 = (int32_t*)input;
    float_t* output_f32 = (float_t*)output;
    float_t * input_scale_f32 = (float_t *)input_scale;
    float_t * weight_scale_f32 = (float_t *)weight_scale;
    float_t * output_max_f32 = (float_t *)output_max;

    for (int s = 0; s < dim_i; s++) {
        int input_offset = s * input_stride;
        int output_offset = s * output_stride;
        int32_t* input_row = (int32_t*)(input + input_offset);
        float_t* output_row = (float_t*)(output + output_offset);
        size_t avl, vl;
        float_t scale = input_scale_f32[s] * weight_scale_f32[0];
        vl = __riscv_vsetvl_e32m4(avl);
        vfloat32m4_t absmax_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (int i = 0, avl = dim_j; avl > 0; i += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(&input_row[i], vl);
            vfloat32m4_t x = __riscv_vfmul_vf_f32m4(__riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale, vl);

            vfloat32m4_t hadamard_vec = __riscv_vle32_v_f32m4(&output_row[i], vl);
            vfloat32m4_t hadamard_res = __riscv_vfmul_vv_f32m4(x, hadamard_vec, vl);

            __riscv_vse32_v_f32m4(&output_row[i], hadamard_res, vl);

            //quant stage 1
            //find absmax
            vfloat32m4_t abs_hadamard = __riscv_vfsgnj_vv_f32m4(hadamard_res, hadamard_res, vl);
            absmax_vec = __riscv_vfmax_vv_f32m4(absmax_vec, abs_hadamard, vl);
        }
        float_t current_max =__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(absmax_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl));
        output_max_f32[s] = current_max > output_max_f32[s] ? current_max : output_max_f32[s];
    }
}

void CUTE_TASK_END(uint64_t task_id)
{
    // printf("waiting for task end\n");
    //等待任务结束
    // uint64_t finish_tag = 1 << task_id;
    // uint64_t res1 = cute_marco_inst_fifo_finish_search();
    // while(!(res1&finish_tag))
    // {
    //     // printf("Waiting for finish task_id = %d\n",task_id);
    //     res1 = cute_marco_inst_fifo_finish_search();
    // }
    // cute_marco_inst_fifo_dequeue();
    // printf("SIM Task End\n");
    return;
}

int cute_buf_id = 0;
int CUTE_result_index = 0;
static char fakeTCM[1024*1024*2] __attribute__((aligned(64))) = {0};
void * CUTE_result[4] = {(void *) (fakeTCM), (void *) (fakeTCM + SEQ_LEN * SEQ_LEN * 4), (void *) (fakeTCM + SEQ_LEN * SEQ_LEN * 4 * 2), (void *) (fakeTCM + SEQ_LEN * SEQ_LEN * 4 * 3)};//double buffer use shuttle tcm
void * TCM_BUFF = (void *) (fakeTCM);//2MB

int issue_cute_matmul_marco_inst_sim(uint64_t ATensor_Base_Addr,uint64_t ATensor_M_Stride,
    uint64_t BTensor_Base_Addr,uint64_t BTensor_M_Stride,
    uint64_t BiasTensor_Base_Addr,uint64_t BiasTensor_M_Stride,
    uint64_t CTensor_Base_Addr,uint64_t CTensor_M_Stride,
    uint64_t M,uint64_t N,uint64_t K,
    uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t matmul_m_index)
{

    printf("SIM issue cute matmul marco inst\n");
    if(element_type == CUTEDataTypeI8I8I32)
    {
        for(int i=0;i<M;i++)
        {
            int8_t* A = (int8_t*)(ATensor_Base_Addr + i*ATensor_M_Stride);
            int8_t* B = (int8_t*)(BTensor_Base_Addr + i*BTensor_M_Stride);
            int32_t* C = (int32_t*)(CTensor_Base_Addr + i*CTensor_M_Stride);
            for(int j=0;j<N;j++)
            {
                int32_t acc = 0;
                for(int k=0;k<K;k++)
                {
                    int8_t a_val = A[k];
                    int8_t b_val = B[K];
                    acc += (int32_t)a_val * (int32_t)b_val;
                }
                C[j] = acc;
            }
        }
    }
    else
    {
        for(int i=0;i<M;i++)
        {
            _Float16* A = (_Float16*)(ATensor_Base_Addr + i*ATensor_M_Stride);
            _Float16* B = (_Float16*)(BTensor_Base_Addr + i*BTensor_M_Stride);
            float* C = (float*)(CTensor_Base_Addr + i*CTensor_M_Stride);
            for(int j=0;j<N;j++)
            {
                float acc = 0.0f;
                for(int k=0;k<K;k++)
                {
                    float a_val = (float)A[k];
                    float b_val = (float)B[k];
                    acc += a_val * b_val;
                }
                C[j] = acc;
            }
        }

    }
    return 1;
}

float_t quant_absmax_buff[SEQ_LEN]__attribute__((aligned(256))) = {0};

static void matmul_cute(size_t DIM_M, size_t DIM_N, size_t DIM_K,
        const void* A, const void* B, void* C,void* element_wise_tensor,void* scale_out,
        float_t* A_scale_factor, float_t* B_scale_factor,int scale_type,
        size_t stride_A, size_t stride_B, size_t stride_C,
        int datatype,int after_ops,int transpose_result)
{

  if(!(DIM_M % 64 == 0 && DIM_N % 64 == 0 && DIM_K % 64 == 0))
  {
    printf("Can't Till Now!");
    exit(1);
  }

//   void afater_operation(void * input,void *output,void * input_scale,void *weight_scale,int dim_i,int dim_j,uint64_t input_stride,uint64_t output_stride, void* output_scale);
  void (*afater_operation)(void* ,void* ,void* ,void* ,int ,int ,uint64_t ,uint64_t,void*) = NULL;

  switch (after_ops) {
    case FUSE_DEQUANT_ROPE_BF16CVRT://TODO:这里的rope刚好维度是64，所以可以展开
      afater_operation = fuse_ops_DEQUANT_ROPE_BF16CVRT;
      break;
    case FUSE_DEQUANT_BF16CVRT:
      afater_operation = fuse_ops_DEQUANT_BF16CVRT;
      break;
    case FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT:
      afater_operation = fuse_ops_MASKED_SOFTMAX_KVSCALE_BF16CVRT;
      break;
    case FUSE_DEQUANT_SILU:
      afater_operation = fuse_ops_DEQUANT_SILU;
      break;
    case FUSE_DEQUANT_HADAMARD_QUANTSTAGE1:
      afater_operation = fuse_ops_DEQUANT_HADAMARD_QUANTSTAGE1;
      break;
    case FUSE_DEQUANT_RESADD:
      afater_operation = fuse_ops_DEQUANT_RESADD;
      break;
    default:
      afater_operation = NULL;
      break;
  }

  int A_element_size = (datatype==CUTEDataTypeI8I8I32) ? 1 : ((datatype==CUTEDataTypeBF16BF16F32 || datatype==CUTEDataTypeF16F16F32) ? 2 : 4);
  int B_element_size = (datatype==CUTEDataTypeI8I8I32) ? 1 : ((datatype==CUTEDataTypeBF16BF16F32 || datatype==CUTEDataTypeF16F16F32) ? 2 : 4);
  int C_element_size = (after_ops==FUSE_DEQUANT_BF16CVRT || after_ops==FUSE_DEQUANT_ROPE_BF16CVRT || FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT) ? 2 : 4;

  if(after_ops != FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT && after_ops != NO_ACTIVATION)
  {
    int Tile_I = DIM_M / 64;
    int Tile_J = DIM_N / 64;

    int Application_M = 64;
    int Application_N = 64;
    int Application_K = DIM_K;

    int Application_stride_A = stride_A;
    int Application_stride_B = stride_B;
    int Application_stride_C = Tensor_M * 4;
    int Application_stride_D = 0;

    int Is_Transpose = transpose_result;
    int Is_repeating_row = 0;
    int Is_Zero_Load = 1;
    uint64_t bias_type = TaskTypeTensorZeroLoad;


    uint64_t wait_after_operation_cute_task_id = 0;
    uint64_t wait_after_operation_cute_task_id_pre = 0;

    const void* Tile_A = A;
    const void* Tile_B = B;
    void* Tile_C = CUTE_result[CUTE_result_index];
    void* Tile_D = NULL;

    wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst_sim(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C, Application_M, Application_N, Application_K, datatype, bias_type, Is_Transpose, 0);

    int i = 0;
    int j = 1;
    int pre_i = 0;
    int pre_j = 0;

    int acc_not_finish = 1;
    int next_CUTE_result_index = CUTE_result_index==3?0:CUTE_result_index+1;
    volatile int acc_finish = 0;
    for (i=0;i<Tile_I;i++)
    for (j=(i==0?1:0);j<Tile_J;j++)
    {

        CUTE_TASK_END(wait_after_operation_cute_task_id_pre);
        next_CUTE_result_index = CUTE_result_index==3?0:CUTE_result_index+1;

        Tile_A = A + i * 64 * stride_A;
        Tile_B = B + j * 64 * stride_B;
        Tile_C = CUTE_result[next_CUTE_result_index];//下一组任务
        Tile_D = NULL;

        wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst_sim(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C, Application_M, Application_N, Application_K, datatype, bias_type, Is_Transpose, 0);
        //   void afater_operation(void * input,void *output,void * input_scale,void *weight_scale,int dim_i,int dim_j,uint64_t input_stride,uint64_t output_stride, void* output_scale);
        printf("AFTER OPS= %s\n",activation_name(after_ops));
        afater_operation(CUTE_result[CUTE_result_index],(C+(transpose_result ? pre_j : pre_i)*64*stride_C+(transpose_result ? pre_i : pre_j)*64*C_element_size),A_scale_factor,B_scale_factor,64,64,Application_stride_C,stride_C,scale_out);

        CUTE_result_index = next_CUTE_result_index;
        pre_i = i;
        pre_j = j;
    }
    CUTE_TASK_END(wait_after_operation_cute_task_id_pre);
    printf("AFTER OPS= %s\n",activation_name(after_ops));
    afater_operation(CUTE_result[CUTE_result_index],(C+(transpose_result ? pre_j : pre_i)*64*stride_C+(transpose_result ? pre_i : pre_j)*64*C_element_size),A_scale_factor,B_scale_factor,64,64,Application_stride_C,stride_C,scale_out);

  }else if(after_ops != NO_ACTIVATION)
  {
    ////TODO:will get proj_o_buf_after_RMSNORM_q8_scale
    int Tile_I = DIM_M / 64;
    // int Tile_J = DIM_J / 64;

    int Application_M = 64;
    int Application_N = DIM_N;
    int Application_K = DIM_K;

    int Application_stride_A = stride_A;
    int Application_stride_B = stride_B;
    int Application_stride_C = DIM_N * 4;
    int Application_stride_D = 0;

    int Is_Transpose = 0;

    uint64_t bias_type = TaskTypeTensorZeroLoad;


    uint64_t wait_after_operation_cute_task_id_pre = 0;

    const void* Tile_A = A;
    const void* Tile_B = B;
    void* Tile_C = CUTE_result[CUTE_result_index];
    void* Tile_D = NULL;

    wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst_sim(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C, Application_M, Application_N, Application_K, datatype, bias_type, Is_Transpose, 0);

    int i = 1;
    int pre_i = 0;

    int acc_not_finish = 1;
    volatile int acc_finish = 0;
    for (i=1;i<Tile_I;i++)
    {

        CUTE_TASK_END(wait_after_operation_cute_task_id_pre);

        Tile_A = A + i * 64 * stride_A;
        Tile_B = B;
        Tile_C = CUTE_result[CUTE_result_index==0?1:0];
        Tile_D = NULL;
        wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst_sim(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C, Application_M, Application_N, Application_K, 1, bias_type, Is_Transpose, 0);

        printf("AFTER OPS= %s\n",activation_name(after_ops));
        afater_operation(CUTE_result[CUTE_result_index],(C+pre_i*64*stride_C),A_scale_factor,B_scale_factor,64,DIM_N,Application_stride_C,stride_C,scale_out);
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
    }
    CUTE_TASK_END(wait_after_operation_cute_task_id_pre);
    printf("AFTER OPS= %s\n",activation_name(after_ops));
    afater_operation(CUTE_result[CUTE_result_index],(C+pre_i*64*stride_C),A_scale_factor,B_scale_factor,64,DIM_N,Application_stride_C,stride_C,scale_out);
    
  }else 
  {
    //NO_ACTIVATION
    printf("AFTER OPS= %s\n",activation_name(after_ops));
    issue_cute_matmul_marco_inst_sim(A, stride_A, B, stride_B, NULL, 0, C, stride_C, DIM_M, DIM_N, DIM_K, datatype, TaskTypeTensorZeroLoad, transpose_result, 0);
  }
}


float idx_theta_buf[MAX_CTX_LEN][KEY_DIMENSION/2] __attribute__((aligned(256)));


static void matmul_cpu(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        float *A, float *B, float *D, float *C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        int act, bool repeating_bias, int transpose_result) {

    return;
    const int no_bias = D == NULL;

    for (size_t i = 0; i < DIM_I; i++) {
        for (size_t j = 0; j < DIM_J; j++) {

            size_t bias_row = repeating_bias ? 0 : i;
            float result = no_bias ? 0 : D[bias_row * stride_D + j];

            for (size_t k = 0; k < DIM_K; k++) {
                result += A[i * stride_A + k] * B[j * stride_B + k];
            }

            if (transpose_result) {
                C[j * stride_C + i] = result;
            } else {
                C[i * stride_C + j] = result;
            }
        }
    }
}


void smoothquant(float *input, int dim_i, int dim_j, int8_t *output, float_t* output_scale,bool need_stage1) 
{

    
    assert(dim_j%(64*4) == 0);
    assert(dim_i%16 == 0);
    
    if(need_stage1)
    {
        //先对A进行abs_max的量化
        //量化激活
        for (int i = 0; i < dim_i; i++) {
            float* row_A = &input[i * dim_j];
            int8_t* q_row_A = &output[i * dim_j];

            
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(dim_j);
            vl = vl_0;
            vfloat32m4_t tmp = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int j = 0, avl = dim_j; avl > 0; j += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(&row_A[j], vl);
                vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
                tmp = __riscv_vfmax_vv_f32m4(tmp, vfabs, vl);
            }
            vfloat32m1_t tmp_m1_max = __riscv_vfmv_v_f_f32m1(0.0f, vl_0);
            tmp_m1_max = __riscv_vfredmax_vs_f32m4_f32m1(tmp, tmp_m1_max, vl_0);

            float token_max = __riscv_vfmv_f_s_f32m1_f32(tmp_m1_max);

            const float d = token_max / (127.0f);
            const float id = d ? 1.0f / d : 0.0f;
            output_scale[i] = d;
            // 第三步，量化
            for (int j = 0, avl = dim_j; avl > 0; j += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row_A[j], vl);
                vfloat32m4_t x0  = __riscv_vfmul_vf_f32m4(v_x, id, vl);
                vint16m2_t   vi  = __riscv_vfncvt_x_f_w_i16m2(x0, vl);//默认舍入模式为round to nearest, ties to even
                vint8m1_t    vs  = __riscv_vncvt_x_x_w_i8m1(vi, vl);
                __riscv_vse8_v_i8m1(&q_row_A[j], vs, vl);
            }
        }
    }
    else
    {
        //量化激活
        for (int i = 0; i < dim_i; i++) {
            float* row_A = &input[i * dim_j];
            int8_t* q_row_A = &output[i * dim_j];

            const float d = output_scale[i];
            const float id = d ? 1.0f / d : 0.0f;
            // 第三步，量化
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(dim_j);
            vl = vl_0;
            for (int j = 0, avl = dim_j; avl > 0; j += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row_A[j], vl);
                vfloat32m4_t x0  = __riscv_vfmul_vf_f32m4(v_x, id, vl);
                vint16m2_t   vi  = __riscv_vfncvt_x_f_w_i16m2(x0, vl);//默认舍入模式为round to nearest, ties to even
                vint8m1_t    vs  = __riscv_vncvt_x_x_w_i8m1(vi, vl);
                __riscv_vse8_v_i8m1(&q_row_A[j], vs, vl);
            }
        }
    }

}

void RMSnorm(float* input, float* output, float* per_channle_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    assert(hidden_dim%(64*4) == 0);
    assert(seq_len%16 == 0);
    
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            // 计算平方和
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t vec_2 = __riscv_vfmul_vv_f32m4(vec, vec, vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec_2, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float rms = 1.0 / fast_sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            // 归一化并缩放
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t per_channle_scale_vec = __riscv_vle32_v_f32m4(&per_channle_scale[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, per_channle_scale_vec, vl);
                __riscv_vse32_v_f32m4(&output[b * seq_len * hidden_dim + j * hidden_dim + h], scaled_vec, vl);
            }
        }
    }
}

void RMSnorm_With_getabsmax_scale(float* input, float* output, float* per_channle_scale,float* per_token_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            // 计算平方和
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t vec_2 = __riscv_vfmul_vv_f32m4(vec, vec, vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec_2, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float rms = 1.0 / fast_sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(0.0, vl_0);
            // 归一化并缩放
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t per_channle_scale_vec = __riscv_vle32_v_f32m4(&per_channle_scale[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, per_channle_scale_vec, vl);
                __riscv_vse32_v_f32m4(&output[b * seq_len * hidden_dim + j * hidden_dim + h], scaled_vec, vl);
                vfloat32m4_t abs_max_vec = __riscv_vfabs_v_f32m4(scaled_vec, vl);
                max_vec = __riscv_vfmax_vv_f32m4(max_vec, abs_max_vec, vl);
            }

            float token_max = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(max_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            per_token_scale[b * seq_len + j] = token_max / (127.0f);
        }
    }
}

void reshape_to_head(float* input, float* output, int seq_len, int num_heads, int head_size) 
{
    //seq_len,numheads,head_size -> numheads,seq_len,head_size
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < head_size; k++) {
                output[j * seq_len * head_size + i * head_size + k] = input[i * num_heads * head_size + j * head_size + k];
            }
        }
    }
}

void reshape_to_headf16(uint16_t* input, uint16_t* output, int seq_len, int num_heads, int head_size) 
{
    //seq_len,numheads,head_size -> numheads,seq_len,head_size
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < head_size; k++) {
                output[j * seq_len * head_size + i * head_size + k] = input[i * num_heads * head_size + j * head_size + k];
            }
        }
    }
}

void reshape_to_seq(float* input, float* output, int seq_len, int num_heads, int head_size) 
{
    //numheads,seq_len,head_size --> seq_len,numheads,head_size
    for (int i = 0; i < num_heads; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < head_size; k++) {
                output[j * num_heads * head_size + i * head_size + k] = input[i * seq_len * head_size + j * head_size + k];
            }
        }
    }
}


uint64_t llama_block(
        int input_size, int d, int dk, int dv, int head_q, int head_kv, int dffn,
        float * hidden_states, float * q_buf, float * k_buf, float * v_buf, float * o_buf, float * o_transpose, float * k_interleave, float * v_interleave, float * mat_buf, float * scores_buf,
        float * gate, float * up, float * down)
{

    // uint64_t start = read_cycles();
    // input_size=6, d=2048, dk=64, dv=64, head_q=32, head_kv=8
    printf("[WorkLoad-(%5d,%5d,*****)LayerWise]RMSnorm_input\n",SEQ_LEN, EMBEDING_DIMENSION);

    memcpy(gloden_identity, hidden_states, sizeof(float) * SEQ_LEN * EMBEDING_DIMENSION);
    
    //pre rmsnorm
    RMSnorm_With_getabsmax_scale(identity, TCM_BUFF, attn_norm_weight, hidden_states_buf_q8_after_pre_rmsnorm_scale, RMS_EPSILON, 1, SEQ_LEN, EMBEDING_DIMENSION);
    smoothquant(TCM_BUFF,SEQ_LEN, EMBEDING_DIMENSION,hidden_states_buf_q8_after_pre_rmsnorm, hidden_states_buf_q8_after_pre_rmsnorm_scale, false);

    __gloden_RMSnorm(gloden_identity,gloden_TCM_buffer,attn_norm_weight,RMS_EPSILON,1,SEQ_LEN,EMBEDING_DIMENSION);
    __gloden_smoothquantO1(gloden_TCM_buffer,gloden_hidden_states_buf_q8_after_pre_rmsnorm,gloden_hidden_states_buf_q8_after_pre_rmsnorm_scale,SEQ_LEN,EMBEDING_DIMENSION);

    printf("check pre_rmsnorm and smoothquant:\n");
    int res = check_diff_byte(hidden_states_buf_q8_after_pre_rmsnorm,gloden_hidden_states_buf_q8_after_pre_rmsnorm,SEQ_LEN*EMBEDING_DIMENSION);
    if(res)
    {
        printf("\033[31mpre_rmsnorm and smoothquant have diff\033[0m\n");
        exit(-1);
    }

    //proj_q
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION, EMBEDING_DIMENSION,
        hidden_states_buf_q8_after_pre_rmsnorm, proj_q_weight, proj_q_buf_q16, NULL,NULL,
        hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_q_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, EMBEDING_DIMENSION * 2,//bf16
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_ROPE_BF16CVRT,0);

    __gloden_Q_matmul_I8I8I32_pertoken_pertensor(hidden_states_buf_q8_after_pre_rmsnorm,proj_q_weight,hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_q_scale,tmpbuffer1,SEQ_LEN,EMBEDING_DIMENSION,EMBEDING_DIMENSION);
    reshape_to_head(tmpbuffer1,tmpbuffer0,SEQ_LEN,N_HEAD_Q,KEY_DIMENSION);
    __gloden_rope(tmpbuffer0,tmpbuffer1,rope_theta,0,1,N_HEAD_Q,SEQ_LEN,KEY_DIMENSION);
    reshape_to_seq(tmpbuffer1,tmpbuffer0,SEQ_LEN,N_HEAD_Q,KEY_DIMENSION);
    __gloden_cvrtfp16(tmpbuffer0,gloden_proj_q_buf_q16,SEQ_LEN,EMBEDING_DIMENSION);
    printf("check proj_q with FUSE_DEQUANT_ROPE_BF16CVRT:\n");
    res = check_diff_byte(proj_q_buf_q16,gloden_proj_q_buf_q16,SEQ_LEN*EMBEDING_DIMENSION);
    if(res)
    {
        printf("\033[31mproj_q with FUSE_DEQUANT_ROPE_BF16CVRT have diff\033[0m\n");
        exit(-1);
    }
    

    
    //proj_k
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION / 4, EMBEDING_DIMENSION,
        hidden_states_buf_q8_after_pre_rmsnorm, proj_k_weight, proj_k_buf_q16, NULL,NULL,
        hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_k_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, EMBEDING_DIMENSION / 4 * 2,//bf16
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_ROPE_BF16CVRT,0);
    
    __gloden_Q_matmul_I8I8I32_pertoken_pertensor(hidden_states_buf_q8_after_pre_rmsnorm,proj_k_weight,hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_k_scale,tmpbuffer1,SEQ_LEN,EMBEDING_DIMENSION/4,EMBEDING_DIMENSION);
    reshape_to_head(tmpbuffer1,tmpbuffer0,SEQ_LEN,N_HEAD_KV,KEY_DIMENSION);
    __gloden_rope(tmpbuffer0,tmpbuffer1,rope_theta,0,1,N_HEAD_KV,SEQ_LEN,KEY_DIMENSION);
    reshape_to_seq(tmpbuffer1,tmpbuffer0,SEQ_LEN,N_HEAD_KV,KEY_DIMENSION);
    __gloden_cvrtfp16(tmpbuffer0,gloden_proj_k_buf_q16,SEQ_LEN,EMBEDING_DIMENSION/4);
    printf("check proj_k with FUSE_DEQUANT_ROPE_BF16CVRT:\n");
    res = check_diff_byte(proj_k_buf_q16,gloden_proj_k_buf_q16,SEQ_LEN*EMBEDING_DIMENSION/4);
    if(res)
    {
        printf("\033[31mproj_k with FUSE_DEQUANT_ROPE_BF16CVRT have diff\033[0m\n");
        exit(-1);
    }
    //proj_v
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION / 4, EMBEDING_DIMENSION,
        hidden_states_buf_q8_after_pre_rmsnorm, proj_v_weight, proj_v_buf_q16, NULL,NULL,
        hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_v_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, EMBEDING_DIMENSION / 4 * 2,//bf16
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_BF16CVRT,1);

    __gloden_Q_matmul_I8I8I32_pertoken_pertensor(hidden_states_buf_q8_after_pre_rmsnorm,proj_v_weight,hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_v_scale,tmpbuffer1,SEQ_LEN,EMBEDING_DIMENSION/4,EMBEDING_DIMENSION);
    
    __gloden_cvrtfp16(tmpbuffer0,gloden_proj_v_buf_q16,SEQ_LEN,EMBEDING_DIMENSION/4);
    
    printf("check proj_v with FUSE_DEQUANT_BF16CVRT:\n");
    res = check_diff_byte(proj_v_buf_q16,gloden_proj_v_buf_q16,SEQ_LEN*EMBEDING_DIMENSION/4);
    if(res)
    {
        printf("\033[31mproj_v with FUSE_DEQUANT_BF16CVRT have diff\033[0m\n");
        exit(-1);
    }
    
    //scores
    for (int i = 0; i < N_HEAD_Q; i++) {
        void *A = (void*)proj_q_buf_q16 + i * KEY_DIMENSION * 2;//*2 for bf16 2Byte 
        void *B = (void*)proj_k_buf_q16 + (i/(N_HEAD_Q/N_HEAD_KV)) * KEY_DIMENSION * 2;
        void *C = (void*)scores_buf_q16 + i * SEQ_LEN * SEQ_LEN * 2;

        int factor = 8;//TODO:!KVSCALE
        matmul_cute(SEQ_LEN, SEQ_LEN, KEY_DIMENSION,
            A, B, C, NULL,NULL,
            NULL,NULL,SCALE_TYPE_NONE,
            KEY_DIMENSION*N_HEAD_Q * 2, KEY_DIMENSION*N_HEAD_KV * 2, SEQ_LEN*2,
            CUTEDataTypeF16F16F32,FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT,0);
    }

    reshape_to_head(proj_q_buf_q16,tmpbuffer0,SEQ_LEN,N_HEAD_Q,KEY_DIMENSION);
    reshape_to_head(proj_k_buf_q16,tmpbuffer1,SEQ_LEN,N_HEAD_KV,KEY_DIMENSION);
    for(int i=0;i<N_HEAD_Q;i++)
    {
        uint16_t* A = (uint16_t*)tmpbuffer0 + i * SEQ_LEN * KEY_DIMENSION;
        uint16_t* B = (uint16_t*)tmpbuffer1 + (i/(N_HEAD_Q/N_HEAD_KV)) * SEQ_LEN * KEY_DIMENSION;
        float* C = (float*)tmpbuffer2 + i * SEQ_LEN * SEQ_LEN;
        __gloden_f16_matmul(A,B,C,SEQ_LEN,SEQ_LEN,KEY_DIMENSION);
        for(int j=0;j<SEQ_LEN*SEQ_LEN;j++)
        {
            C[j] = C[j]/8.0f;
        }
        __gloden_softmax(C,C,bitmask_ptr,SEQ_LEN,SEQ_LEN);
        __gloden_cvrtfp16(C,gloden_scores_buf_q16+i*SEQ_LEN*SEQ_LEN,SEQ_LEN,SEQ_LEN);
    }
    printf("check scores with FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT:\n");
    res = check_diff_byte((void*)scores_buf_q16,gloden_scores_buf_q16,N_HEAD_Q*SEQ_LEN*SEQ_LEN);
    if(res)
    {
        printf("\033[31mscores with FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT have diff\033[0m\n");
        exit(-1);
    }
    //attention
    for (int i = 0; i < N_HEAD_Q; i++) {
        float *A = (void*)scores_buf_q16 + i * SEQ_LEN * SEQ_LEN * 2;//32*128*128
        float *B = (void*)proj_v_buf_q16 + (i/(N_HEAD_Q/N_HEAD_KV)) * SEQ_LEN * VALUE_DIMENSION * 2;//8*64*128
        float *C = (void*)TCM_BUFF + i * VALUE_DIMENSION * 4;//128*2048
        matmul_cute(SEQ_LEN, VALUE_DIMENSION, SEQ_LEN,
            A, B, C,NULL,
            NULL,NULL,SCALE_TYPE_NONE, NULL,
            SEQ_LEN * 2, SEQ_LEN * 2, EMBEDING_DIMENSION*4,//
            CUTEDataTypeF16F16F32,NO_ACTIVATION,0);
    }

    //smoothquant attention
    smoothquant(TCM_BUFF,SEQ_LEN, EMBEDING_DIMENSION,attn_buf_q8,attn_buf_q8_scale, true);

    for(int i=0;i<N_HEAD_Q;i++)
    {
        uint16_t* A = (uint16_t*)scores_buf_q16 + i * SEQ_LEN * SEQ_LEN;
        uint16_t* B = (uint16_t*)proj_v_buf_q16 + (i/(N_HEAD_Q/N_HEAD_KV)) * VALUE_DIMENSION * SEQ_LEN;
        float* C = (float*)tmpbuffer0 + i * VALUE_DIMENSION * SEQ_LEN;
        __gloden_f16_matmul(A,B,C,SEQ_LEN,VALUE_DIMENSION,SEQ_LEN);
    }
    reshape_to_seq(tmpbuffer0,tmpbuffer1,SEQ_LEN,N_HEAD_Q,VALUE_DIMENSION);
    __gloden_smoothquantO1(tmpbuffer1,gloden_attn_buf_q8,gloden_attn_buf_q8_scale,SEQ_LEN,EMBEDING_DIMENSION);
    printf("check attention smoothquant:\n");
    res = check_diff_byte(attn_buf_q8,gloden_attn_buf_q8,SEQ_LEN*EMBEDING_DIMENSION);
    if(res)
    {
        printf("\033[31mattention smoothquant have diff\033[0m\n");
        exit(-1);
    }


    memcpy(tmpbuffer2, identity, sizeof(float) * SEQ_LEN * EMBEDING_DIMENSION);
    //proj_o
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION, EMBEDING_DIMENSION,
        attn_buf_q8, proj_o_weight, proj_o_buf_f32,identity,NULL,//proj_o_buf_f32 == identity == hidden_states_output
        attn_buf_q8_scale,proj_o_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, EMBEDING_DIMENSION*4,//fp32
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_RESADD,0);

    RMSnorm_With_getabsmax_scale(proj_o_buf_f32, TCM_BUFF, ffn_norm_weight, proj_o_buf_after_RMSNORM_q8_scale, RMS_EPSILON, 1, SEQ_LEN, EMBEDING_DIMENSION);
    smoothquant(TCM_BUFF,SEQ_LEN, EMBEDING_DIMENSION,proj_o_buf_after_RMSNORM_q8, proj_o_buf_after_RMSNORM_q8_scale, false);

    __gloden_Q_matmul_I8I8I32(attn_buf_q8,proj_o_weight,attn_buf_q8_scale,proj_o_scale,tmpbuffer0,SEQ_LEN,EMBEDING_DIMENSION,EMBEDING_DIMENSION);
    for(int i=0;i<SEQ_LEN*EMBEDING_DIMENSION;i++)
    {
        tmpbuffer1[i] = tmpbuffer0[i] + tmpbuffer2[i];
    }
    __gloden_RMSnorm(tmpbuffer1,gloden_TCM_buffer,ffn_norm_weight,RMS_EPSILON,1,SEQ_LEN,EMBEDING_DIMENSION);
    __gloden_smoothquantO1(gloden_TCM_buffer,gloden_proj_o_buf_after_RMSNORM_q8,gloden_proj_o_buf_after_RMSNORM_q8_scale,SEQ_LEN,EMBEDING_DIMENSION);
    printf("check proj_o and rmsnorm and smoothquant:\n");
    res = check_diff_byte(proj_o_buf_after_RMSNORM_q8,gloden_proj_o_buf_after_RMSNORM_q8,SEQ_LEN*EMBEDING_DIMENSION);
    if(res)
    {
        printf("\033[31mproj_o and rmsnorm and smoothquant have diff\033[0m\n");
        exit(-1);
    }

    //ffn_gate
    matmul_cute(SEQ_LEN, FFN_DIMENSION, EMBEDING_DIMENSION,
        proj_o_buf_after_RMSNORM_q8, ffn_gate_weight, ffn_gate_buf_f32, NULL,NULL,//ffn_gate_buf_f32 == ffn_up_buf_f32
        proj_o_buf_after_RMSNORM_q8_scale,ffn_gate_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, FFN_DIMENSION * 4,//fp32
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_SILU,0);
    
    __gloden_Q_matmul_I8I8I32(proj_o_buf_after_RMSNORM_q8,ffn_gate_weight,proj_o_buf_after_RMSNORM_q8_scale,ffn_gate_scale,tmpbuffer0,SEQ_LEN,FFN_DIMENSION,EMBEDING_DIMENSION);
    __gloden_silu(tmpbuffer0,gloden_ffn_gate_buf_f32,1,FFN_DIMENSION,EMBEDING_DIMENSION);

    //ffn_up
    matmul_cute(SEQ_LEN, FFN_DIMENSION, EMBEDING_DIMENSION,
        proj_o_buf_after_RMSNORM_q8, ffn_up_weight, ffn_up_buf_f32, ffn_gate_buf_f32,ffn_up_buf_q8_scale, //ffn_gate_buf_f32 == ffn_up_buf_f32
        proj_o_buf_after_RMSNORM_q8_scale,ffn_up_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, FFN_DIMENSION * 4,//fp32
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_HADAMARD_QUANTSTAGE1,0);

    smoothquant(ffn_up_buf_f32,SEQ_LEN, FFN_DIMENSION,ffn_up_buf_q8, ffn_up_buf_q8_scale, false);
    __gloden_Q_matmul_I8I8I32(proj_o_buf_after_RMSNORM_q8,ffn_up_weight,proj_o_buf_after_RMSNORM_q8_scale,ffn_up_scale,tmpbuffer0,SEQ_LEN,FFN_DIMENSION,EMBEDING_DIMENSION);
    for(int i=0;i<SEQ_LEN*FFN_DIMENSION;i++)
    {
        tmpbuffer1[i] = tmpbuffer0[i] * ((float_t *)gloden_ffn_gate_buf_f32)[i];
    }
    __gloden_smoothquantO1(tmpbuffer1,gloden_ffn_up_buf_q8,gloden_ffn_up_buf_q8_scale,SEQ_LEN,FFN_DIMENSION);
    printf("check ffn_up and silu and smoothquant:\n");
    res = check_diff_byte(ffn_up_buf_q8,gloden_ffn_up_buf_q8,SEQ_LEN*FFN_DIMENSION);
    if(res)
    {
        printf("\033[31mffn_up and silu and smoothquant have diff\033[0m\n");
        exit(-1);
    }
    //ffn_down
    memcpy(tmpbuffer2, proj_o_buf_f32, sizeof(float) * SEQ_LEN * EMBEDING_DIMENSION);//identity
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION, FFN_DIMENSION,
        ffn_up_buf_f32, ffn_down_weight, hidden_states_output,proj_o_buf_f32,NULL,//proj_o_buf_f32 == identity == hidden_states_output
        ffn_up_buf_q8_scale,ffn_down_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, FFN_DIMENSION, EMBEDING_DIMENSION * 4,//fp32
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_RESADD,0);
    __gloden_Q_matmul_I8I8I32(ffn_up_buf_q8,ffn_down_weight,ffn_up_buf_q8_scale,ffn_down_scale,tmpbuffer0,SEQ_LEN,EMBEDING_DIMENSION,FFN_DIMENSION);
    for(int i=0;i<SEQ_LEN*EMBEDING_DIMENSION;i++)
    {
        gloden_hidden_states_output[i] = tmpbuffer0[i] + tmpbuffer2[i];
    }
    printf("check ffn_down and resadd:\n");
    res = check_diff_byte(hidden_states_output,gloden_hidden_states_output,SEQ_LEN*EMBEDING_DIMENSION);
    if(res)
    {
        printf("\033[31mffn_down and resadd have diff\033[0m\n");
        exit(-1);
    }

    return 0;
}


#define LLAMA(input_size, d, dk, dv, head_q, head_kv, dffn) ({ \
    static float hidden_states[input_size][d];\
    static float q_buf[head_q][input_size][dk]; \
    static float k_buf[head_kv][input_size][dk]; \
    static float v_buf[head_kv][input_size][dv]; \
    static float o_buf[head_q][input_size][dv]; \
    static float o_transpose[input_size][head_q][dv]; \
    static float k_interleave[head_q][input_size][dk]; \
    static float v_interleave[head_q][input_size * dv]; \
    static float mat_buf[head_q][input_size][dk]; \
    static float scores_buf[head_q][input_size][input_size]; \
    static float gate[input_size][dffn]; \
    static float up[input_size][dffn]; \
    static float down[input_size][d]; \
    uint64_t cycles = llama_block( \
            input_size, d, dk, dv, head_q, head_kv, dffn, \
            hidden_states, q_buf, k_buf, v_buf, o_buf, o_transpose, k_interleave, v_interleave, mat_buf, scores_buf, \
            gate, up, down \
    ); \
    \
    cycles; \
})

#define PRINT_LLAMA(input_size, d, dk, dv, head_q, head_kv, dffn) { \
    \
    printf("input_size=%d, d=%d, dk=%d, dv=%d, head_q=%d, head_kv=%d\n", \
            input_size, d, dk, dv, head_q, head_kv); \
    uint64_t cycles = LLAMA(input_size, d, dk, dv, head_q, head_kv, dffn); \
}

int main (int argc, char * argv[]) {

    // gemmini_flush(0);

    PRINT_LLAMA(/*input_size=*/SEQ_LEN, /*d=*/EMBEDING_DIMENSION, /*dk=*/KEY_DIMENSION, /*dv=*/VALUE_DIMENSION, /*head_q=*/N_HEAD_Q, /*head_kv=*/N_HEAD_KV, /*dffn=*/FFN_DIMENSION);

}

