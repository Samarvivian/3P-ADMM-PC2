#include <stdint.h>
#include <cuda_runtime.h>

#define BASE 65536ULL
#define LEN 128   // 2048bits / 16bits = 128个元素

// 一个block处理一个ModExp任务
// blockDim.x = LEN = 128，每个线程负责一个元素位置

__device__ void bigint_mul_shared(
    uint32_t *a, uint32_t *b, uint32_t *c,
    int len, int tid
) {
    // c = a * b，结果2*len
    // 每个线程计算c[tid]和c[tid+len]
    if (tid < 2*len) c[tid] = 0;
    __syncthreads();

    // 每个线程累加自己负责的列
    if (tid < len) {
        uint64_t sum_lo = 0, sum_hi = 0;
        for (int k = 0; k <= tid; k++) {
            sum_lo += (uint64_t)a[k] * b[tid-k];
        }
        for (int k = 0; k < len; k++) {
            if (tid+len-k >= 0 && tid+len-k < len)
                sum_hi += (uint64_t)a[k] * b[tid+len-k];
        }
        atomicAdd(&c[tid], (uint32_t)(sum_lo % BASE));
        atomicAdd(&c[tid+1], (uint32_t)(sum_lo / BASE));
    }
    __syncthreads();
}

__device__ void bigint_sub_shared(uint32_t *a, uint32_t *b, uint32_t *c, int len, int tid) {
    if (tid == 0) {
        int64_t borrow = 0;
        for (int i = 0; i < len; i++) {
            int64_t d = (int64_t)a[i] - b[i] - borrow;
            if (d < 0) { d += BASE; borrow = 1; } else borrow = 0;
            c[i] = (uint32_t)d;
        }
    }
    __syncthreads();
}

__device__ int bigint_geq_shared(uint32_t *a, uint32_t *b, int len, int tid) {
    __shared__ int result;
    if (tid == 0) {
        result = 1;
        for (int i = len-1; i >= 0; i--) {
            if (a[i] > b[i]) { result = 1; break; }
            if (a[i] < b[i]) { result = 0; break; }
        }
    }
    __syncthreads();
    return result;
}

__device__ void bigint_shr_bits_shared(uint32_t *a, int k, uint32_t *out, int in_len, int out_len, int tid) {
    int word_shift = k / 16;
    int bit_shift  = k % 16;
    if (tid < out_len) {
        int src = tid + word_shift;
        if (src >= in_len) { out[tid] = 0; return; }
        uint32_t lo = a[src] >> bit_shift;
        uint32_t hi = (bit_shift > 0 && src+1 < in_len) ?
                      ((a[src+1] << (16-bit_shift)) & 0xFFFF) : 0;
        out[tid] = lo | hi;
    }
    __syncthreads();
}

__global__ void modexp_kernel_v2(
    uint32_t *g_all, uint32_t *m_all,
    uint32_t *n_all, uint32_t *R_all,
    uint32_t *out_all,
    int len, int m_bits, int n_bits, int num_tasks
) {
    int task_id = blockIdx.x;
    int tid     = threadIdx.x;
    if (task_id >= num_tasks) return;

    uint32_t *g   = g_all   + task_id * LEN;
    uint32_t *m   = m_all   + task_id * LEN;
    uint32_t *n   = n_all   + task_id * LEN;
    uint32_t *R   = R_all   + task_id * LEN;
    uint32_t *out = out_all + task_id * LEN;

    // Shared memory布局
    extern __shared__ uint32_t smem[];
    uint32_t *T     = smem;                  // len
    uint32_t *g_cur = smem + LEN;            // len
    uint32_t *tmp   = smem + 2*LEN;          // 3*len (x*R需要)
    uint32_t *q     = smem + 5*LEN;          // len
    uint32_t *qn    = smem + 6*LEN;          // 2*len
    uint32_t *s_n   = smem + 8*LEN;          // len
    uint32_t *s_R   = smem + 9*LEN;          // len

    // 加载n和R到shared memory
    if (tid < len) {
        s_n[tid] = n[tid];
        s_R[tid] = R[tid];
        T[tid]     = (tid == 0) ? 1 : 0;
        g_cur[tid] = g[tid];
    }
    __syncthreads();

    // 平方乘法
    for (int i = 0; i < m_bits; i++) {
        int word  = i / 16;
        int bit   = i % 16;
        int m_bit = (m[word] >> bit) & 1;

        if (m_bit) {
            // tmp = T * g_cur
            if (tid < 3*len) tmp[tid] = 0;
            __syncthreads();
            if (tid < len) {
                for (int j = 0; j < len; j++) {
                    uint64_t p = (uint64_t)T[tid] * g_cur[j];
                    atomicAdd(&tmp[tid+j],   (uint32_t)(p % BASE));
                    if (tid+j+1 < 3*len)
                        atomicAdd(&tmp[tid+j+1], (uint32_t)(p / BASE));
                }
            }
            __syncthreads();

            // Barrett: T = tmp mod n
            // q = (tmp * R) >> 2*n_bits
            if (tid < 3*len) { uint32_t t = tmp[tid]; q[tid < len ? tid : 0] = 0; }
            __syncthreads();

            // 简化：tid==0串行做Barrett（后续再并行化）
            if (tid == 0) {
                // xR = tmp(2*len) * R(len)
                uint32_t xR[LEN*3];
                for (int j = 0; j < 3*len; j++) xR[j] = 0;
                for (int a = 0; a < 2*len; a++) {
                    uint64_t carry = 0;
                    for (int b = 0; b < len; b++) {
                        uint64_t p = (uint64_t)tmp[a]*s_R[b] + xR[a+b] + carry;
                        xR[a+b] = (uint32_t)(p % BASE);
                        carry    = p / BASE;
                    }
                    if (a+len < 3*len) xR[a+len] += (uint32_t)carry;
                }
                // q = xR >> 2*n_bits
                int ws = (2*n_bits)/16, bs = (2*n_bits)%16;
                for (int j = 0; j < len; j++) {
                    int src = j+ws;
                    uint32_t lo = src < 3*len ? xR[src]>>bs : 0;
                    uint32_t hi = (bs>0 && src+1<3*len) ? ((xR[src+1]<<(16-bs))&0xFFFF) : 0;
                    q[j] = lo|hi;
                }
                // qn = q * n
                uint32_t qn_loc[LEN*2];
                for (int j = 0; j < 2*len; j++) qn_loc[j] = 0;
                for (int a = 0; a < len; a++) {
                    uint64_t carry = 0;
                    for (int b = 0; b < len; b++) {
                        uint64_t p = (uint64_t)q[a]*s_n[b] + qn_loc[a+b] + carry;
                        qn_loc[a+b] = (uint32_t)(p%BASE);
                        carry = p/BASE;
                    }
                    if (a+len < 2*len) qn_loc[a+len] += (uint32_t)carry;
                }
                // T = tmp - qn
                int64_t borrow = 0;
                for (int j = 0; j < len; j++) {
                    int64_t d = (int64_t)tmp[j] - qn_loc[j] - borrow;
                    if (d<0){d+=BASE;borrow=1;}else borrow=0;
                    T[j] = (uint32_t)d;
                }
                for (int iter = 0; iter < 3; iter++) {
                    int geq = 1;
                    for (int j = len-1; j >= 0; j--) {
                        if (T[j]>s_n[j]){geq=1;break;}
                        if (T[j]<s_n[j]){geq=0;break;}
                    }
                    if (!geq) break;
                    borrow = 0;
                    for (int j = 0; j < len; j++) {
                        int64_t d = (int64_t)T[j]-s_n[j]-borrow;
                        if(d<0){d+=BASE;borrow=1;}else borrow=0;
                        T[j]=(uint32_t)d;
                    }
                }
            }
            __syncthreads();
        }

        // g_cur = g_cur^2 mod n（同上）
        if (tid < 3*len) tmp[tid] = 0;
        __syncthreads();
        if (tid < len) {
            for (int j = 0; j < len; j++) {
                uint64_t p = (uint64_t)g_cur[tid] * g_cur[j];
                atomicAdd(&tmp[tid+j],   (uint32_t)(p % BASE));
                if (tid+j+1 < 3*len)
                    atomicAdd(&tmp[tid+j+1], (uint32_t)(p / BASE));
            }
        }
        __syncthreads();

        if (tid == 0) {
            uint32_t xR[LEN*3];
            for (int j = 0; j < 3*len; j++) xR[j] = 0;
            for (int a = 0; a < 2*len; a++) {
                uint64_t carry = 0;
                for (int b2 = 0; b2 < len; b2++) {
                    uint64_t p = (uint64_t)tmp[a]*s_R[b2] + xR[a+b2] + carry;
                    xR[a+b2] = (uint32_t)(p%BASE); carry = p/BASE;
                }
                if (a+len < 3*len) xR[a+len] += (uint32_t)carry;
            }
            int ws = (2*n_bits)/16, bs2 = (2*n_bits)%16;
            for (int j = 0; j < len; j++) {
                int src = j+ws;
                uint32_t lo = src<3*len ? xR[src]>>bs2 : 0;
                uint32_t hi = (bs2>0&&src+1<3*len)?((xR[src+1]<<(16-bs2))&0xFFFF):0;
                q[j]=lo|hi;
            }
            uint32_t qn_loc[LEN*2];
            for (int j = 0; j < 2*len; j++) qn_loc[j] = 0;
            for (int a = 0; a < len; a++) {
                uint64_t carry = 0;
                for (int b2 = 0; b2 < len; b2++) {
                    uint64_t p = (uint64_t)q[a]*s_n[b2]+qn_loc[a+b2]+carry;
                    qn_loc[a+b2]=(uint32_t)(p%BASE); carry=p/BASE;
                }
                if (a+len<2*len) qn_loc[a+len]+=(uint32_t)carry;
            }
            int64_t borrow = 0;
            for (int j = 0; j < len; j++) {
                int64_t d = (int64_t)tmp[j]-qn_loc[j]-borrow;
                if(d<0){d+=BASE;borrow=1;}else borrow=0;
                g_cur[j]=(uint32_t)d;
            }
            for (int iter = 0; iter < 3; iter++) {
                int geq = 1;
                for (int j = len-1; j >= 0; j--) {
                    if(g_cur[j]>s_n[j]){geq=1;break;}
                    if(g_cur[j]<s_n[j]){geq=0;break;}
                }
                if (!geq) break;
                borrow = 0;
                for (int j = 0; j < len; j++) {
                    int64_t d=(int64_t)g_cur[j]-s_n[j]-borrow;
                    if(d<0){d+=BASE;borrow=1;}else borrow=0;
                    g_cur[j]=(uint32_t)d;
                }
            }
        }
        __syncthreads();
    }

    if (tid < len) out[tid] = T[tid];
}
