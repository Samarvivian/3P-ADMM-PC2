#include <stdint.h>
#include <cuda_runtime.h>

#define BASE 65536ULL
#define LEN 128

// 把所有工作空间放在全局内存，完全消除栈
// 每个线程独立完成一个ModExp，最大化并发任务数

__device__ void dev_mul(uint32_t *a, uint32_t *b, uint32_t *c, int len) {
    for (int i = 0; i < 2*len; i++) c[i] = 0;
    for (int i = 0; i < len; i++) {
        if (a[i] == 0) continue;
        uint64_t carry = 0;
        for (int j = 0; j < len; j++) {
            uint64_t p = (uint64_t)a[i]*b[j] + c[i+j] + carry;
            c[i+j] = (uint32_t)(p % BASE);
            carry   = p / BASE;
        }
        int k = i + len;
        while (carry && k < 2*len) {
            uint64_t p = c[k] + carry;
            c[k] = (uint32_t)(p % BASE);
            carry = p / BASE;
            k++;
        }
    }
}

__device__ void dev_mul_2l(uint32_t *a, uint32_t *b, uint32_t *c, int alen, int blen) {
    int clen = alen + blen;
    for (int i = 0; i < clen; i++) c[i] = 0;
    for (int i = 0; i < alen; i++) {
        if (a[i] == 0) continue;
        uint64_t carry = 0;
        for (int j = 0; j < blen; j++) {
            uint64_t p = (uint64_t)a[i]*b[j] + c[i+j] + carry;
            c[i+j] = (uint32_t)(p % BASE);
            carry   = p / BASE;
        }
        int k = i + blen;
        while (carry && k < clen) {
            uint64_t p = c[k] + carry;
            c[k] = (uint32_t)(p % BASE);
            carry = p / BASE;
            k++;
        }
    }
}

__device__ void dev_sub(uint32_t *a, uint32_t *b, uint32_t *c, int len) {
    int64_t borrow = 0;
    for (int i = 0; i < len; i++) {
        int64_t d = (int64_t)a[i] - b[i] - borrow;
        if (d < 0) { d += BASE; borrow = 1; } else borrow = 0;
        c[i] = (uint32_t)d;
    }
}

__device__ int dev_geq(uint32_t *a, uint32_t *b, int len) {
    for (int i = len-1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;
}

__device__ void dev_shr(uint32_t *a, int k, uint32_t *out, int alen, int olen) {
    int ws = k/16, bs = k%16;
    for (int i = 0; i < olen; i++) {
        int src = i+ws;
        uint32_t lo = src<alen ? a[src]>>bs : 0;
        uint32_t hi = (bs>0&&src+1<alen)?((a[src+1]<<(16-bs))&0xFFFF):0;
        out[i] = lo|hi;
    }
}

__device__ void dev_barrett(
    uint32_t *x,    // 2*LEN
    uint32_t *n,    // LEN
    uint32_t *R,    // LEN
    uint32_t *r,    // LEN 输出
    uint32_t *xR,   // 3*LEN 临时
    uint32_t *q,    // LEN 临时
    uint32_t *qn,   // 2*LEN 临时
    int n_bits
) {
    dev_mul_2l(x, R, xR, 2*LEN, LEN);
    dev_shr(xR, 2*n_bits, q, 3*LEN, LEN);
    dev_mul(q, n, qn, LEN);
    dev_sub(x, qn, r, LEN);
    for (int i = 0; i < 3; i++)
        if (dev_geq(r, n, LEN)) dev_sub(r, n, r, LEN);
}

// 每个线程的工作空间大小
// T(LEN) + g_cur(LEN) + tmp(2*LEN) + xR(3*LEN) + q(LEN) + qn(2*LEN) = 10*LEN
#define WS (10*LEN)

__global__ void modexp_kernel(
    uint32_t *g_all, uint32_t *m_all,
    uint32_t *n_all, uint32_t *R_all,
    uint32_t *out_all,
    uint32_t *ws_all,   // 全局工作空间
    int m_bits, int n_bits, int num_tasks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tasks) return;

    uint32_t *g   = g_all   + tid*LEN;
    uint32_t *m   = m_all   + tid*LEN;
    uint32_t *n   = n_all   + tid*LEN;
    uint32_t *R   = R_all   + tid*LEN;
    uint32_t *out = out_all + tid*LEN;
    uint32_t *ws  = ws_all  + tid*WS;

    uint32_t *T    = ws;
    uint32_t *gcur = ws + LEN;
    uint32_t *tmp  = ws + 2*LEN;
    uint32_t *xR   = ws + 4*LEN;
    uint32_t *q    = ws + 7*LEN;
    uint32_t *qn   = ws + 8*LEN;

    for (int i = 0; i < LEN; i++) {
        T[i]    = (i==0)?1:0;
        gcur[i] = g[i];
    }

    for (int i = 0; i < m_bits; i++) {
        int m_bit = (m[i/16] >> (i%16)) & 1;
        if (m_bit) {
            dev_mul(T, gcur, tmp, LEN);
            dev_barrett(tmp, n, R, T, xR, q, qn, n_bits);
        }
        dev_mul(gcur, gcur, tmp, LEN);
        dev_barrett(tmp, n, R, gcur, xR, q, qn, n_bits);
    }

    for (int i = 0; i < LEN; i++) out[i] = T[i];
}
