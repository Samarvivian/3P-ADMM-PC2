#include <stdint.h>
#include <cuda_runtime.h>

#define BASE 65536ULL
#define MAX_LEN 256

__device__ void bigint_mul_2l(uint32_t *a, int la, uint32_t *b, int lb, uint32_t *c) {
    // a(la) * b(lb) -> c(la+lb)
    for (int i = 0; i < la+lb; i++) c[i] = 0;
    for (int i = 0; i < la; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < lb; j++) {
            uint64_t p = (uint64_t)a[i]*b[j] + c[i+j] + carry;
            c[i+j] = (uint32_t)(p % BASE);
            carry   = p / BASE;
        }
        int kk = i + lb;
        while (carry > 0 && kk < la+lb) {
            uint64_t pp = (uint64_t)c[kk] + carry;
            c[kk] = (uint32_t)(pp % BASE);
            carry = pp / BASE;
            kk++;
        }
    }
}

__device__ void bigint_sub(uint32_t *a, uint32_t *b, uint32_t *c, int len) {
    int64_t borrow = 0;
    for (int i = 0; i < len; i++) {
        int64_t d = (int64_t)a[i] - b[i] - borrow;
        if (d < 0) { d += BASE; borrow = 1; } else borrow = 0;
        c[i] = (uint32_t)d;
    }
}

__device__ int bigint_geq(uint32_t *a, uint32_t *b, int len) {
    for (int i = len-1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;
}

// 右移k个比特，输入长度in_len，输出长度out_len
__device__ void bigint_shr_bits(uint32_t *a, int k, uint32_t *out, int in_len, int out_len) {
    int word_shift = k / 16;
    int bit_shift  = k % 16;
    for (int i = 0; i < out_len; i++) {
        int src = i + word_shift;
        if (src >= in_len) { out[i] = 0; continue; }
        uint32_t lo = a[src] >> bit_shift;
        uint32_t hi = (bit_shift > 0 && src+1 < in_len) ?
                      ((a[src+1] << (16-bit_shift)) & 0xFFFF) : 0;
        out[i] = lo | hi;
    }
}

// Barrett Reduction: r = x mod n
// x: 2*len, n/R: len, k: n的比特数
__device__ void barrett_reduce(
    uint32_t *x, uint32_t *n, uint32_t *R,
    uint32_t *r, int len, int k
) {
    uint32_t xR[MAX_LEN*3];  // x(2*len) * R(len) = 3*len
    uint32_t q[MAX_LEN];
    uint32_t qn[MAX_LEN*2];

    // xR = x * R
    bigint_mul_2l(x, 2*len, R, len, xR);

    // q = xR >> 2k
    bigint_shr_bits(xR, 2*k, q, 3*len, len);

    // qn = q * n
    bigint_mul_2l(q, len, n, len, qn);

    // r = x - qn (低len位)
    bigint_sub(x, qn, r, len);

    // 修正
    for (int i = 0; i < 3; i++) {
        if (bigint_geq(r, n, len)) bigint_sub(r, n, r, len);
    }
}

__global__ void modexp_kernel(
    uint32_t *g_all, uint32_t *m_all,
    uint32_t *n_all, uint32_t *R_all,
    uint32_t *out_all,
    int len, int m_bits, int n_bits, int num_tasks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tasks) return;

    uint32_t *g   = g_all   + tid * MAX_LEN;
    uint32_t *m   = m_all   + tid * MAX_LEN;
    uint32_t *n   = n_all   + tid * MAX_LEN;
    uint32_t *R   = R_all   + tid * MAX_LEN;
    uint32_t *out = out_all + tid * MAX_LEN;

    uint32_t T[MAX_LEN];
    uint32_t g_cur[MAX_LEN];
    uint32_t tmp[MAX_LEN*2];

    for (int i = 0; i < len; i++) {
        T[i]     = (i == 0) ? 1 : 0;
        g_cur[i] = g[i];
    }

    for (int i = 0; i < m_bits; i++) {
        int word  = i / 16;
        int bit   = i % 16;
        int m_bit = (m[word] >> bit) & 1;

        if (m_bit) {
            bigint_mul_2l(T, len, g_cur, len, tmp);
            barrett_reduce(tmp, n, R, T, len, n_bits);
        }

        bigint_mul_2l(g_cur, len, g_cur, len, tmp);
        barrett_reduce(tmp, n, R, g_cur, len, n_bits);
    }

    for (int i = 0; i < len; i++) out[i] = T[i];
}
