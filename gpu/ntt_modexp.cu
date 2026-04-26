#include <stdint.h>
#include <cuda_runtime.h>

#define BASE 65536ULL
#define LEN 128
#define NTT_N 256    // NTT点数，需要>=2*LEN
#define P1 998244353ULL
#define P2 985661441ULL
#define G1 3ULL
#define G2 3ULL

__device__ uint64_t power_mod(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t r = 1;
    a %= mod;
    while (b) {
        if (b & 1) r = r * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return r;
}

// 位反转置换
__device__ void bit_reverse(uint64_t *a, int n, int lane) {
    // 单线程做，n=256时只有8次操作
    if (lane == 0) {
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                uint64_t t = a[i]; a[i] = a[j]; a[j] = t;
            }
        }
    }
    __syncthreads();
}

// NTT变换，每个lane处理一部分蝶形运算
__device__ void ntt_kernel(uint64_t *a, int n, bool invert, uint64_t p, uint64_t g, int lane) {
    bit_reverse(a, n, lane);

    for (int length = 2; length <= n; length <<= 1) {
        uint64_t w = power_mod(g, (p-1)/length, p);
        if (invert) w = power_mod(w, p-2, p);

        // 每个lane处理length/2个蝶形
        int half = length >> 1;
        int total_butterflies = (n / length) * half;

        for (int idx = lane; idx < total_butterflies; idx += blockDim.x) {
            int group = idx / half;
            int k     = idx % half;
            int base  = group * length;

            uint64_t wn = power_mod(w, k, p);
            uint64_t u  = a[base + k];
            uint64_t v  = a[base + k + half] * wn % p;
            a[base + k]        = (u + v) % p;
            a[base + k + half] = (u - v + p) % p;
        }
        __syncthreads();
    }

    if (invert) {
        uint64_t inv_n = power_mod(n, p-2, p);
        for (int i = lane; i < n; i += blockDim.x)
            a[i] = a[i] * inv_n % p;
        __syncthreads();
    }
}

// NTT大整数乘法：a(LEN) * b(LEN) -> c(2*LEN)
__device__ void ntt_bigint_mul(
    uint32_t *a, uint32_t *b, uint32_t *c,
    uint64_t *fa1, uint64_t *fb1,  // P1的NTT工作空间，各NTT_N
    uint64_t *fa2, uint64_t *fb2,  // P2的NTT工作空间，各NTT_N
    int lane
) {
    // 加载输入
    for (int i = lane; i < NTT_N; i += blockDim.x) {
        fa1[i] = i < LEN ? a[i] : 0;
        fb1[i] = i < LEN ? b[i] : 0;
        fa2[i] = i < LEN ? a[i] : 0;
        fb2[i] = i < LEN ? b[i] : 0;
    }
    __syncthreads();

    // NTT变换
    ntt_kernel(fa1, NTT_N, false, P1, G1, lane);
    ntt_kernel(fb1, NTT_N, false, P1, G1, lane);
    ntt_kernel(fa2, NTT_N, false, P2, G2, lane);
    ntt_kernel(fb2, NTT_N, false, P2, G2, lane);

    // 点乘
    for (int i = lane; i < NTT_N; i += blockDim.x) {
        fa1[i] = fa1[i] * fb1[i] % P1;
        fa2[i] = fa2[i] * fb2[i] % P2;
    }
    __syncthreads();

    // 逆NTT
    ntt_kernel(fa1, NTT_N, true, P1, G1, lane);
    ntt_kernel(fa2, NTT_N, true, P2, G2, lane);

    // CRT合并 + 处理进位
    // inv_P1_P2 = P1^{-1} mod P2
    const uint64_t INV_P1_P2 = 657107549ULL; // power_mod(P1, P2-2, P2)预计算

    if (lane == 0) {
        uint64_t carry = 0;
        for (int i = 0; i < 2*LEN; i++) {
            uint64_t r1 = i < NTT_N ? fa1[i] : 0;
            uint64_t r2 = i < NTT_N ? fa2[i] : 0;
            uint64_t t  = (r2 - r1 % P2 + P2) % P2 * INV_P1_P2 % P2;
            uint64_t val = r1 + P1 * t + carry;
            c[i]  = (uint32_t)(val % BASE);
            carry = val / BASE;
        }
    }
    __syncthreads();
}

// Barrett Reduction（复用之前的逻辑）
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

__device__ void barrett_ntt(
    uint32_t *x,    // 2*LEN
    uint32_t *n,    // LEN
    uint32_t *R,    // LEN
    uint32_t *r,    // LEN 输出
    uint32_t *xR,   // 3*LEN
    uint32_t *q,    // LEN
    uint32_t *qn,   // 2*LEN
    uint64_t *fa1, uint64_t *fb1,
    uint64_t *fa2, uint64_t *fb2,
    int n_bits, int lane
) {
    // xR = x(2*LEN) * R(LEN) 用NTT
    // 但NTT_N=256只支持LEN=128，x是2*LEN=256...需要调整
    // 简化：用普通乘法做Barrett里的x*R（只做一次）
    if (lane == 0) {
        for (int i = 0; i < 3*LEN; i++) xR[i] = 0;
        for (int i = 0; i < 2*LEN; i++) {
            if (x[i] == 0) continue;
            uint64_t carry = 0;
            for (int j = 0; j < LEN; j++) {
                uint64_t p = (uint64_t)x[i]*R[j] + xR[i+j] + carry;
                xR[i+j] = (uint32_t)(p%BASE); carry = p/BASE;
            }
            int k = i+LEN;
            while(carry&&k<3*LEN){uint64_t p=xR[k]+carry;xR[k]=(uint32_t)(p%BASE);carry=p/BASE;k++;}
        }
        dev_shr(xR, 2*n_bits, q, 3*LEN, LEN);
    }
    __syncthreads();

    // qn = q * n 用NTT
    ntt_bigint_mul(q, n, qn, fa1, fb1, fa2, fb2, lane);

    if (lane == 0) {
        dev_sub(x, qn, r, LEN);
        for (int i = 0; i < 3; i++)
            if (dev_geq(r, n, LEN)) dev_sub(r, n, r, LEN);
    }
    __syncthreads();
}

// shared memory布局
// T(LEN) + gcur(LEN) + tmp(2*LEN) + xR(3*LEN) + q(LEN) + qn(2*LEN) = 10*LEN uint32
// fa1(NTT_N) + fb1(NTT_N) + fa2(NTT_N) + fb2(NTT_N) = 4*NTT_N uint64

__global__ void ntt_modexp_kernel(
    uint32_t *g_all, uint32_t *m_all,
    uint32_t *n_all, uint32_t *R_all,
    uint32_t *out_all,
    int m_bits, int n_bits, int num_tasks
) {
    int task_id = blockIdx.x;
    int lane    = threadIdx.x;
    if (task_id >= num_tasks) return;

    uint32_t *g   = g_all   + task_id*LEN;
    uint32_t *m   = m_all   + task_id*LEN;
    uint32_t *n   = n_all   + task_id*LEN;
    uint32_t *R   = R_all   + task_id*LEN;
    uint32_t *out = out_all + task_id*LEN;

    extern __shared__ uint8_t smem_raw[];
    uint32_t *smem32 = (uint32_t*)smem_raw;
    uint32_t *T    = smem32;
    uint32_t *gcur = smem32 + LEN;
    uint32_t *tmp  = smem32 + 2*LEN;
    uint32_t *xR   = smem32 + 4*LEN;
    uint32_t *q    = smem32 + 7*LEN;
    uint32_t *qn   = smem32 + 8*LEN;
    uint32_t *nloc = smem32 + 10*LEN;
    uint32_t *Rloc = smem32 + 11*LEN;

    // uint64工作空间对齐到8字节
    uint64_t *smem64 = (uint64_t*)(smem32 + 12*LEN);
    uint64_t *fa1 = smem64;
    uint64_t *fb1 = smem64 + NTT_N;
    uint64_t *fa2 = smem64 + 2*NTT_N;
    uint64_t *fb2 = smem64 + 3*NTT_N;

    for (int i = lane; i < LEN; i += blockDim.x) {
        T[i]    = (i==0)?1:0;
        gcur[i] = g[i];
        nloc[i] = n[i];
        Rloc[i] = R[i];
    }
    __syncthreads();

    for (int i = 0; i < m_bits; i++) {
        int m_bit = (m[i/16] >> (i%16)) & 1;

        if (m_bit) {
            ntt_bigint_mul(T, gcur, tmp, fa1, fb1, fa2, fb2, lane);
            barrett_ntt(tmp, nloc, Rloc, T, xR, q, qn, fa1, fb1, fa2, fb2, n_bits, lane);
        }

        ntt_bigint_mul(gcur, gcur, tmp, fa1, fb1, fa2, fb2, lane);
        barrett_ntt(tmp, nloc, Rloc, gcur, xR, q, qn, fa1, fb1, fa2, fb2, n_bits, lane);
    }

    for (int i = lane; i < LEN; i += blockDim.x)
        out[i] = T[i];
}
