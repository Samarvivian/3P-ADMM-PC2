#include <stdint.h>
#include <cuda_runtime.h>
#include "ntt_roots.h"

#define BASE 65536ULL
#define LEN 128
#define NTT_N 256
#define EPT 8
#define P1 998244353ULL
#define P2 985661441ULL
#define LOG_N 8
#define SHARED_U64 384

__device__ __noinline__ void reg_ntt_256(
    uint64_t *regs, bool invert,
    uint64_t p, const uint64_t *roots, const uint64_t *iroots,
    uint64_t inv_n, int lane, uint64_t *smem
) {
    for(int i=0;i<EPT;i++) smem[lane*EPT+i]=regs[i];
    __syncthreads();
    if(lane==0) {
        for(int i=1,j=0;i<NTT_N;i++) {
            int bit=NTT_N>>1;
            for(;j&bit;bit>>=1) j^=bit;
            j^=bit;
            if(i<j) {uint64_t t=smem[i];smem[i]=smem[j];smem[j]=t;}
        }
    }
    __syncthreads();
    for(int i=0;i<EPT;i++) regs[i]=smem[lane*EPT+i];
    __syncthreads();

    for(int stage=0;stage<3;stage++) {
        int span=1<<stage, group_size=span*2;
        for(int i=0;i<EPT;i++) {
            int gidx=lane*EPT+i, k=gidx%group_size;
            if(k<span) {
                int w_idx=k*(NTT_N/group_size)%NTT_N;
                uint64_t w=invert?iroots[w_idx]:roots[w_idx];
                uint64_t u=regs[i], v=regs[i+span]*w%p;
                regs[i]=(u+v)%p; regs[i+span]=(u-v+p)%p;
            }
        }
    }

    for(int stage=3;stage<LOG_N;stage++) {
        int span=1<<stage, group_size=span*2;
        int delta=span/EPT;
        bool is_upper=((lane/delta)%2==0);
        uint64_t new_regs[EPT];
        for(int i=0;i<EPT;i++) {
            int gidx=lane*EPT+i;
            int k_in_span=is_upper?gidx%span:(gidx-span)%span;
            int w_idx=k_in_span*(NTT_N/group_size)%NTT_N;
            uint64_t w=invert?iroots[w_idx]:roots[w_idx];
            uint64_t partner=__shfl_xor_sync(0xffffffff,regs[i],delta);
            if(is_upper) {
                new_regs[i]=(regs[i]+partner*w%p)%p;
            } else {
                new_regs[i]=(partner-regs[i]*w%p+p)%p;
            }
        }
        for(int i=0;i<EPT;i++) regs[i]=new_regs[i];
    }
    if(invert) for(int i=0;i<EPT;i++) regs[i]=regs[i]*inv_n%p;
}

__device__ __noinline__ void reg_ntt_mul(
    uint32_t *a, uint32_t *b, uint32_t *c,
    uint64_t *smem, int lane
) {
    uint64_t ra1[EPT],rb1[EPT],ra2[EPT],rb2[EPT];
    for(int i=0;i<EPT;i++) {
        int idx=lane*EPT+i;
        ra1[i]=idx<LEN?a[idx]:0; rb1[i]=idx<LEN?b[idx]:0;
        ra2[i]=idx<LEN?a[idx]:0; rb2[i]=idx<LEN?b[idx]:0;
    }
    reg_ntt_256(ra1,false,P1,ROOT1,IROOT1,INV_N1,lane,smem);
    reg_ntt_256(rb1,false,P1,ROOT1,IROOT1,INV_N1,lane,smem);
    for(int i=0;i<EPT;i++) ra1[i]=ra1[i]*rb1[i]%P1;
    reg_ntt_256(ra1,true,P1,ROOT1,IROOT1,INV_N1,lane,smem);

    reg_ntt_256(ra2,false,P2,ROOT2,IROOT2,INV_N2,lane,smem);
    reg_ntt_256(rb2,false,P2,ROOT2,IROOT2,INV_N2,lane,smem);
    for(int i=0;i<EPT;i++) ra2[i]=ra2[i]*rb2[i]%P2;
    reg_ntt_256(ra2,true,P2,ROOT2,IROOT2,INV_N2,lane,smem);

    for(int i=0;i<EPT;i++) {
        int idx=lane*EPT+i;
        uint64_t r1=ra1[i],r2=ra2[i];
        uint64_t t=(r2-r1%P2+P2)%P2*INV_P1_P2%P2;
        smem[idx]=r1+P1*t;
    }
    __syncthreads();

    if(lane==0) {
        uint64_t carry=0;
        for(int i=0;i<2*LEN;i++) {
            uint64_t val=(i<NTT_N?smem[i]:0)+carry;
            c[i]=(uint32_t)(val%BASE); carry=val/BASE;
        }
    }
    __syncthreads();
}

__device__ void par_xR(
    uint32_t *x, uint32_t *R,
    uint64_t *smem, int lane
) {
    for(int i=lane;i<SHARED_U64;i+=32) smem[i]=0;
    __syncthreads();
    for(int i=lane;i<2*LEN;i+=32) {
        uint64_t xi=x[i];
        if(!xi) continue;
        for(int j=0;j<LEN;j++) {
            uint64_t prod=xi*R[j];
            atomicAdd((unsigned long long*)&smem[i+j],  prod%BASE);
            atomicAdd((unsigned long long*)&smem[i+j+1],prod/BASE);
        }
    }
    __syncthreads();
    if(lane==0) {
        for(int i=0;i<SHARED_U64-1;i++) {
            smem[i+1]+=smem[i]/BASE; smem[i]%=BASE;
        }
    }
    __syncthreads();
}

__device__ void dev_sub(uint32_t *a, uint32_t *b, uint32_t *c, int len) {
    int64_t borrow=0;
    for(int i=0;i<len;i++) {
        int64_t d=(int64_t)a[i]-b[i]-borrow;
        if(d<0){d+=BASE;borrow=1;}else borrow=0;
        c[i]=(uint32_t)d;
    }
}

__device__ int dev_geq(uint32_t *a, uint32_t *b, int len) {
    for(int i=len-1;i>=0;i--) {
        if(a[i]>b[i])return 1; if(a[i]<b[i])return 0;
    }
    return 1;
}

__device__ __noinline__ void barrett_v3(
    uint32_t *x, uint32_t *n, uint32_t *R, uint32_t *r,
    uint32_t *q, uint32_t *qn,
    uint64_t *smem, int n_bits, int lane
) {
    par_xR(x, R, smem, lane);

    if(lane==0) {
        int ws=(2*n_bits)/16, bs=(2*n_bits)%16;
        for(int i=0;i<LEN;i++) {
            int src=i+ws;
            uint64_t lo=src<SHARED_U64?smem[src]>>bs:0;
            uint64_t hi=(bs>0&&src+1<SHARED_U64)?(smem[src+1]<<(16-bs))&0xFFFFULL:0;
            q[i]=(uint32_t)((lo|hi)%BASE);
        }
    }
    __syncthreads();

    reg_ntt_mul(q, n, qn, smem, lane);

    if(lane==0) {
        dev_sub(x, qn, r, LEN);
        for(int i=0;i<3;i++)
            if(dev_geq(r,n,LEN)) dev_sub(r,n,r,LEN);
    }
    __syncthreads();
}

__global__
void reg_ntt_modexp_v3(
    uint32_t *g_all, uint32_t *m_all,
    uint32_t *n_all, uint32_t *R_all,
    uint32_t *out_all,
    int m_bits, int n_bits, int num_tasks
) {
    int task_id=blockIdx.x, lane=threadIdx.x;
    if(task_id>=num_tasks) return;

    uint32_t *g=g_all+task_id*LEN, *m=m_all+task_id*LEN;
    uint32_t *n=n_all+task_id*LEN, *R=R_all+task_id*LEN;
    uint32_t *out=out_all+task_id*LEN;

    extern __shared__ uint8_t smem_raw[];
    uint32_t *sm32=(uint32_t*)smem_raw;
    // 布局: T(L)+gcur(L)+tmp(2L)+q(L)+qn(2L)+nloc(L)+Rloc(L) = 9L uint32
    // 然后是 smem64(384 uint64)
    uint32_t *T=sm32, *gcur=sm32+LEN, *tmp=sm32+2*LEN;
    uint32_t *q=sm32+4*LEN, *qn=sm32+5*LEN;
    uint32_t *nloc=sm32+7*LEN, *Rloc=sm32+8*LEN;
    uint64_t *smem64=(uint64_t*)(sm32+9*LEN);

    for(int i=lane;i<LEN;i+=32) {
        T[i]=(i==0)?1:0; gcur[i]=g[i];
        nloc[i]=n[i]; Rloc[i]=R[i];
    }
    __syncthreads();

    #pragma unroll 1
    for(int i=0;i<m_bits;i++) {
        int m_bit=(m[i/16]>>(i%16))&1;
        if(m_bit) {
            reg_ntt_mul(T,gcur,tmp,smem64,lane);
            barrett_v3(tmp,nloc,Rloc,T,q,qn,smem64,n_bits,lane);
        }
        reg_ntt_mul(gcur,gcur,tmp,smem64,lane);
        barrett_v3(tmp,nloc,Rloc,gcur,q,qn,smem64,n_bits,lane);
    }

    for(int i=lane;i<LEN;i+=32) out[i]=T[i];
}
