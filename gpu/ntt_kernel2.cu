#include <stdint.h>
#include <cuda_runtime.h>
#define BASE 65536ULL
#define LEN 128
#define NTT_N 256
#define SHARED_U64 384

extern __device__ void ntt_mul(uint32_t*, uint32_t*, uint32_t*, uint64_t*, int);
extern __device__ void barrett(uint32_t*, uint32_t*, uint32_t*, uint32_t*,
                                uint32_t*, uint32_t*, uint64_t*, int, int);

__global__ void modexp_kernel2(
    uint32_t *g_all, uint32_t *m_all,
    uint32_t *n_all, uint32_t *R_all,
    uint32_t *out_all,
    int m_bits, int n_bits, int num_tasks
) {
    int task_id=blockIdx.x, lane=threadIdx.x;
    if(task_id>=num_tasks) return;
    uint32_t *g=g_all+task_id*LEN,*m=m_all+task_id*LEN;
    uint32_t *n=n_all+task_id*LEN,*R=R_all+task_id*LEN;
    uint32_t *out=out_all+task_id*LEN;
    extern __shared__ uint8_t sr[];
    uint32_t *sm32=(uint32_t*)sr;
    uint32_t *T=sm32,*gcur=sm32+LEN,*tmp=sm32+2*LEN;
    uint32_t *q=sm32+4*LEN,*qn=sm32+5*LEN;
    uint32_t *nloc=sm32+7*LEN,*Rloc=sm32+8*LEN;
    uint64_t *sm64=(uint64_t*)(sm32+9*LEN);
    for(int i=lane;i<LEN;i+=32){
        T[i]=(i==0)?1:0;gcur[i]=g[i];
        nloc[i]=n[i];Rloc[i]=R[i];
    }
    __syncthreads();
    for(int i=0;i<m_bits;i++){
        int mb=(m[i/16]>>(i%16))&1;
        if(mb){ntt_mul(T,gcur,tmp,sm64,lane);barrett(tmp,nloc,Rloc,T,q,qn,sm64,n_bits,lane);}
        ntt_mul(gcur,gcur,tmp,sm64,lane);
        barrett(tmp,nloc,Rloc,gcur,q,qn,sm64,n_bits,lane);
    }
    for(int i=lane;i<LEN;i+=32) out[i]=T[i];
}
