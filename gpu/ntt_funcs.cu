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
    uint64_t inv_n, int lane, uint64_t *smem) {
    for(int i=0;i<EPT;i++) smem[lane*EPT+i]=regs[i];
    __syncthreads();
    if(lane==0){
        for(int i=1,j=0;i<NTT_N;i++){
            int bit=NTT_N>>1;
            for(;j&bit;bit>>=1)j^=bit;j^=bit;
            if(i<j){uint64_t t=smem[i];smem[i]=smem[j];smem[j]=t;}
        }
    }
    __syncthreads();
    for(int i=0;i<EPT;i++) regs[i]=smem[lane*EPT+i];
    __syncthreads();
    for(int stage=0;stage<3;stage++){
        int span=1<<stage,gs=span*2;
        for(int i=0;i<EPT;i++){
            int gidx=lane*EPT+i,k=gidx%gs;
            if(k<span){
                int wi=k*(NTT_N/gs)%NTT_N;
                uint64_t ww=invert?iroots[wi]:roots[wi];
                uint64_t u=regs[i],v=regs[i+span]*ww%p;
                regs[i]=(u+v)%p;regs[i+span]=(u-v+p)%p;
            }
        }
    }
    for(int stage=3;stage<LOG_N;stage++){
        int span=1<<stage,gs=span*2,delta=span/EPT;
        bool up=((lane/delta)%2==0);
        uint64_t nr[EPT];
        for(int i=0;i<EPT;i++){
            int gidx=lane*EPT+i;
            int ks=up?gidx%span:(gidx-span)%span;
            int wi=ks*(NTT_N/gs)%NTT_N;
            uint64_t ww=invert?iroots[wi]:roots[wi];
            uint64_t partner=__shfl_xor_sync(0xffffffff,regs[i],delta);
            nr[i]=up?(regs[i]+partner*ww%p)%p:(partner-regs[i]*ww%p+p)%p;
        }
        for(int i=0;i<EPT;i++) regs[i]=nr[i];
    }
    if(invert) for(int i=0;i<EPT;i++) regs[i]=regs[i]*inv_n%p;
}

__device__ __noinline__ void ntt_mul(
    uint32_t *a, uint32_t *b, uint32_t *c, uint64_t *sm, int lane){
    uint64_t ra1[EPT],rb1[EPT],ra2[EPT],rb2[EPT];
    for(int i=0;i<EPT;i++){
        int idx=lane*EPT+i;
        ra1[i]=idx<LEN?a[idx]:0;rb1[i]=idx<LEN?b[idx]:0;
        ra2[i]=idx<LEN?a[idx]:0;rb2[i]=idx<LEN?b[idx]:0;
    }
    reg_ntt_256(ra1,false,P1,ROOT1,IROOT1,INV_N1,lane,sm);
    reg_ntt_256(rb1,false,P1,ROOT1,IROOT1,INV_N1,lane,sm);
    for(int i=0;i<EPT;i++) ra1[i]=ra1[i]*rb1[i]%P1;
    reg_ntt_256(ra1,true,P1,ROOT1,IROOT1,INV_N1,lane,sm);
    reg_ntt_256(ra2,false,P2,ROOT2,IROOT2,INV_N2,lane,sm);
    reg_ntt_256(rb2,false,P2,ROOT2,IROOT2,INV_N2,lane,sm);
    for(int i=0;i<EPT;i++) ra2[i]=ra2[i]*rb2[i]%P2;
    reg_ntt_256(ra2,true,P2,ROOT2,IROOT2,INV_N2,lane,sm);
    for(int i=0;i<EPT;i++){
        int idx=lane*EPT+i;
        uint64_t r1=ra1[i],r2=ra2[i];
        uint64_t t=(r2-r1%P2+P2)%P2*INV_P1_P2%P2;
        sm[idx]=r1+P1*t;
    }
    __syncthreads();
    if(lane==0){
        uint64_t carry=0;
        for(int i=0;i<2*LEN;i++){
            uint64_t v=(i<NTT_N?sm[i]:0)+carry;
            c[i]=(uint32_t)(v%BASE);carry=v/BASE;
        }
    }
    __syncthreads();
}

__device__ __noinline__ void barrett(
    uint32_t *x, uint32_t *n, uint32_t *R, uint32_t *r,
    uint32_t *q, uint32_t *qn, uint64_t *sm, int nb, int lane){
    for(int i=lane;i<SHARED_U64;i+=32) sm[i]=0;
    __syncthreads();
    for(int i=lane;i<2*LEN;i+=32){
        uint64_t xi=x[i];if(!xi)continue;
        for(int j=0;j<LEN;j++){
            uint64_t p=xi*R[j];
            atomicAdd((unsigned long long*)&sm[i+j],  p%BASE);
            atomicAdd((unsigned long long*)&sm[i+j+1],p/BASE);
        }
    }
    __syncthreads();
    if(lane==0){
        for(int i=0;i<SHARED_U64-1;i++){sm[i+1]+=sm[i]/BASE;sm[i]%=BASE;}
        int ws=(2*nb)/16,bs=(2*nb)%16;
        for(int i=0;i<LEN;i++){
            int src=i+ws;
            uint64_t lo=src<SHARED_U64?sm[src]>>bs:0;
            uint64_t hi=(bs>0&&src+1<SHARED_U64)?(sm[src+1]<<(16-bs))&0xFFFFULL:0;
            q[i]=(uint32_t)((lo|hi)%BASE);
        }
    }
    __syncthreads();
    ntt_mul(q,n,qn,sm,lane);
    if(lane==0){
        int64_t bw=0;
        for(int i=0;i<LEN;i++){
            int64_t d=(int64_t)x[i]-qn[i]-bw;
            if(d<0){d+=BASE;bw=1;}else bw=0;r[i]=(uint32_t)d;
        }
        for(int k=0;k<3;k++){
            int geq=1;
            for(int i=LEN-1;i>=0;i--){if(r[i]>n[i]){geq=1;break;}if(r[i]<n[i]){geq=0;break;}}
            if(!geq)break;bw=0;
            for(int i=0;i<LEN;i++){int64_t d=(int64_t)r[i]-n[i]-bw;if(d<0){d+=BASE;bw=1;}else bw=0;r[i]=(uint32_t)d;}
        }
    }
    __syncthreads();
}
