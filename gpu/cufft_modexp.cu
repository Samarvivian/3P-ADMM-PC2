
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define BASE 65536ULL
#define LEN 128
#define FFT_N 256
#define FFT_L 512

__global__ void cmul(cufftDoubleComplex *a, cufftDoubleComplex *b,
                     cufftDoubleComplex *c, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    double ar=a[i].x,ai=a[i].y,br=b[i].x,bi=b[i].y;
    c[i].x=ar*br-ai*bi; c[i].y=ar*bi+ai*br;
}

__global__ void norm_round(cufftDoubleComplex *c, double *out, int n, double scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    out[i]=round(c[i].x*scale);
}

__global__ void load_complex(uint32_t *src, cufftDoubleComplex *dst,
                              int N, int src_len, int fft_n) {
    int t=blockIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=N||j>=fft_n) return;
    dst[t*fft_n+j].x = j<src_len ? (double)src[t*src_len+j] : 0.0;
    dst[t*fft_n+j].y = 0.0;
}

__global__ void carry_prop(double *in, uint32_t *out,
                            int N, int fft_n, int out_len) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=N) return;
    uint64_t carry=0;
    for(int j=0;j<out_len;j++){
        uint64_t val=(uint64_t)(in[t*fft_n+j])+carry;
        out[t*out_len+j]=(uint32_t)(val%BASE);
        carry=val/BASE;
    }
}

__global__ void shr_kernel(uint32_t *xR, uint32_t *q,
                            int N, int xR_len, int n_bits) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=N) return;
    uint32_t *src=xR+t*xR_len, *dst=q+t*LEN;
    int ws=(2*n_bits)/16, bs=(2*n_bits)%16;
    for(int i=0;i<LEN;i++){
        int s=i+ws;
        uint32_t lo=s<xR_len?src[s]>>bs:0;
        uint32_t hi=(bs>0&&s+1<xR_len)?((src[s+1]<<(16-bs))&0xFFFF):0;
        dst[i]=lo|hi;
    }
}

__global__ void sub_correct(uint32_t *x, uint32_t *qn, uint32_t *r,
                             uint32_t *n, int N) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=N) return;
    uint32_t *px=x+t*LEN,*pqn=qn+t*LEN;
    uint32_t *pr=r+t*LEN,*pn=n+t*LEN;
    int64_t borrow=0;
    for(int i=0;i<LEN;i++){
        int64_t d=(int64_t)px[i]-pqn[i]-borrow;
        if(d<0){d+=BASE;borrow=1;}else borrow=0;
        pr[i]=(uint32_t)d;
    }
    for(int k=0;k<3;k++){
        int geq=1;
        for(int i=LEN-1;i>=0;i--){
            if(pr[i]>pn[i]){geq=1;break;}
            if(pr[i]<pn[i]){geq=0;break;}
        }
        if(!geq) break;
        borrow=0;
        for(int i=0;i<LEN;i++){
            int64_t d=(int64_t)pr[i]-pn[i]-borrow;
            if(d<0){d+=BASE;borrow=1;}else borrow=0;
            pr[i]=(uint32_t)d;
        }
    }
}

// 从GPU上的m数组提取第i位，存到d_mbit
__global__ void extract_bit(uint32_t *m_all, uint8_t *mbit, int bit_idx, int N) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=N) return;
    int word=bit_idx/16, bit=bit_idx%16;
    mbit[t]=(m_all[t*LEN+word]>>bit)&1;
}

// 条件拷贝
__global__ void cond_copy(uint32_t *src, uint32_t *dst,
                           uint8_t *mbit, int N) {
    int t=blockIdx.x, j=threadIdx.x;
    if(t>=N) return;
    if(mbit[t])
        for(int i=j;i<LEN;i+=blockDim.x)
            dst[t*LEN+i]=src[t*LEN+i];
}

static cufftHandle plan_fwd_n,plan_inv_n,plan_fwd_l,plan_inv_l;
static cufftDoubleComplex *d_fa,*d_fb,*d_fc;
static double *d_ftmp;
static uint32_t *d_T,*d_gcur,*d_tmp,*d_xR,*d_q,*d_qn;
static uint32_t *d_n_buf,*d_R_buf,*d_T_new,*d_m_buf;
static uint8_t *d_mbit;
static int g_N=0;

void gpu_mul(uint32_t *a, int alen, uint32_t *b, int blen,
             uint32_t *c, int clen, cufftHandle fwd, cufftHandle inv,
             int fft_n, int N) {
    int total=N*fft_n;
    dim3 grid((fft_n+31)/32,N);
    load_complex<<<grid,32>>>(a,d_fa,N,alen,fft_n);
    load_complex<<<grid,32>>>(b,d_fb,N,blen,fft_n);
    cufftExecZ2Z(fwd,d_fa,d_fa,CUFFT_FORWARD);
    cufftExecZ2Z(fwd,d_fb,d_fb,CUFFT_FORWARD);
    cmul<<<(total+255)/256,256>>>(d_fa,d_fb,d_fc,total);
    cufftExecZ2Z(inv,d_fc,d_fc,CUFFT_INVERSE);
    norm_round<<<(total+255)/256,256>>>(d_fc,d_ftmp,total,1.0/fft_n);
    carry_prop<<<(N+255)/256,256>>>(d_ftmp,c,N,fft_n,clen);
}

void barrett(uint32_t *x, uint32_t *r, int n_bits, int N) {
    gpu_mul(x,2*LEN,d_R_buf,LEN,d_xR,3*LEN,plan_fwd_l,plan_inv_l,FFT_L,N);
    shr_kernel<<<(N+255)/256,256>>>(d_xR,d_q,N,3*LEN,n_bits);
    gpu_mul(d_q,LEN,d_n_buf,LEN,d_qn,2*LEN,plan_fwd_n,plan_inv_n,FFT_N,N);
    sub_correct<<<(N+255)/256,256>>>(x,d_qn,r,d_n_buf,N);
}

extern "C" {

void cufft_init(int N) {
    if(g_N==N) return;
    if(g_N>0) {
        cufftDestroy(plan_fwd_n);cufftDestroy(plan_inv_n);
        cufftDestroy(plan_fwd_l);cufftDestroy(plan_inv_l);
        cudaFree(d_fa);cudaFree(d_fb);cudaFree(d_fc);cudaFree(d_ftmp);
        cudaFree(d_T);cudaFree(d_gcur);cudaFree(d_tmp);cudaFree(d_xR);
        cudaFree(d_q);cudaFree(d_qn);cudaFree(d_n_buf);cudaFree(d_R_buf);
        cudaFree(d_T_new);cudaFree(d_mbit);cudaFree(d_m_buf);
    }
    int ns[1]={FFT_N},nl[1]={FFT_L};
    cufftPlanMany(&plan_fwd_n,1,ns,NULL,1,FFT_N,NULL,1,FFT_N,CUFFT_Z2Z,N);
    cufftPlanMany(&plan_inv_n,1,ns,NULL,1,FFT_N,NULL,1,FFT_N,CUFFT_Z2Z,N);
    cufftPlanMany(&plan_fwd_l,1,nl,NULL,1,FFT_L,NULL,1,FFT_L,CUFFT_Z2Z,N);
    cufftPlanMany(&plan_inv_l,1,nl,NULL,1,FFT_L,NULL,1,FFT_L,CUFFT_Z2Z,N);
    cudaMalloc(&d_fa,  N*FFT_L*sizeof(cufftDoubleComplex));
    cudaMalloc(&d_fb,  N*FFT_L*sizeof(cufftDoubleComplex));
    cudaMalloc(&d_fc,  N*FFT_L*sizeof(cufftDoubleComplex));
    cudaMalloc(&d_ftmp,N*FFT_L*sizeof(double));
    cudaMalloc(&d_T,    N*LEN*4);cudaMalloc(&d_gcur, N*LEN*4);
    cudaMalloc(&d_tmp,  N*2*LEN*4);cudaMalloc(&d_xR,N*3*LEN*4);
    cudaMalloc(&d_q,    N*LEN*4);cudaMalloc(&d_qn,  N*2*LEN*4);
    cudaMalloc(&d_n_buf,N*LEN*4);cudaMalloc(&d_R_buf,N*LEN*4);
    cudaMalloc(&d_T_new,N*LEN*4);cudaMalloc(&d_mbit,N*sizeof(uint8_t));
    cudaMalloc(&d_m_buf,N*LEN*4);
    g_N=N;
}

void cufft_modexp(
    uint32_t *h_g, uint32_t *h_m, uint32_t *h_n, uint32_t *h_R,
    uint32_t *h_out, int N, int m_bits, int n_bits) {

    cufft_init(N);
    size_t sz=(size_t)N*LEN*4;

    uint32_t *h_T=(uint32_t*)calloc(N*LEN,4);
    for(int t=0;t<N;t++) h_T[t*LEN]=1;
    cudaMemcpy(d_T,    h_T,sz,cudaMemcpyHostToDevice); free(h_T);
    cudaMemcpy(d_gcur, h_g,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_buf,h_n,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_buf,h_R,sz,cudaMemcpyHostToDevice);
    // 把m一次性传到GPU
    cudaMemcpy(d_m_buf,h_m,sz,cudaMemcpyHostToDevice);

    for(int i=0;i<m_bits;i++){
        // 在GPU上提取第i位
        extract_bit<<<(N+255)/256,256>>>(d_m_buf,d_mbit,i,N);

        // T_new = T * gcur mod n
        gpu_mul(d_T,LEN,d_gcur,LEN,d_tmp,2*LEN,
                plan_fwd_n,plan_inv_n,FFT_N,N);
        barrett(d_tmp,d_T_new,n_bits,N);
        // 条件更新T
        cond_copy<<<N,32>>>(d_T_new,d_T,d_mbit,N);

        // gcur = gcur^2 mod n
        gpu_mul(d_gcur,LEN,d_gcur,LEN,d_tmp,2*LEN,
                plan_fwd_n,plan_inv_n,FFT_N,N);
        barrett(d_tmp,d_gcur,n_bits,N);
    }

    cudaMemcpy(h_out,d_T,sz,cudaMemcpyDeviceToHost);
}

}
