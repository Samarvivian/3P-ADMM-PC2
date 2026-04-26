#include <stdint.h>
#include <cuda_runtime.h>
#define LEN 128
#define SHARED_U64 384

extern __global__ void modexp_kernel2(
    uint32_t*, uint32_t*, uint32_t*, uint32_t*,
    uint32_t*, int, int, int);

extern "C" {
void run_modexp(
    uint32_t *h_g, uint32_t *h_m,
    uint32_t *h_n, uint32_t *h_R,
    uint32_t *h_out,
    int N, int m_bits, int n_bits)
{
    uint32_t *d_g,*d_m,*d_n,*d_R,*d_out;
    size_t sz=(size_t)N*LEN*4;
    cudaMalloc(&d_g,sz);cudaMalloc(&d_m,sz);
    cudaMalloc(&d_n,sz);cudaMalloc(&d_R,sz);
    cudaMalloc(&d_out,sz);
    cudaMemcpy(d_g,h_g,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_m,h_m,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_n,h_n,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_R,h_R,sz,cudaMemcpyHostToDevice);
    int smem=9*LEN*4+SHARED_U64*8;
    modexp_kernel2<<<N,32,smem>>>(d_g,d_m,d_n,d_R,d_out,m_bits,n_bits,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,sz,cudaMemcpyDeviceToHost);
    cudaFree(d_g);cudaFree(d_m);cudaFree(d_n);
    cudaFree(d_R);cudaFree(d_out);
}
}
