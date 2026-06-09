// CUDA -> HIP compatibility shim for the live cuFFT modexp kernel.
//
// On AMD/HIP this maps the exact CUDA runtime + cuFFT symbols used by
// cufft_modexp.cu onto their hipFFT/HIP spellings, so the .cu body keeps
// its CUDA spelling and the NVIDIA build stays byte-identical. On NVIDIA
// it just pulls in the original CUDA headers.
//
// libc headers (cstdlib/cstring/cmath) are included BEFORE the HIP runtime
// on purpose: inside a .cu compiled as HIP, host calloc/free/round can
// otherwise bind to HIP __device__ overloads. cufft_modexp.cu calls
// calloc/free on the host and round() in a __global__.
#ifndef CUDA_TO_HIP_H
#define CUDA_TO_HIP_H

#include <cstdlib>
#include <cstring>
#include <cmath>

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

// runtime
#define cudaMalloc            hipMalloc
#define cudaFree              hipFree
#define cudaMemcpy            hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

// cuFFT -> hipFFT
typedef hipfftHandle        cufftHandle;
typedef hipfftDoubleComplex cufftDoubleComplex;
#define cufftPlanMany   hipfftPlanMany
#define cufftExecZ2Z    hipfftExecZ2Z
#define cufftDestroy    hipfftDestroy
#define CUFFT_Z2Z       HIPFFT_Z2Z
#define CUFFT_FORWARD   HIPFFT_FORWARD
#define CUFFT_INVERSE   HIPFFT_BACKWARD

#else

#include <cuda_runtime.h>
#include <cufft.h>

#endif

#endif // CUDA_TO_HIP_H
