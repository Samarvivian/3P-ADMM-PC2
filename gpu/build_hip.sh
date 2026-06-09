#!/usr/bin/env bash
# ROCm/HIP build of the live cuFFT modexp kernel, mirroring the nvcc recipe in
# README.md sec.1 but producing the same /tmp/lib_cufft.so with the identical
# init_gpu / run_modexp C ABI, so the Python ctypes layer is untouched.
#
# Arch comes from HIP_ARCH (never a literal), so other AMD GPU targets reuse
# this script with only an env change. A semicolon/comma list builds fat code:
#   HIP_ARCH="gfx90a,gfx1100" ./build_hip.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIP_ARCH="${HIP_ARCH:-gfx90a}"
OUT="${OUT:-/tmp/lib_cufft.so}"
BUILD="${BUILD:-/tmp}"

OFFLOAD=""
IFS=',;' read -ra ARCHES <<< "$HIP_ARCH"
for a in "${ARCHES[@]}"; do OFFLOAD+=" --offload-arch=$a"; done

# Step 1: main kernel
hipcc -x hip -O2 $OFFLOAD -fPIC -c "$HERE/cufft_modexp.cu" -o "$BUILD/cufft_modexp.o"

# Step 2: C ABI wrapper (same names the Python loader expects)
cat > "$BUILD/wr_cufft.cu" << 'EOF'
#include <stdint.h>
extern "C" {
    void cufft_init(int N);
    void cufft_modexp(uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t*, int, int, int);
}
extern "C" {
void init_gpu(int N){ cufft_init(N); }
void run_modexp(uint32_t *hg, uint32_t *hm, uint32_t *hn, uint32_t *hR,
                uint32_t *ho, int N, int mb, int nb){
    cufft_modexp(hg,hm,hn,hR,ho,N,mb,nb);
}
}
EOF
hipcc -x hip -O2 $OFFLOAD -fPIC -c "$BUILD/wr_cufft.cu" -o "$BUILD/wr_cufft.o"

# Step 3: shared lib (hipcc does host+device link in one step for a non-RDC build)
hipcc -shared -fPIC "$BUILD/cufft_modexp.o" "$BUILD/wr_cufft.o" -lhipfft -o "$OUT"

ls -lh "$OUT"
