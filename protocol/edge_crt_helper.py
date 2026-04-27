"""
边缘节点CRT辅助：计算 r^(n mod phi(p²)) mod p²
由主节点通过SSH调用
"""
import sys
import pickle
import numpy as np
import gmpy2
import ctypes

sys.path.append('/mnt/3p-admm-pc2')

LEN = 128
BASE = 65536

def to_arr(n, l=LEN):
    n = int(n)
    nbytes = max(l*2, (n.bit_length()+15)//16*2)
    b = n.to_bytes(nbytes, 'little')
    arr = np.frombuffer(b, dtype=np.uint16).astype(np.uint32)
    result = np.zeros(l, dtype=np.uint32)
    result[:min(len(arr),l)] = arr[:min(len(arr),l)]
    return result.copy()

def compute_R(n):
    n = int(n)
    k = n.bit_length()
    R = int(gmpy2.mpz(1) << (2*k)) // n
    return to_arr(R), k

def ptr(a):
    return np.ascontiguousarray(a).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

def gpu_modexp_diff_g(g_list, m_val, n_val, m_bits):
    lib = ctypes.CDLL('/tmp/lib_cufft.so')
    lib.run_modexp.argtypes = [ctypes.POINTER(ctypes.c_uint32)]*5+[ctypes.c_int]*3
    lib.run_modexp.restype = None
    lib.init_gpu.argtypes = [ctypes.c_int]
    lib.init_gpu.restype = None

    N = len(g_list)
    R_arr, n_bits = compute_R(n_val)
    lib.init_gpu(ctypes.c_int(N))

    g_all = np.zeros((N,LEN), dtype=np.uint32)
    for i,g in enumerate(g_list):
        b = int(g).to_bytes(LEN*2,'little')
        g_all[i] = np.frombuffer(b,dtype=np.uint16).astype(np.uint32).copy()
    g_all = np.ascontiguousarray(g_all)
    m_all = np.ascontiguousarray(np.tile(to_arr(m_val),(N,1)))
    n_all = np.ascontiguousarray(np.tile(to_arr(n_val),(N,1)))
    R_all = np.ascontiguousarray(np.tile(R_arr,(N,1)))
    out_all = np.ascontiguousarray(np.zeros((N,LEN),dtype=np.uint32))

    lib.run_modexp(ptr(g_all),ptr(m_all),ptr(n_all),ptr(R_all),ptr(out_all),
                   ctypes.c_int(N),ctypes.c_int(m_bits),ctypes.c_int(int(n_bits)))
    n_int = int(n_val)
    out_u16 = out_all.astype(np.uint16)
    return [int.from_bytes(out_u16[i].tobytes(),'little')%n_int for i in range(N)]

# 读取任务
task_file = sys.argv[1]
result_file = sys.argv[2]

with open(task_file, 'rb') as f:
    task = pickle.load(f)

r_list = task['r_list']      # r mod p²
n_mod_phi_p2 = task['n_mod_phi_p2']  # n mod phi(p²)
p2 = task['p2']              # p²
m_bits = task['m_bits']      # 指数bits

print(f'边缘节点CRT: 计算{len(r_list)}个r^n mod p²...')
import time
t0 = time.time()
results = gpu_modexp_diff_g(r_list, n_mod_phi_p2, p2, m_bits)
print(f'完成: {time.time()-t0:.2f}s')

with open(result_file, 'wb') as f:
    pickle.dump({'rn_p2': results}, f)
print('结果已保存')
