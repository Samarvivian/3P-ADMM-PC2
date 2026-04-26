import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import gmpy2, random, time

BASE = 65536
MAX_LEN = 256

def int_to_array(n, length=MAX_LEN):
    arr = np.zeros(length, dtype=np.uint32)
    n = int(n)
    for i in range(length):
        arr[i] = n % BASE
        n //= BASE
    return arr

def array_to_int(arr):
    result = 0
    for i in range(len(arr)-1, -1, -1):
        result = result * BASE + int(arr[i])
    return result

def compute_R(n):
    k = int(n).bit_length()
    R = (1 << (2*k)) // int(n)
    return int_to_array(R), k

def load_kernel():
    with open('/mnt/3p-admm-pc2/gpu/modexp.cu', 'r') as f:
        code = f.read()
    mod = SourceModule(code)
    return mod.get_function("modexp_kernel")

def gpu_modexp_batch(g_list, m_list, n_val, m_bits=None):
    kernel = load_kernel()
    num_tasks = len(g_list)
    if m_bits is None:
        m_bits = max(int(x).bit_length() for x in m_list)

    g_all  = np.zeros((num_tasks, MAX_LEN), dtype=np.uint32)
    m_all  = np.zeros((num_tasks, MAX_LEN), dtype=np.uint32)
    n_all  = np.zeros((num_tasks, MAX_LEN), dtype=np.uint32)
    R_all  = np.zeros((num_tasks, MAX_LEN), dtype=np.uint32)
    out_all= np.zeros((num_tasks, MAX_LEN), dtype=np.uint32)

    R_arr, n_bits = compute_R(n_val)

    for i in range(num_tasks):
        g_all[i] = int_to_array(g_list[i])
        m_all[i] = int_to_array(m_list[i])
        n_all[i] = int_to_array(n_val)
        R_all[i] = R_arr

    g_gpu  = cuda.to_device(g_all.flatten())
    m_gpu  = cuda.to_device(m_all.flatten())
    n_gpu  = cuda.to_device(n_all.flatten())
    R_gpu  = cuda.to_device(R_all.flatten())
    out_gpu= cuda.mem_alloc(out_all.nbytes)

    block = 32
    grid  = (num_tasks + block - 1) // block

    kernel(
        g_gpu, m_gpu, n_gpu, R_gpu, out_gpu,
        np.int32(MAX_LEN), np.int32(m_bits),
        np.int32(n_bits), np.int32(num_tasks),
        block=(block,1,1), grid=(grid,1)
    )
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(out_all, out_gpu)

    return [array_to_int(out_all[i]) % int(n_val) for i in range(num_tasks)]

if __name__ == '__main__':
    print("测试GPU ModExp...")
    n = int(gmpy2.next_prime(random.getrandbits(64))) * \
        int(gmpy2.next_prime(random.getrandbits(64)))
    g = random.randint(2, 1000)
    m = random.randint(2, 1000)

    cpu = int(gmpy2.powmod(g, m, n))
    gpu = gpu_modexp_batch([g], [m], n, m_bits=m.bit_length())[0]

    print(f'CPU: {cpu}')
    print(f'GPU: {gpu}')
    print(f'{"✓ 正确" if cpu==gpu else "✗ 错误"}')

def gpu_modexp_batch_v2(g_list, m_list, n_val, m_bits=None):
    """使用shared memory优化版本"""
    with open('/mnt/3p-admm-pc2/gpu/modexp_v2.cu', 'r') as f:
        code = f.read()
    mod = SourceModule(code)
    kernel = mod.get_function("modexp_kernel_v2")

    num_tasks = len(g_list)
    LEN = 128
    if m_bits is None:
        m_bits = max(int(x).bit_length() for x in m_list)

    g_all  = np.zeros((num_tasks, LEN), dtype=np.uint32)
    m_all  = np.zeros((num_tasks, LEN), dtype=np.uint32)
    n_all  = np.zeros((num_tasks, LEN), dtype=np.uint32)
    R_all  = np.zeros((num_tasks, LEN), dtype=np.uint32)
    out_all= np.zeros((num_tasks, LEN), dtype=np.uint32)

    R_arr, n_bits = compute_R(n_val)
    R_arr = R_arr[:LEN]

    for i in range(num_tasks):
        g_all[i] = int_to_array(g_list[i], LEN)
        m_all[i] = int_to_array(m_list[i], LEN)
        n_all[i] = int_to_array(n_val, LEN)
        R_all[i] = R_arr

    g_gpu  = cuda.to_device(g_all.flatten())
    m_gpu  = cuda.to_device(m_all.flatten())
    n_gpu  = cuda.to_device(n_all.flatten())
    R_gpu  = cuda.to_device(R_all.flatten())
    out_gpu= cuda.mem_alloc(out_all.nbytes)

    # 每个block处理一个任务，LEN个线程
    smem_size = 10 * LEN * 4  # 10*LEN个uint32
    kernel(
        g_gpu, m_gpu, n_gpu, R_gpu, out_gpu,
        np.int32(LEN), np.int32(m_bits),
        np.int32(n_bits), np.int32(num_tasks),
        block=(LEN, 1, 1), grid=(num_tasks, 1),
        shared=smem_size
    )
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(out_all, out_gpu)

    return [array_to_int(out_all[i]) % int(n_val) for i in range(num_tasks)]

if __name__ == '__main__':
    print("测试V2 GPU ModExp...")
    n = int(gmpy2.next_prime(random.getrandbits(64))) * \
        int(gmpy2.next_prime(random.getrandbits(64)))
    g = random.randint(2, 1000)
    m = random.randint(2, 1000)

    cpu = int(gmpy2.powmod(g, m, n))
    gpu = gpu_modexp_batch_v2([g], [m], n, m_bits=m.bit_length())[0]

    print(f'CPU: {cpu}')
    print(f'GPU: {gpu}')
    print(f'{"✓ 正确" if cpu==gpu else "✗ 错误"}')
