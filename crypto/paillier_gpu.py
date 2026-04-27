import numpy as np
import gmpy2
import ctypes
import random

# 加载cuFFT版本的GPU库
_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL('/tmp/lib_cufft.so')
        _lib.run_modexp.argtypes = [ctypes.POINTER(ctypes.c_uint32)]*5 + [ctypes.c_int]*3
        _lib.run_modexp.restype = None
        _lib.init_gpu.argtypes = [ctypes.c_int]
        _lib.init_gpu.restype = None
    return _lib

BASE = 65536
LEN = 128

def _to_arr(n, l=LEN):
    n = int(n)
    # 动态计算所需字节数，至少l*2字节
    nbytes = max(l*2, (n.bit_length() + 15) // 16 * 2)
    b = n.to_bytes(nbytes, 'little')
    arr = np.frombuffer(b, dtype=np.uint16).astype(np.uint32)
    # 截断或填充到l个元素
    result = np.zeros(l, dtype=np.uint32)
    result[:min(len(arr),l)] = arr[:min(len(arr),l)]
    return result.copy()

def _compute_R(n):
    n = int(n)
    k = n.bit_length()
    R = int(gmpy2.mpz(1) << (2*k)) // n
    return _to_arr(R), k

def _ptr(a):
    return np.ascontiguousarray(a).ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

def _to_int(arr):
    return int.from_bytes(arr.astype(np.uint16).tobytes(), 'little')

def gpu_batch_modexp(g_list, m_list, n_val, m_bits):
    """批量计算g_list[i]^m_list[i] mod n_val"""
    lib = _get_lib()
    N = len(g_list)
    R_arr, n_bits = _compute_R(n_val)

    lib.init_gpu(ctypes.c_int(N))

    g_all  = np.zeros((N,LEN), dtype=np.uint32)
    m_all  = np.zeros((N,LEN), dtype=np.uint32)
    n_all  = np.ascontiguousarray(np.tile(_to_arr(n_val),(N,1)))
    R_all  = np.ascontiguousarray(np.tile(R_arr,(N,1)))
    out_all= np.ascontiguousarray(np.zeros((N,LEN), dtype=np.uint32))

    for i,g in enumerate(g_list):
        b = int(g).to_bytes(LEN*2,'little')
        g_all[i] = np.frombuffer(b,dtype=np.uint16).astype(np.uint32)
    for i,m in enumerate(m_list):
        b = int(m).to_bytes(LEN*2,'little')
        m_all[i] = np.frombuffer(b,dtype=np.uint16).astype(np.uint32)

    g_all = np.ascontiguousarray(g_all)
    m_all = np.ascontiguousarray(m_all)

    lib.run_modexp(
        _ptr(g_all), _ptr(m_all), _ptr(n_all), _ptr(R_all), _ptr(out_all),
        ctypes.c_int(N), ctypes.c_int(m_bits), ctypes.c_int(int(n_bits))
    )

    n_int = int(n_val)
    out_u16 = out_all.astype(np.uint16)
    return [int.from_bytes(out_u16[i].tobytes(),'little') % n_int for i in range(N)]

def encrypt_batch_gpu(messages, public_key):
    """
    批量GPU加密
    messages: 整数列表
    返回: 密文列表
    """
    n, g = public_key
    n2 = int(n * n)
    N = len(messages)

    # 生成随机数r
    rs = []
    while len(rs) < N:
        r = int(gmpy2.mpz(random.getrandbits(int(n).bit_length())))
        if r > 0 and gmpy2.gcd(r, n) == 1:
            rs.append(r)

    messages = [int(m) for m in messages]
    m_bits = max(x.bit_length() for x in messages) if messages else 1
    n_bits_exp = int(n).bit_length()

    # GPU计算g^m mod n²（小指数，快）
    print(f'  GPU计算g^m ({N}个，{m_bits}bits指数)...')
    gm_list = gpu_batch_modexp([int(g)]*N, messages, n2, m_bits)

    # GPU计算r^n mod n²（大指数，慢）
    # 用CRT加速：r^n mod p² 和 r^n mod q²
    # 暂时用CPU，后续可以集成CRT+GPU
    print(f'  CPU计算r^n ({N}个，{n_bits_exp}bits指数)...')
    rn_list = [int(gmpy2.powmod(r, n, n2)) for r in rs]

    # 合并
    ciphertexts = [gm * rn % n2 for gm, rn in zip(gm_list, rn_list)]
    return ciphertexts

def encrypt_batch_gpu_crt(messages, public_key, p, q, nodes):
    """
    分布式CRT批量GPU加密
    主节点计算g^m和r^n mod q²
    边缘节点计算r^n mod p²
    """
    import subprocess, pickle, tempfile, os, time
    n, g = public_key
    n2 = int(n*n)
    p2 = int(p*p)
    q2 = int(q*q)
    N = len(messages)

    # CRT预计算
    inv_p2_q2 = int(gmpy2.invert(p2, q2))
    phi_p2 = int(p2 - p2//p)
    phi_q2 = int(q2 - q2//q)
    n_mod_p = int(n % phi_p2)
    n_mod_q = int(n % phi_q2)
    m_bits_n_p = n_mod_p.bit_length()
    m_bits_n_q = n_mod_q.bit_length()

    # 生成随机数r
    rs = []
    while len(rs) < N:
        r = int(gmpy2.mpz(random.getrandbits(int(n).bit_length())))
        if r > 0 and gmpy2.gcd(r, n) == 1:
            rs.append(r)

    messages = [int(m) for m in messages]
    m_bits_msg = max(x.bit_length() for x in messages) if messages else 1

    lib = _get_lib()
    lib.init_gpu(ctypes.c_int(N))

    # 1. 主节点GPU: g^m mod n²
    t0 = time.time()
    print(f'  主节点GPU: g^m mod n²...')
    R_arr, n_bits_n2 = _compute_R(n2)
    g_all = np.ascontiguousarray(np.tile(_to_arr(int(g)),(N,1)))
    m_all = np.zeros((N,LEN), dtype=np.uint32)
    if m_bits_msg <= 50:
        mu64 = np.array(messages, dtype=np.uint64)
        for j in range(4): m_all[:,j]=(mu64>>np.uint64(16*j))&np.uint64(0xFFFF)
    else:
        for i,m in enumerate(messages):
            b=m.to_bytes(LEN*2,'little')
            m_all[i]=np.frombuffer(b,dtype=np.uint16).astype(np.uint32).copy()
    m_all = np.ascontiguousarray(m_all)
    n_all = np.ascontiguousarray(np.tile(_to_arr(n2),(N,1)))
    R_all = np.ascontiguousarray(np.tile(R_arr,(N,1)))
    out_all = np.ascontiguousarray(np.zeros((N,LEN),dtype=np.uint32))
    lib.run_modexp(_ptr(g_all),_ptr(m_all),_ptr(n_all),_ptr(R_all),_ptr(out_all),
                   ctypes.c_int(N),ctypes.c_int(m_bits_msg),ctypes.c_int(int(n_bits_n2)))
    out_u16 = out_all.astype(np.uint16)
    gm_list = [int.from_bytes(out_u16[i].tobytes(),'little')%n2 for i in range(N)]
    print(f'  g^m完成: {time.time()-t0:.2f}s')

    # 2. 发送任务给边缘节点（r^n mod p²）
    rp_list = [int(r%p2) for r in rs]
    task = {
        'r_list': rp_list,
        'n_mod_phi_p2': n_mod_p,
        'p2': p2,
        'm_bits': m_bits_n_p,
    }
    node = nodes[0]  # 用第一个边缘节点
    task_file = '/mnt/crt_task.pkl'
    result_file = '/mnt/crt_result.pkl'
    local_task = '/tmp/crt_task.pkl'
    local_result = '/tmp/crt_result.pkl'

    with open(local_task, 'wb') as f:
        pickle.dump(task, f)

    # 3. 主节点GPU: r^n mod q²（和边缘节点并行）
    print(f'  发送任务到边缘节点并行计算...')
    # 先发任务
    subprocess.run(['scp','-P',str(node['port']),
                    local_task,f"root@{node['host']}:{task_file}"],check=True)
    # 启动边缘节点计算（异步）
    proc = subprocess.Popen([
        'ssh','-p',str(node['port']),f"root@{node['host']}",
        f"/root/miniconda3/envs/myconda/bin/python3 "
        f"/mnt/3p-admm-pc2/protocol/edge_crt_helper.py "
        f"{task_file} {result_file}"
    ])

    # 同时主节点计算r^n mod q²
    t0 = time.time()
    rq_list = [int(r%q2) for r in rs]
    R_arr_q, n_bits_q2 = _compute_R(q2)
    g_all2 = np.zeros((N,LEN), dtype=np.uint32)
    for i,rq in enumerate(rq_list):
        b=int(rq).to_bytes(LEN*2,'little')
        g_all2[i]=np.frombuffer(b,dtype=np.uint16).astype(np.uint32).copy()
    g_all2 = np.ascontiguousarray(g_all2)
    m_all2 = np.ascontiguousarray(np.tile(_to_arr(n_mod_q),(N,1)))
    n_all2 = np.ascontiguousarray(np.tile(_to_arr(q2),(N,1)))
    R_all2 = np.ascontiguousarray(np.tile(R_arr_q,(N,1)))
    out_all2 = np.ascontiguousarray(np.zeros((N,LEN),dtype=np.uint32))
    lib.run_modexp(_ptr(g_all2),_ptr(m_all2),_ptr(n_all2),_ptr(R_all2),_ptr(out_all2),
                   ctypes.c_int(N),ctypes.c_int(m_bits_n_q),ctypes.c_int(int(n_bits_q2)))
    out_u16_2 = out_all2.astype(np.uint16)
    rn_q = [int.from_bytes(out_u16_2[i].tobytes(),'little')%q2 for i in range(N)]
    print(f'  主节点r^n mod q²完成: {time.time()-t0:.2f}s')

    # 4. 等待边缘节点完成
    proc.wait()
    subprocess.run(['scp','-P',str(node['port']),
                    f"root@{node['host']}:{result_file}",local_result],check=True)
    with open(local_result,'rb') as f:
        result = pickle.load(f)
    rn_p = result['rn_p2']
    print(f'  边缘节点r^n mod p²收到')

    # 5. CRT合并
    t0 = time.time()
    rn_list = [(rp+(rq-rp)*inv_p2_q2%q2*p2)%n2
               for rp,rq in zip(rn_p,rn_q)]
    ciphers = [gm*rn%n2 for gm,rn in zip(gm_list,rn_list)]
    print(f'  CRT合并: {time.time()-t0:.2f}s')

    return ciphers


# 预计算的r^n缓存
_rn_cache = {}

def precompute_rn(public_key, N, cache_key=None):
    """预计算N个r^n mod n²，缓存复用"""
    global _rn_cache
    if cache_key and cache_key in _rn_cache:
        print('  使用缓存的r^n')
        return _rn_cache[cache_key]
    
    n, g = public_key
    n2 = int(n*n)
    print(f'  预计算{N}个r^n...')
    import time
    t0=time.time()
    rs = []
    while len(rs) < N:
        r = int(gmpy2.mpz(random.getrandbits(int(n).bit_length())))
        if r > 0 and gmpy2.gcd(r, n) == 1:
            rs.append(r)
    rn_list = [int(gmpy2.powmod(r, n, n2)) for r in rs]
    print(f'  预计算完成: {time.time()-t0:.2f}s')
    
    if cache_key:
        _rn_cache[cache_key] = rn_list
    return rn_list

def encrypt_batch_gpu_fast(messages, public_key, rn_precomputed=None):
    """
    快速GPU批量加密（使用预计算的r^n）
    rn_precomputed: 预计算的r^n列表，None时实时计算
    """
    import time
    n, g = public_key
    n2 = int(n*n)
    N = len(messages)
    messages = [int(m) for m in messages]
    m_bits = max(x.bit_length() for x in messages) if messages else 1
    if m_bits == 0: m_bits = 1

    lib = _get_lib()
    lib.init_gpu(ctypes.c_int(N))
    R_arr, n_bits = _compute_R(n2)

    # GPU: g^m
    g_all = np.ascontiguousarray(np.tile(_to_arr(int(g)),(N,1)))
    n_all = np.ascontiguousarray(np.tile(_to_arr(n2),(N,1)))
    R_all = np.ascontiguousarray(np.tile(R_arr,(N,1)))
    out_all = np.ascontiguousarray(np.zeros((N,LEN),dtype=np.uint32))
    m_all = np.zeros((N,LEN), dtype=np.uint32)
    if m_bits <= 50:
        mu64 = np.array(messages, dtype=np.uint64)
        for j in range(4): m_all[:,j]=(mu64>>np.uint64(16*j))&np.uint64(0xFFFF)
    else:
        for i,m in enumerate(messages):
            b=m.to_bytes(LEN*2,'little')
            m_all[i]=np.frombuffer(b,dtype=np.uint16).astype(np.uint32).copy()
    m_all = np.ascontiguousarray(m_all)

    lib.run_modexp(_ptr(g_all),_ptr(m_all),_ptr(n_all),_ptr(R_all),_ptr(out_all),
                   ctypes.c_int(N),ctypes.c_int(m_bits),ctypes.c_int(int(n_bits)))
    out_u16 = out_all.astype(np.uint16)
    gm_list = [int.from_bytes(out_u16[i].tobytes(),'little')%n2 for i in range(N)]

    # r^n
    if rn_precomputed is None:
        rn_precomputed = precompute_rn(public_key, N)

    # 合并
    return [gm*rn%n2 for gm,rn in zip(gm_list,rn_precomputed)]
