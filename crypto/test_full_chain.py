import sys
import numpy as np
sys.path.append('/mnt/3p-admm-pc2')
from crypto.paillier import generate_keypair, encrypt, decrypt, homo_add, homo_mul_const
from crypto.quantization import gamma1, gamma2

pub, priv = generate_keypair(bits=1024)
n, g = pub
delta = 10**10

np.random.seed(42)
z = np.random.randn(5)
v = np.random.randn(5)
B_bar = np.random.rand(5)
alpha = np.random.randn(5)

# 统一用同一个zmin zmax（论文中所有向量用同一量化范围）
all_vals = np.concatenate([z, -v, B_bar, alpha])
zmin = all_vals.min()
zmax = all_vals.max()

print(f"zmin={zmin:.4f}, zmax={zmax:.4f}")

# 量化（统一范围）
def q2(v_arr):
    return np.floor(delta * (v_arr - zmin) / (zmax - zmin)).astype(object)

def q1(v_arr):
    return np.floor(delta**2 * (v_arr - zmin) / (zmax - zmin)**2).astype(object)

q_z = q2(z)
q_v = q2(-v)
q_B = q2(B_bar)
q_alpha = q1(alpha)

# 加密
c_z     = [encrypt(int(qi), pub) for qi in q_z]
c_v     = [encrypt(int(qi), pub) for qi in q_v]
c_alpha = [encrypt(int(qi), pub) for qi in q_alpha]

# 同态计算
results = []
for i in range(5):
    c_sum   = homo_add(c_z[i], c_v[i], pub)
    c_mul   = homo_mul_const(c_sum, int(q_B[i]), pub)
    c_final = homo_add(c_alpha[i], c_mul, pub)
    results.append(c_final)

# 解密
decrypted = np.array([decrypt(c, priv) for c in results], dtype=float)

# 逆量化，严格按论文公式(30)
scale = (zmax - zmin)**2 / delta**2
correction = (2 * B_bar + (z - v) + 1) * zmin - 2 * zmin**2
recovered = decrypted * scale + correction

# 真实值
true_result = alpha + B_bar * (z + (-v))

print(f"逆量化结果: {recovered}")
print(f"真实结果:   {true_result}")
error = np.max(np.abs(recovered - true_result))
print(f"最大误差: {error:.2e}")
print(f"{'✓ 通过' if error < 1e-5 else '✗ 误差偏大'}")
