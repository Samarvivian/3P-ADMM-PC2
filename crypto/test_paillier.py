import sys
sys.path.append('/mnt/3p-admm-pc2')
from crypto.paillier import generate_keypair, encrypt, decrypt, homo_add, homo_mul_const

# 用1024bits测试，速度快
pub, priv = generate_keypair(bits=1024)

# 测试基本加解密
m1, m2 = 12345, 67890
c1 = encrypt(m1, pub)
c2 = encrypt(m2, pub)

print(f"明文 m1={m1}, m2={m2}")
print(f"解密 c1={decrypt(c1, priv)}")
print(f"解密 c2={decrypt(c2, priv)}")

# 测试同态加法
c_add = homo_add(c1, c2, pub)
print(f"\n同态加法: {m1}+{m2}={m1+m2}")
print(f"解密结果: {decrypt(c_add, priv)}")
print(f"验证: {'✓ 通过' if decrypt(c_add, priv) == m1+m2 else '✗ 失败'}")

# 测试同态常数乘
k = 5
c_mul = homo_mul_const(c1, k, pub)
print(f"\n同态常数乘: {k}*{m1}={k*m1}")
print(f"解密结果: {decrypt(c_mul, priv)}")
print(f"验证: {'✓ 通过' if decrypt(c_mul, priv) == k*m1 else '✗ 失败'}")
