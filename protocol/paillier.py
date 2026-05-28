import gmpy2
import random

def generate_keypair(bits=2048):
    """
    生成Paillier密钥对
    bits: 密钥长度，论文用2048
    返回: (公钥, 私钥)
    """
    print(f"生成 {bits} bits 密钥对，请稍等...")

    # 生成两个大素数p, q
    while True:
        p = gmpy2.mpz(gmpy2.next_prime(random.getrandbits(bits // 2)))
        q = gmpy2.mpz(gmpy2.next_prime(random.getrandbits(bits // 2)))
        if p != q:
            break

    n = p * q
    n2 = n * n

    # lambda = lcm(p-1, q-1)
    lam = gmpy2.lcm(p - 1, q - 1)

    # 选g = n+1（简化版，安全性等价）
    g = n + 1

    # mu = (L(g^lambda mod n^2))^{-1} mod n
    def L(x):
        return (x - 1) // n

    g_lam = gmpy2.powmod(g, lam, n2)
    mu = gmpy2.invert(L(g_lam), n)

    public_key = (n, g)
    private_key = (lam, mu, n, n2)

    print("密钥生成完成")
    return public_key, private_key


def encrypt(m, public_key):
    """
    加密单个非负整数m
    """
    n, g = public_key
    n2 = n * n

    m = gmpy2.mpz(m)
    assert 0 <= m < n, f"明文必须在[0, n)范围内，当前m={m}"

    # 随机数r，gcd(r, n) = 1
    while True:
        r = gmpy2.mpz(random.getrandbits(n.bit_length()))
        if r > 0 and gmpy2.gcd(r, n) == 1:
            break

    # c = g^m * r^n mod n^2
    c = gmpy2.powmod(g, m, n2) * gmpy2.powmod(r, n, n2) % n2
    return c


def decrypt(c, private_key):
    """
    解密密文c
    """
    lam, mu, n, n2 = private_key
    c = gmpy2.mpz(c)

    def L(x):
        return (x - 1) // n

    # m = L(c^lambda mod n^2) * mu mod n
    m = L(gmpy2.powmod(c, lam, n2)) * mu % n
    return int(m)


def homo_add(c1, c2, public_key):
    """
    同态加法：fen(m1) * fen(m2) mod n^2 = fen(m1+m2)
    对应论文 Definition 1
    """
    n, g = public_key
    n2 = n * n
    return c1 * c2 % n2


def homo_mul_const(c, k, public_key):
    """
    同态常数乘：fen(m)^k mod n^2 = fen(k*m)
    对应论文 Definition 2
    """
    n, g = public_key
    n2 = n * n
    return gmpy2.powmod(c, gmpy2.mpz(k), n2)
