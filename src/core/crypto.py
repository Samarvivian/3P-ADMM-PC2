"""
Paillier同态加密实现

提供Paillier公钥加密系统，支持加法同态和标量乘法同态运算。

Paillier加密系统特性：
- 加法同态: E(m1) ⊕ E(m2) = E(m1 + m2)
- 标量乘法同态: k ⊗ E(m) = E(k·m)

参考文献:
    Paillier, P. (1999). "Public-Key Cryptosystems Based on Composite Degree
    Residuosity Classes". EUROCRYPT 1999.
"""

import secrets
from math import gcd
from typing import Tuple


def is_probable_prime(n: int, k: int = 8) -> bool:
    """
    Miller-Rabin素性测试

    使用Miller-Rabin算法判断一个数是否可能是素数。

    Args:
        n: 待测试的数
        k: 测试轮数（越大越准确，默认8轮）

    Returns:
        True如果n可能是素数，False如果n确定是合数

    Examples:
        >>> is_probable_prime(17)
        True
        >>> is_probable_prime(16)
        False
    """
    if n < 2:
        return False

    # 检查小素数
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n % p == 0:
            return n == p

    # Miller-Rabin测试
    # 将n-1写成2^s * d的形式
    d, s = n - 1, 0
    while d % 2 == 0:
        s += 1
        d //= 2

    def trial(a: int) -> bool:
        """单次Miller-Rabin测试"""
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    # 进行k轮测试
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        if not trial(a):
            return False

    return True


def gen_prime(bits: int) -> int:
    """
    生成指定位数的素数

    使用密码学安全的随机数生成器和Miller-Rabin素性测试。

    Args:
        bits: 素数的位数

    Returns:
        一个bits位的素数

    Raises:
        ValueError: 如果bits < 2

    Examples:
        >>> p = gen_prime(512)
        >>> p.bit_length()
        512
        >>> is_probable_prime(p)
        True
    """
    if bits < 2:
        raise ValueError(f"bits必须 >= 2，实际值：{bits}")

    while True:
        # 生成随机奇数，最高位和最低位都是1
        p = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if is_probable_prime(p):
            return p


def lcm(a: int, b: int) -> int:
    """
    计算最小公倍数

    Args:
        a: 第一个整数
        b: 第二个整数

    Returns:
        a和b的最小公倍数
    """
    return a * b // gcd(a, b)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    扩展欧几里得算法

    计算gcd(a,b)以及满足ax + by = gcd(a,b)的x和y。

    Args:
        a: 第一个整数
        b: 第二个整数

    Returns:
        (g, x, y)，其中g = gcd(a,b)，且ax + by = g
    """
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def modinv(a: int, m: int) -> int:
    """
    计算模逆

    计算a在模m下的乘法逆元，即满足(a * x) % m = 1的x。

    Args:
        a: 待求逆的数
        m: 模数

    Returns:
        a在模m下的逆元

    Raises:
        ValueError: 如果逆元不存在（gcd(a,m) != 1）

    Examples:
        >>> modinv(3, 11)
        4
        >>> (3 * 4) % 11
        1
    """
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"模逆不存在：gcd({a}, {m}) = {g} != 1")
    return x % m


class PaillierKeypair:
    """
    Paillier公钥加密系统

    实现Paillier加密系统，支持加法同态和标量乘法同态运算。

    加密方案：
        - 公钥: (n, g)，其中n = p*q，g = n+1
        - 私钥: (λ, μ)，其中λ = lcm(p-1, q-1)
        - 加密: c = g^m * r^n mod n²
        - 解密: m = L(c^λ mod n²) * μ mod n，其中L(x) = (x-1)/n

    同态性质：
        - E(m1) * E(m2) mod n² = E(m1 + m2 mod n)
        - E(m)^k mod n² = E(k*m mod n)

    Attributes:
        p, q: 两个大素数（私钥）
        n: 模数 n = p*q（公钥）
        n2: n的平方
        g: 生成元 g = n+1（公钥）
        lam: λ = lcm(p-1, q-1)（私钥）
        mu: μ = L(g^λ mod n²)^(-1) mod n（私钥）

    Examples:
        >>> # 生成密钥对
        >>> keypair = PaillierKeypair(bits=512)
        >>>
        >>> # 加密
        >>> m1, m2 = 15, 25
        >>> c1 = keypair.encrypt(m1)
        >>> c2 = keypair.encrypt(m2)
        >>>
        >>> # 同态加法
        >>> c_sum = keypair.e_add(c1, c2)
        >>> m_sum = keypair.decrypt(c_sum)
        >>> assert m_sum == (m1 + m2) % keypair.n
        >>>
        >>> # 同态标量乘法
        >>> k = 3
        >>> c_mul = keypair.e_mul_const(c1, k)
        >>> m_mul = keypair.decrypt(c_mul)
        >>> assert m_mul == (k * m1) % keypair.n
    """

    def __init__(self, bits: int = 512):
        """
        生成Paillier密钥对

        Args:
            bits: 模数n的位数（默认512位）
                 注意：生产环境建议使用2048位或更高

        Raises:
            ValueError: 如果bits < 8
        """
        if bits < 8:
            raise ValueError(f"bits必须 >= 8，实际值：{bits}")

        # 生成两个素数
        self.p = gen_prime(bits // 2)
        self.q = gen_prime(bits // 2)

        # 计算公钥参数
        self.n = self.p * self.q
        self.n2 = self.n**2
        self.g = self.n + 1  # 简化形式，g = n+1

        # 计算私钥参数
        self.lam = lcm(self.p - 1, self.q - 1)

        # 计算μ = L(g^λ mod n²)^(-1) mod n
        x = pow(self.g, self.lam, self.n2)
        Lx = (x - 1) // self.n
        self.mu = modinv(Lx % self.n, self.n)

    def encrypt(self, m: int) -> int:
        """
        加密明文

        Args:
            m: 明文（整数，0 <= m < n）

        Returns:
            密文c

        Raises:
            ValueError: 如果明文超出范围

        Note:
            加密使用随机数r，因此同一明文每次加密结果不同（语义安全）
        """
        m = int(m) % self.n

        if m < 0 or m >= self.n:
            raise ValueError(f"明文超出范围：0 <= m < {self.n}")

        # 生成随机数r ∈ Z*_n
        r = secrets.randbelow(self.n - 1) + 1

        # c = g^m * r^n mod n²
        return (pow(self.g, m, self.n2) * pow(r, self.n, self.n2)) % self.n2

    def decrypt(self, c: int) -> int:
        """
        解密密文

        Args:
            c: 密文

        Returns:
            明文m

        Note:
            解密公式: m = L(c^λ mod n²) * μ mod n
            其中L(x) = (x-1)/n
        """
        # 计算c^λ mod n²
        x = pow(c, self.lam, self.n2)

        # 计算L(x) = (x-1)/n
        Lx = (x - 1) // self.n

        # 计算m = Lx * μ mod n
        return (Lx * self.mu) % self.n

    def e_add(self, c1: int, c2: int) -> int:
        """
        同态加法

        计算E(m1) ⊕ E(m2) = E(m1 + m2 mod n)

        Args:
            c1: 第一个密文
            c2: 第二个密文

        Returns:
            两个明文之和的密文

        Note:
            同态加法通过密文相乘实现：
            E(m1) * E(m2) mod n² = E(m1 + m2 mod n)
        """
        return (c1 * c2) % self.n2

    def e_mul_const(self, c: int, k: int) -> int:
        """
        同态标量乘法

        计算k ⊗ E(m) = E(k*m mod n)

        Args:
            c: 密文
            k: 标量（整数）

        Returns:
            明文与标量乘积的密文

        Note:
            同态标量乘法通过密文幂运算实现：
            E(m)^k mod n² = E(k*m mod n)
        """
        return pow(c, int(k), self.n2)

    def get_public_key(self) -> Tuple[int, int]:
        """
        获取公钥

        Returns:
            (n, g)元组
        """
        return (self.n, self.g)

    def get_key_size(self) -> int:
        """
        获取密钥大小（位数）

        Returns:
            模数n的位数
        """
        return self.n.bit_length()
