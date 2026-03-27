import sys
import numpy as np
sys.path.append('/mnt/3p-admm-pc2')
from crypto.quantization import gamma1, gamma2

delta = 10**10

# 测试gamma2
v = np.array([0.5, -0.3, 1.2, -0.8, 0.1])
q, v_min, v_max = gamma2(v, delta)

print("测试 Γ2 量化：")
print(f"原始向量: {v}")
print(f"量化结果: {q}")
print(f"v_min={v_min:.4f}, v_max={v_max:.4f}")

# 验证量化值都是非负整数
assert all(qi >= 0 for qi in q), "量化结果包含负数！"
print("✓ 量化结果全部非负")

# 测试不同delta的精度损失
print("\n测试不同delta的精度损失：")
for exp in [5, 10, 15]:
    d = 10**exp
    q2, mn, mx = gamma2(v, d)
    # 逆量化
    v_recover = np.array(q2, dtype=np.float64) * (mx - mn) / d + mn
    loss = np.max(np.abs(v_recover - v))
    print(f"delta=10^{exp}: 精度损失={loss:.2e}")
