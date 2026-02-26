from phe import paillier

# 1. 生成钥匙（就像生成银行卡号和密码）
print("正在生成密钥...")
public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

# 2. 我们想保护的秘密数字（比如一个像素值）
secret_number = 150
print(f"原始数字: {secret_number}")

# 3. 加密 (使用公钥)
encrypted_number = public_key.encrypt(secret_number)
print(f"加密后的乱码: {encrypted_number.ciphertext()}") # 你会看到一个超级大的数字

# 4. 盲计算 (重点！在不解密的情况下加 50)
# 这就是论文里说的隐私计算：第三方只碰密文，不碰原数据
encrypted_result = encrypted_number + 50
print("已经在不解密的情况下完成了 +50 运算")

# 5. 解密 (使用私钥)
decrypted_number = private_key.decrypt(encrypted_result)
print(f"解密后的结果: {decrypted_number}")

if decrypted_number == 200:
    print("恭喜！第 0 步通关：加密、同态运算、解密全部成功！")