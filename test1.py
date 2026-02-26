import numpy as np
from PIL import Image
from phe import paillier
import time

# 1. 初始化
print("正在准备密钥...")
public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)

# 2. 读取图片 (找一张你的图片放在目录下，改名为 test.jpg)
# 注意：一定要缩小，否则加密会慢到你怀疑人生
img = Image.open('Image/1.png').convert('L') # 先转成黑白灰度图
img = img.resize((10, 10)) # 先从 10x10 像素开始测试！
pixels = np.array(img).flatten()

# 3. 加密整个图片
print(f"正在加密 {len(pixels)} 个像素...")
start_time = time.time()
# 这行就是论文里说的隐私预处理
encrypted_image = [public_key.encrypt(int(x)) for x in pixels]
print(f"加密耗时: {time.time() - start_time:.2f} 秒")

# 4. 盲处理：所有人变亮
print("第三方（盲处理中）...")
bright_encrypted = [p + 100 for p in encrypted_image]

# 5. 解密并还原
print("解密中...")
decrypted_pixels = [private_key.decrypt(p) for p in bright_encrypted]
# 限制像素范围 0-255 并转回图片
result_array = np.clip(decrypted_pixels, 0, 255).astype(np.uint8).reshape((10, 10))
result_img = Image.fromarray(result_array)
result_img=result_img.resize((100,100))
result_img.save('blind_result.png')

print("第一步图片盲处理已完成！请查看 blind_result.png")
