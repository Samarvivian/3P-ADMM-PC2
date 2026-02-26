import numpy as np
from PIL import Image
from phe import paillier
import time

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)  # 先用1024位

img = Image.open('Image/1.png').convert('L')
img = img.resize((64, 64))  # 逐步扩大
pixels = np.array(img).flatten().astype(float)

# 关键改进：论文的量化思路
# 把浮点像素值量化到整数区间，避免精度问题
DELTA = 1  # 像素本来就是0-255整数，直接用
pixels_int = pixels.astype(int)

print(f"加密 {len(pixels_int)} 个像素...")
start = time.time()

# 用并行加速：phe库支持多线程
from concurrent.futures import ThreadPoolExecutor
def encrypt_pixel(x):
    return public_key.encrypt(int(x))

with ThreadPoolExecutor(max_workers=8) as executor:
    encrypted_image = list(executor.map(encrypt_pixel, pixels_int))

print(f"并行加密耗时: {time.time()-start:.2f}s")

# 盲滤镜：亮度+50（第三方执行，看不到原图）
def blind_brighten(enc_pixels, amount=50):
    return [p + amount for p in enc_pixels]

bright = blind_brighten(encrypted_image, 50)

# 解密
decrypted = np.array([private_key.decrypt(p) for p in bright])
result = np.clip(decrypted, 0, 255).astype(np.uint8).reshape((64, 64))
Image.fromarray(result).resize((256, 256)).save('result_bright.png')
print("完成！")