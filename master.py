# master.py  ← 在你自己电脑上运行
import requests
import numpy as np
from PIL import Image
from phe import paillier

SERVER_URL = "https://blind-filter-edge.onrender.com"  # 换成你的URL

def run(image_path, operation='brighten'):
    # 1. 只在本机生成密钥
    print("生成密钥（仅本机）...")
    public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

    # 2. 读取并加密图片
    img = Image.open(image_path).convert('L').resize((16, 16))
    pixels = np.array(img).flatten().astype(int)
    w, h = img.size

    print(f"加密 {len(pixels)} 个像素...")
    enc_pixels = [public_key.encrypt(int(p)) for p in pixels]

    # 3. 发送密文给服务器（服务器看不到图片）
    print(f"发送给服务器执行 {operation}...")
    payload = {
        'public_key_n':      str(public_key.n),
        'encrypted_pixels':  [str(e.ciphertext()) for e in enc_pixels],
        'operation':         operation,
        'amount':            80,
        'width':             w,
        'height':            h
    }
    resp = requests.post(f"{SERVER_URL}/blind_process", json=payload, timeout=120)
    print("服务器状态码:", resp.status_code)
    print("服务器返回内容:", resp.text)  # ← 加这行
    data = resp.json()

    # 4. 只有本机能解密
    print("解密还原...")
    processed = []
    for enc_str, count in zip(data['result'], data['counts']):
        enc_num = paillier.EncryptedNumber(public_key, int(enc_str))
        val = private_key.decrypt(enc_num) / count
        processed.append(val)

    result = np.clip(processed, 0, 255).astype(np.uint8).reshape((h, w))
    out_path = f'result_{operation}.png'
    Image.fromarray(result).resize((256, 256)).save(out_path)
    print(f"完成！已保存到 {out_path}")

run('Image/1.png', operation='brighten')
