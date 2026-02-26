from flask import Flask, request, jsonify
from phe import paillier
import time

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'edge node running', 'time': time.time()})


@app.route('/blind_process', methods=['POST'])
def blind_process():
    data = request.json

    # 从主节点收到：公钥 + 密文列表 + 操作类型
    public_key = paillier.PaillierPublicKey(n=int(data['public_key_n']))
    operation = data['operation']  # 'brighten' / 'blur'
    width = data['width']
    height = data['height']

    # 恢复密文列表（服务器只看到大整数，完全不知道图片内容）
    enc_pixels = [
        paillier.EncryptedNumber(public_key, int(c))
        for c in data['encrypted_pixels']
    ]

    print(f"收到 {len(enc_pixels)} 个密文，执行 {operation}")

    if operation == 'brighten':
        amount = data.get('amount', 50)
        result = [p + amount for p in enc_pixels]

    elif operation == 'blur':
        result, counts = blind_blur(enc_pixels, width, height)
        # blur需要额外返回counts供解密时使用
        return jsonify({
            'result': [str(e.ciphertext()) for e in result],
            'counts': counts,
            'status': 'ok'
        })

    return jsonify({
        'result': [str(e.ciphertext()) for e in result],
        'counts': [1] * len(result),
        'status': 'ok'
    })


def blind_blur(enc_pixels, width, height):
    """均值模糊——服务器看不到任何像素值"""
    result = []
    counts = []
    for i in range(height):
        for j in range(width):
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbors.append(enc_pixels[ni * width + nj])
            total = neighbors[0]
            for n in neighbors[1:]:
                total = total + n
            result.append(total)
            counts.append(len(neighbors))
    return result, counts


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
