from flask import Flask, request, jsonify
import numpy as np
import argparse

app = Flask(__name__)

STATE = {
    'Bk': None,       # inverse matrix (Nk x Nk)
    'alpha': None,    # alpha_k = Bk @ (A_k.T @ y)
    'Bbar': None,     # rho * Bk
    'Nk': None
}

def to_list(x):
    return x.tolist()

def to_np(arr):
    return np.array(arr, dtype=float)

@app.route('/init_atat', methods=['POST'])
def init_atat():
    """
    Master sends: {'AtA': [...], 'rho': float}
    Edge computes Bk = inv(AtA + rho I) and returns Bk (as list-of-lists)
    """
    js = request.get_json()
    AtA = to_np(js['AtA'])
    rho = float(js['rho'])
    Nk = AtA.shape[0]
    I = np.eye(Nk)
    Bk = np.linalg.inv(AtA + rho * I)
    STATE['Bk'] = Bk
    STATE['Nk'] = Nk
    return jsonify({'ok': True, 'Bk': to_list(Bk)})

@app.route('/init_params', methods=['POST'])
def init_params():
    """
    Master sends unencrypted params:
    {'alpha': [...], 'Bbar': [...]} where those are lists
    """
    js = request.get_json()
    STATE['alpha'] = to_np(js['alpha'])
    STATE['Bbar'] = to_np(js['Bbar'])
    return jsonify({'ok': True})

@app.route('/compute_x', methods=['POST'])
def compute_x():
    """
    Master sends {'z_k': [...], 'v_k': [...]} (these are vectors of length Nk)
    Edge computes x_k = alpha + Bbar @ (z_k - v_k), returns x_k
    """
    js = request.get_json()
    z = to_np(js['z_k'])
    v = to_np(js['v_k'])
    if STATE['alpha'] is None or STATE['Bbar'] is None:
        return jsonify({'ok': False, 'error': 'not initialized'}), 400
    xk = STATE['alpha'] + STATE['Bbar'].dot(z - v)
    return jsonify({'ok': True, 'x_k': to_list(xk)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'id': STATE.get('Nk')})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()
    print(f"[EDGE] Starting on {args.host}:{args.port} id={args.id}")
    app.run(host=args.host, port=args.port)
