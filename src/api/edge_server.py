"""Edge节点 Flask REST API

True federated scenario: Edge holds its own private A_k matrix.
Master holds y and the Paillier private key.

Endpoints:
  POST /init_local   - Receive y from Master; compute B_k, alpha_k, Bbar_k from local A_k
  POST /init_params  - Encrypted mode only: receive encrypted alpha_hat from Master
  POST /compute_x    - Compute local x update for one iteration
  GET  /health       - Health / status check
  POST /reset        - Reset state

All endpoints (except /health) require the header:
  X-Session-Token: <6-digit token>
"""

from flask import Flask, request, jsonify
import numpy as np
from typing import Optional
import logging
from utils.gpu import get_xp, to_numpy

logger = logging.getLogger(__name__)


class EdgeState:
    """Edge节点计算状态"""

    def __init__(self, session_token: str = "", A_k: Optional[np.ndarray] = None):
        self.session_token: str = session_token
        self.A_k: Optional[np.ndarray] = A_k      # private local data matrix
        self.Nk: Optional[int] = None
        # Plain mode params (computed locally from A_k and y)
        self.alpha: Optional[np.ndarray] = None
        self.Bbar: Optional[np.ndarray] = None
        # Encrypted mode params
        self.alpha_hat: Optional[list] = None
        self.Bbar_q: Optional[np.ndarray] = None
        self.n: Optional[int] = None
        self.g: Optional[int] = None
        self.n2: Optional[int] = None
        self.delta: Optional[int] = None
        self.delta2: Optional[int] = None
        self.alpha_min: Optional[float] = None
        self.alpha_max: Optional[float] = None
        self.Bbar_min: Optional[float] = None
        self.Bbar_max: Optional[float] = None
        self.encrypted: bool = False
        # GUI-visible status
        self.compute_count: int = 0
        self.max_iter: int = 0        # total iterations for this task
        self.busy: bool = False

    def reset(self):
        token = self.session_token
        A_k = self.A_k
        self.__init__(token, A_k)

    def has_data(self) -> bool:
        return self.A_k is not None

    def is_initialized(self) -> bool:
        """True after /init_local has been called successfully."""
        return self.Nk is not None

    def is_params_set(self) -> bool:
        if self.encrypted:
            return self.alpha_hat is not None and self.Bbar_q is not None
        return self.alpha is not None and self.Bbar is not None


def create_app(state: Optional[EdgeState] = None) -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    if state is None:
        state = EdgeState()

    # ── Token middleware ──────────────────────────────────────────────────────

    @app.before_request
    def _check_token():
        if request.method == "OPTIONS":
            return
        if request.path in ("/health",):
            return
        if state.session_token:
            tok = request.headers.get("X-Session-Token", "")
            if tok != state.session_token:
                return jsonify({"ok": False, "error": "Invalid session token"}), 403

    # ── Computation endpoints ─────────────────────────────────────────────────

    @app.route("/receive_Ak", methods=["POST"])
    def receive_Ak():
        """
        Receive A_k matrix pushed by Master (background-1 scenario).
        Called before /init_local. Overwrites any previously loaded A_k.
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"ok": False, "error": "No JSON body"}), 400
            A_k = np.array(data["A_k"], dtype=float)
            if A_k.ndim != 2:
                return jsonify({"ok": False, "error": "A_k must be 2-D"}), 400
            state.A_k = A_k
            logger.info(f"receive_Ak: shape={A_k.shape}")
            return jsonify({"ok": True, "shape": list(A_k.shape)})
        except KeyError as e:
            return jsonify({"ok": False, "error": f"Missing key: {e}"}), 400
        except Exception as e:
            logger.exception("receive_Ak error")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/init_local", methods=["POST"])
    def init_local():
        """
        Receive y from Master. Compute B_k, alpha_k, Bbar_k from local A_k.

        Plain mode:  stores alpha and Bbar, returns {"ok": True, "Nk": int}
        Encrypted mode: additionally returns quantized params for Master to encrypt:
            {"ok": True, "Nk": int,
             "alpha_q": [...], "alpha_min": float, "alpha_max": float,
             "Bbar_q": [[...]], "Bbar_min": float, "Bbar_max": float}
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"ok": False, "error": "No JSON body"}), 400
            if not state.has_data():
                return jsonify({"ok": False, "error": "No local A_k loaded. Load data first."}), 400

            y = np.array(data["y"], dtype=float)
            rho = float(data["rho"])
            encrypted = bool(data.get("encrypted", False))

            A_k = state.A_k
            M, Nk = A_k.shape
            if len(y) != M:
                return jsonify({
                    "ok": False,
                    "error": f"y length {len(y)} != A_k rows {M}"
                }), 400

            state.encrypted = encrypted
            state.Nk = Nk
            # Reset iteration counter and store max_iter for progress display
            state.compute_count = 0
            state.max_iter = int(data.get("max_iter", 100))

            if not encrypted:
                xp = get_xp()
                Ak_xp = xp.asarray(A_k)
                y_xp = xp.asarray(y)
                AtA = Ak_xp.T @ Ak_xp
                Bk_xp = xp.linalg.inv(AtA + rho * xp.eye(Nk))
                state.alpha = to_numpy(Bk_xp @ (Ak_xp.T @ y_xp))
                state.Bbar = to_numpy(rho * Bk_xp)
                logger.info(f"init_local (plain): Nk={Nk}")
                return jsonify({"ok": True, "Nk": Nk})
            else:
                # Encrypted mode: compute params and return quantized values to Master
                from core.quantization import quantize_gamma1, quantize_gamma2
                AtA = A_k.T @ A_k
                Bk = np.linalg.inv(AtA + rho * np.eye(Nk))
                alpha_k = Bk @ (A_k.T @ y)
                Bbar_k = rho * Bk

                delta = int(data.get("Delta", 2**12))
                delta2 = int(data.get("Delta2", 2**16))

                alpha_q, a_min, a_max = quantize_gamma1(alpha_k, delta2)
                Bbar_q, b_min, b_max = quantize_gamma2(Bbar_k.flatten(), delta)
                Bbar_q_2d = Bbar_q.reshape(Bbar_k.shape)

                # Store Bbar_q for later use in compute_x
                state.Bbar_q = Bbar_q_2d
                state.Bbar_min = float(b_min)
                state.Bbar_max = float(b_max)
                state.delta = delta
                state.delta2 = delta2

                logger.info(f"init_local (encrypted): Nk={Nk}")
                return jsonify({
                    "ok": True,
                    "Nk": Nk,
                    "alpha_q": alpha_q.tolist(),
                    "alpha_min": float(a_min),
                    "alpha_max": float(a_max),
                    "Bbar_q": Bbar_q_2d.tolist(),
                    "Bbar_min": float(b_min),
                    "Bbar_max": float(b_max),
                })
        except KeyError as e:
            return jsonify({"ok": False, "error": f"Missing key: {e}"}), 400
        except Exception as e:
            logger.exception("init_local error")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/init_params", methods=["POST"])
    def init_params():
        """
        Encrypted mode only: receive encrypted alpha_hat and Paillier params from Master.
        Called after /init_local has returned quantized params.
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"ok": False, "error": "No JSON body"}), 400
            if not state.is_initialized():
                return jsonify({"ok": False, "error": "Call /init_local first"}), 400

            state.alpha_hat = data["alpha_hat"]
            state.n = int(data["n"])
            state.g = int(data["g"])
            state.n2 = state.n * state.n
            state.delta = int(data["Delta"])
            state.delta2 = int(data["Delta2"])
            state.alpha_min = float(data["alpha_min"])
            state.alpha_max = float(data["alpha_max"])
            state.Bbar_min = float(data["Bbar_min"])
            state.Bbar_max = float(data["Bbar_max"])
            logger.info("init_params (encrypted): OK")
            return jsonify({"ok": True})
        except KeyError as e:
            return jsonify({"ok": False, "error": f"Missing key: {e}"}), 400
        except Exception as e:
            logger.exception("init_params error")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/compute_x", methods=["POST"])
    def compute_x():
        """
        Compute local x update.

        Plain mode:  receives z_k, v_k; returns x_k and Ax_k (= A_k @ x_k)
        Encrypted:   receives z_hat, v_hat; returns x_hat
                     (Master decrypts and computes Ax_k itself — not needed here)
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"ok": False, "error": "No JSON body"}), 400
            if not state.is_params_set():
                return jsonify({"ok": False, "error": "Call /init_local (and /init_params for encrypted) first"}), 400

            state.busy = True
            state.compute_count += 1

            if state.encrypted:
                z_hat = data["z_hat"]
                v_hat = data["v_hat"]
                n2 = state.n2
                Bbar_q = state.Bbar_q
                Nk = Bbar_q.shape[0]

                enc_diff = [(int(z_hat[j]) * int(v_hat[j])) % n2
                            for j in range(len(z_hat))]

                bmin, bmax = state.Bbar_min, state.Bbar_max
                if bmax == bmin:
                    Bbar_real = np.zeros_like(Bbar_q, dtype=float)
                else:
                    Bbar_real = (Bbar_q / float(state.delta)) * (bmax - bmin) + bmin

                x_hat_list = []
                for i in range(Nk):
                    xi_hat = int(state.alpha_hat[i])
                    for j, ed in enumerate(enc_diff):
                        coeff = int(round(Bbar_real[i][j]))
                        if coeff != 0:
                            term = pow(ed, coeff, n2)
                            xi_hat = (xi_hat * term) % n2
                    x_hat_list.append(xi_hat)
                return jsonify({"ok": True, "x_hat": x_hat_list})
            else:
                z_k = np.array(data["z_k"], dtype=float)
                v_k = np.array(data["v_k"], dtype=float)
                xp = get_xp()
                x_k = to_numpy(
                    xp.asarray(state.alpha) + xp.asarray(state.Bbar) @ (
                        xp.asarray(z_k) - xp.asarray(v_k)
                    )
                )
                # Also return A_k @ x_k so Master can compute the residual ||Ax-y||²/M
                Ax_k = to_numpy(xp.asarray(state.A_k) @ xp.asarray(x_k))
                return jsonify({"ok": True, "x_k": x_k.tolist(), "Ax_k": Ax_k.tolist()})
        except KeyError as e:
            return jsonify({"ok": False, "error": f"Missing key: {e}"}), 400
        except Exception as e:
            logger.exception("compute_x error")
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            state.busy = False

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "ok": True,
            "has_data": state.has_data(),
            "initialized": state.is_initialized(),
            "params_set": state.is_params_set(),
            "encrypted": state.encrypted,
            "Nk": state.Nk,
        })

    @app.route("/reset", methods=["POST"])
    def reset():
        state.reset()
        logger.info("Edge state reset")
        return jsonify({"ok": True})

    app.edge_state = state
    return app
