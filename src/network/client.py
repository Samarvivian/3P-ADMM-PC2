"""
网络通信模块 — Master → Edge HTTP Client

EdgeClient sends HTTP requests to an Edge node's REST API.
All requests include the X-Session-Token header for authentication.
"""

import requests
import time
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class NetworkError(Exception):
    """Network communication error"""
    pass


class EdgeClient:
    """
    HTTP client for communicating with one Edge node.

    Args:
        base_url:      Edge node URL, e.g. "http://192.168.1.5:5000"
        session_token: 6-digit session token (must match Edge's token)
        timeout:       Request timeout in seconds
        max_retries:   Number of retries on transient failures
        retry_delay:   Seconds to wait between retries
    """

    def __init__(
        self,
        base_url: str,
        session_token: str = "",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if not base_url:
            raise ValueError("base_url must not be empty")
        self.base_url = base_url.rstrip("/")
        self.session_token = session_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _headers(self) -> Dict[str, str]:
        return {"X-Session-Token": self.session_token} if self.session_token else {}

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=data, headers=self._headers(),
                                     timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Timeout {url} (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(f"Connection error {url} (attempt {attempt + 1})")
            except requests.exceptions.HTTPError as e:
                raise NetworkError(f"HTTP error: {e}") from e
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"Request error {url}: {e}")
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
        raise NetworkError(
            f"Request failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"GET request failed: {e}") from e

    # ── High-level API calls ──────────────────────────────────────────────────

    def send_Ak(self, A_k: List[List[float]]) -> Dict[str, Any]:
        """
        Push A_k matrix to Edge (background-1 scenario).
        Must be called before init_local.
        """
        return self._post("/receive_Ak", {"A_k": A_k})

    def init_local(
        self,
        y: List[float],
        rho: float,
        encrypted: bool = False,
        max_iter: int = 100,
        Delta: int = 2**12,
        Delta2: int = 2**16,
    ) -> Dict[str, Any]:
        """
        Send y to Edge. Edge computes B_k, alpha_k, Bbar_k from its local A_k.

        Plain mode:     returns {"ok": True, "Nk": int}
        Encrypted mode: returns {"ok": True, "Nk": int,
                                 "alpha_q": [...], "alpha_min": float, "alpha_max": float,
                                 "Bbar_q": [[...]], "Bbar_min": float, "Bbar_max": float}
        """
        return self._post("/init_local", {
            "y": y, "rho": rho, "encrypted": encrypted,
            "max_iter": max_iter,
            "Delta": Delta, "Delta2": Delta2,
        })

    def init_params_encrypted(
        self,
        alpha_hat: List[int],
        n: int,
        g: int,
        Delta: int,
        Delta2: int,
        alpha_min: float,
        alpha_max: float,
        Bbar_min: float,
        Bbar_max: float,
    ) -> Dict[str, Any]:
        """
        Encrypted mode only: send encrypted alpha_hat and Paillier params to Edge.
        Called after init_local has returned quantized params.
        """
        return self._post("/init_params", {
            "alpha_hat": alpha_hat,
            "n": n, "g": g,
            "Delta": Delta, "Delta2": Delta2,
            "alpha_min": alpha_min, "alpha_max": alpha_max,
            "Bbar_min": Bbar_min, "Bbar_max": Bbar_max,
        })

    def compute_x(
        self,
        z_k: List[float] = None,
        v_k: List[float] = None,
        z_hat: List[int] = None,
        v_hat: List[int] = None,
    ) -> Dict[str, Any]:
        if z_hat is not None:
            if v_hat is None:
                raise ValueError("Encrypted mode requires both z_hat and v_hat")
            data = {"z_hat": z_hat, "v_hat": v_hat}
        else:
            if z_k is None or v_k is None:
                raise ValueError("Plain mode requires z_k and v_k")
            data = {"z_k": z_k, "v_k": v_k}
        return self._post("/compute_x", data)

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def reset(self) -> Dict[str, Any]:
        return self._post("/reset", {})
