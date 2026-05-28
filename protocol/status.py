import threading
import time
from copy import deepcopy

_lock = threading.Lock()
_status = {
    'stage': 'idle',   # idle, generating_keys, init_sending, init_waiting, init_done, sharing_data, iterating, collecting, done, error
    'detail': '',
    'current_iter': 0,
    'max_iter': 0,
    'last_mse': None,
    'nodes_busy': [],
    # per-node state map: { 'edge1': 'initializing', 'edge2': 'computing iter 3', ... }
    'node_states': {},
    'timestamp': time.time(),
}


def set_status(stage: str, detail: str = '', current_iter: int = None, max_iter: int = None, last_mse=None, nodes_busy=None):
    with _lock:
        _status['stage'] = stage
        _status['detail'] = detail
        if current_iter is not None:
            _status['current_iter'] = int(current_iter)
        if max_iter is not None:
            _status['max_iter'] = int(max_iter)
        if last_mse is not None:
            try:
                _status['last_mse'] = float(last_mse)
            except Exception:
                _status['last_mse'] = last_mse
        if nodes_busy is not None:
            _status['nodes_busy'] = list(nodes_busy)
        # optional: allow passing a dict of node_states
        if isinstance(max_iter, dict) and nodes_busy is None:
            # legacy: avoid mis-ordered args; do nothing
            pass
        # no change to node_states here unless provided separately via set_node_states / set_node_state
        _status['timestamp'] = time.time()


def set_node_state(node_name: str, state: str):
    with _lock:
        if not isinstance(_status.get('node_states'), dict):
            _status['node_states'] = {}
        # normalize node name to avoid accidental whitespace/mismatch
        try:
            key = str(node_name).strip()
        except Exception:
            key = node_name
        _status['node_states'][key] = state
        _status['timestamp'] = time.time()


def set_node_states(states: dict):
    with _lock:
        if not isinstance(_status.get('node_states'), dict):
            _status['node_states'] = {}
        # normalize incoming dict keys
        for k, v in (states or {}).items():
            try:
                key = str(k).strip()
            except Exception:
                key = k
            _status['node_states'][key] = v
        _status['timestamp'] = time.time()


def get_status():
    with _lock:
        return deepcopy(_status)


def reset():
    with _lock:
        _status.update({
            'stage': 'idle',
            'detail': '',
            'current_iter': 0,
            'max_iter': 0,
            'last_mse': None,
            'nodes_busy': [],
            'node_states': {},
            'timestamp': time.time(),
        })
