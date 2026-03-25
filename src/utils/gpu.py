"""GPU acceleration helper — lazy CuPy singleton with numpy fallback.

Supports loading CuPy from system Python installation when running from
a PyInstaller packaged executable.
"""

import sys
import os
import logging
import platform

_logger = logging.getLogger(__name__)
_xp = None
_gpu_available = False
_gpu_error = None


def _find_system_cupy_paths():
    """
    Find potential CuPy installation paths in system Python.
    Returns a list of site-packages directories to search.
    """
    seen = set()
    paths = []

    def _add(p):
        if p and p not in seen and os.path.isdir(p):
            seen.add(p)
            paths.append(p)

    # 1. Explicit override via environment variable
    env_path = os.environ.get("CUPY_PATH", "")
    if env_path:
        _add(env_path)

    if platform.system() == "Windows":
        # 2. Windows Registry — most reliable; covers all official Python installs
        try:
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for reg_path in (
                    r"SOFTWARE\Python\PythonCore",
                    r"SOFTWARE\WOW6432Node\Python\PythonCore",
                ):
                    try:
                        core_key = winreg.OpenKey(hive, reg_path)
                    except OSError:
                        continue
                    i = 0
                    while True:
                        try:
                            ver = winreg.EnumKey(core_key, i)
                            i += 1
                        except OSError:
                            break
                        try:
                            ip_key = winreg.OpenKey(core_key, ver + r"\InstallPath")
                            install_path, _ = winreg.QueryValueEx(ip_key, None)
                            _add(os.path.join(install_path, "Lib", "site-packages"))
                        except OSError:
                            pass
        except ImportError:
            pass

        # 3. Ask the system `python` executable directly (covers PATH-based installs)
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-c",
                 "import site, os; paths = site.getsitepackages() + [site.getusersitepackages()];"
                 "[print(p) for p in paths if p and os.path.isdir(p)]"],
                capture_output=True, text=True, timeout=5,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    _add(line)
        except Exception:
            pass

        # 4. Hardcoded fallback paths
        for version in ["314", "313", "312", "311", "310"]:
            for base in [
                f"C:\\Python{version}",
                os.path.expandvars(f"%LOCALAPPDATA%\\Programs\\Python\\Python{version}"),
            ]:
                _add(os.path.join(base, "Lib", "site-packages"))
            # pip install --user puts packages in %APPDATA%\Python\PythonXYZ\site-packages
            _add(os.path.expandvars(
                f"%APPDATA%\\Python\\Python{version}\\site-packages"))

        # 5. Anaconda / Miniconda
        for conda_base in [
            os.path.expandvars("%USERPROFILE%\\anaconda3"),
            os.path.expandvars("%USERPROFILE%\\miniconda3"),
            "C:\\ProgramData\\anaconda3",
            "C:\\ProgramData\\miniconda3",
        ]:
            if os.path.isdir(conda_base):
                _add(os.path.join(conda_base, "Lib", "site-packages"))
                envs_dir = os.path.join(conda_base, "envs")
                if os.path.isdir(envs_dir):
                    for env_name in os.listdir(envs_dir):
                        _add(os.path.join(envs_dir, env_name, "Lib", "site-packages"))

    else:
        # Linux / macOS
        try:
            import subprocess
            result = subprocess.run(
                ["python3", "-c",
                 "import site, os; "
                 "[print(p) for p in site.getsitepackages() if os.path.isdir(p)]"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                _add(line.strip())
        except Exception:
            pass

        for p in [
            "/usr/local/lib/python3.14/site-packages",
            "/usr/local/lib/python3.13/site-packages",
            "/usr/local/lib/python3.12/site-packages",
            "/usr/local/lib/python3.11/site-packages",
            "/usr/local/lib/python3.10/site-packages",
            "/usr/lib/python3/dist-packages",
            os.path.expanduser("~/.local/lib/python3.14/site-packages"),
            os.path.expanduser("~/.local/lib/python3.13/site-packages"),
            os.path.expanduser("~/.local/lib/python3.12/site-packages"),
            os.path.expanduser("~/.local/lib/python3.11/site-packages"),
        ]:
            _add(p)

    return paths


def _try_import_cupy_from_system():
    """
    Try to import CuPy from system Python installations.
    Returns the cupy module if successful, None otherwise.
    """
    # First try normal import (works when running from source)
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        pass

    # Search system Python installations (works both from source and packaged exe)
    paths = _find_system_cupy_paths()

    for site_path in paths:
        cupy_path = os.path.join(site_path, "cupy")
        if not os.path.isdir(cupy_path):
            continue

        # Temporarily add to sys.path
        if site_path not in sys.path:
            sys.path.insert(0, site_path)

        try:
            import importlib
            cp = importlib.import_module("cupy")
            # Verify it works
            cp.cuda.runtime.getDeviceCount()
            _logger.info(f"Loaded CuPy from: {site_path}")
            return cp
        except Exception as e:
            _logger.debug(f"CuPy at {site_path} failed: {e}")
            # Remove from path if it didn't work
            if site_path in sys.path:
                sys.path.remove(site_path)
            continue

    return None


def get_xp():
    """Return cupy if available, otherwise numpy."""
    global _xp, _gpu_available, _gpu_error

    if _xp is not None:
        return _xp

    try:
        cp = _try_import_cupy_from_system()
        if cp is not None:
            _xp = cp
            _gpu_available = True
            _logger.info("GPU acceleration enabled via CuPy")
            return _xp
    except Exception as e:
        _gpu_error = str(e)
        _logger.debug(f"CuPy import failed: {e}")

    # Fallback to numpy
    import numpy as np
    _xp = np
    _gpu_available = False
    _gpu_error = "CuPy not found or CUDA not available"
    _logger.info("Using CPU (numpy) — CuPy not available")
    return _xp


def to_numpy(arr):
    """CuPy → numpy. No-op if already numpy."""
    if hasattr(arr, "get"):
        return arr.get()
    return arr


def is_gpu_available() -> bool:
    get_xp()
    return _gpu_available


def get_gpu_error() -> str:
    """Return the error message if GPU is not available."""
    get_xp()
    return _gpu_error or ""