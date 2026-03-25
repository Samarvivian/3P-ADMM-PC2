# -*- mode: python ; coding: utf-8 -*-
"""3P-ADMM-PC2 PyInstaller spec — optimized single exe"""

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('examples/data', 'examples/data'),
    ],
    hiddenimports=[
        # Project internal modules
        'gui', 'gui.style', 'gui.i18n', 'gui.config', 'gui.master_gui',
        'gui.edge_gui', 'gui.data_loader', 'gui.tooltip',
        'core', 'core.admm', 'core.crypto', 'core.distributed_admm',
        'core.quantization',
        'api', 'api.edge_server',
        'network', 'network.client',
        'utils', 'utils.gpu',
        # Third-party dependencies
        'numpy', 'numpy.core',
        'scipy', 'scipy.linalg', 'scipy.io', 'scipy.io.matlab',
        'scipy.sparse', 'scipy.sparse.base',
        'matplotlib', 'matplotlib.pyplot', 'matplotlib.figure',
        'matplotlib.backends', 'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.font_manager', 'matplotlib.patches', 'matplotlib.lines',
        'matplotlib.ticker', 'matplotlib.text', 'matplotlib.path',
        'matplotlib.collections', 'matplotlib.image', 'matplotlib._cm',
        'PIL', 'PIL.Image', 'pyparsing', 'cycler', 'kiwisolver',
        'dateutil', 'packaging',
        'tkinter', 'tkinter.ttk', 'tkinter.messagebox',
        'tkinter.filedialog', 'tkinter.font', 'tkinter.scrolledtext',
        'flask', 'flask.json', 'werkzeug', 'werkzeug.serving',
        'werkzeug.routing', 'jinja2', 'click', 'itsdangerous', 'blinker',
        'requests', 'urllib3', 'urllib3.util', 'certifi',
        'charset_normalizer', 'idna',
        'psutil', 'psutil._pswindows', 'pynvml', 'pynvml.nvml', 'pynvml.nvmlLib',
        'threading', 'logging', 'logging.handlers',
        'json', 'socket', 'secrets', 'queue',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest', '_pytest', 'pluggy',
        'IPython', 'ipython', 'jupyter', 'jupyter_client', 'jupyter_core',
        'notebook', 'nbconvert', 'nbformat', 'ipykernel', 'ipywidgets',
        'pandas', 'pyarrow', 'polars',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'cv2', 'opencv', 'skimage', 'imageio',
        'torch', 'tensorflow', 'keras', 'onnx', 'onnxruntime',
        'cupy', 'cupyx',
        'h5py', 'tables', 'numba', 'llvmlite',
        'sympy', 'nose', 'sphinx', 'docutils',
        'boto3', 'botocore',
        'google', 'googleapiclient',
        'azure', 'azureml',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='3P-ADMM-PC2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)