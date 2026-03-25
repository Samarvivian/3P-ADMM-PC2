"""
数据加载模块

支持多种数据格式的加载、验证和预览功能
"""

import numpy as np
import os
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器

    支持加载.npy, .mat, .csv, .h5格式的数据文件
    提供数据验证、预览和内存检查功能
    """

    def __init__(self, max_memory_mb: float = 2000):
        """
        初始化数据加载器

        Args:
            max_memory_mb: 最大内存限制（MB）
        """
        self.max_memory_mb = max_memory_mb
        self.supported_formats = [".npy", ".mat", ".csv", ".h5", ".hdf5"]

    def detect_format(self, filepath: str) -> str:
        """
        自动检测文件格式

        Args:
            filepath: 文件路径

        Returns:
            文件格式（npy/mat/csv/h5）

        Raises:
            ValueError: 如果格式不支持
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".npy":
            return "npy"
        elif ext == ".mat":
            return "mat"
        elif ext == ".csv":
            return "csv"
        elif ext in [".h5", ".hdf5"]:
            return "h5"
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def load_npy_file(self, filepath: str) -> np.ndarray:
        """
        加载.npy文件

        Args:
            filepath: 文件路径

        Returns:
            numpy数组

        Raises:
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading .npy file: {filepath}")
        data = np.load(filepath)
        logger.info(f"Loaded shape: {data.shape}, dtype: {data.dtype}")
        return data

    def load_mat_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        加载.mat文件

        Args:
            filepath: 文件路径

        Returns:
            包含数据的字典

        Raises:
            FileNotFoundError: 文件不存在
            ImportError: scipy未安装
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy is required to load .mat files. "
                            "Install it with: pip install scipy")

        logger.info(f"Loading .mat file: {filepath}")
        data = scipy.io.loadmat(filepath)

        # 过滤掉MATLAB的元数据
        filtered_data = {k: v for k, v in data.items()
                        if not k.startswith('__')}

        logger.info(f"Loaded {len(filtered_data)} arrays from .mat file")
        return filtered_data

    def load_csv_file(self, filepath: str) -> np.ndarray:
        """
        加载.csv文件

        Args:
            filepath: 文件路径

        Returns:
            numpy数组

        Raises:
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading .csv file: {filepath}")
        data = np.loadtxt(filepath, delimiter=',')
        logger.info(f"Loaded shape: {data.shape}, dtype: {data.dtype}")
        return data

    def load_h5_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        加载.h5文件

        Args:
            filepath: 文件路径

        Returns:
            包含数据的字典

        Raises:
            FileNotFoundError: 文件不存在
            ImportError: h5py未安装
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required to load .h5 files. "
                            "Install it with: pip install h5py")

        logger.info(f"Loading .h5 file: {filepath}")
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]

        logger.info(f"Loaded {len(data)} arrays from .h5 file")
        return data

    def validate_data_type(self, data: np.ndarray) -> None:
        """
        验证数据类型

        Args:
            data: 数据数组

        Raises:
            ValueError: 如果数据类型无效
        """
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"Data must be numeric, got dtype: {data.dtype}")

    def validate_dimensions(self, y: np.ndarray, A: np.ndarray) -> None:
        """
        验证数据维度

        Args:
            y: 测量向量
            A: 测量矩阵

        Raises:
            ValueError: 如果维度不匹配
        """
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape: {y.shape}")

        if A.ndim != 2:
            raise ValueError(f"A must be 2-dimensional, got shape: {A.shape}")

        if y.shape[0] != A.shape[0]:
            raise ValueError(
                f"Dimension mismatch: y.shape[0]={y.shape[0]} != "
                f"A.shape[0]={A.shape[0]}"
            )

    def check_memory_limit(self, shape: Tuple, dtype: np.dtype) -> None:
        """
        检查内存限制

        Args:
            shape: 数据形状
            dtype: 数据类型

        Raises:
            MemoryError: 如果超过内存限制
        """
        # 计算所需内存（MB）
        itemsize = np.dtype(dtype).itemsize
        total_size = np.prod(shape) * itemsize
        size_mb = total_size / (1024 * 1024)

        if size_mb > self.max_memory_mb:
            raise MemoryError(
                f"Data size ({size_mb:.1f} MB) exceeds memory limit "
                f"({self.max_memory_mb:.1f} MB)"
            )

    def get_data_info(self, filepath: str) -> Dict[str, Any]:
        """
        获取数据信息（不加载完整数据）

        Args:
            filepath: 文件路径

        Returns:
            数据信息字典
        """
        fmt = self.detect_format(filepath)

        if fmt == "npy":
            # 使用mmap模式只读取元数据
            data = np.load(filepath, mmap_mode='r')
            shape = data.shape
            dtype = data.dtype
            size = data.size
            memory_mb = data.nbytes / (1024 * 1024)

            # 预览前10个元素
            if data.ndim == 1:
                preview = data[:min(10, len(data))].tolist()
            else:
                preview = data.flat[:min(10, data.size)].tolist()

            return {
                "shape": shape,
                "dtype": str(dtype),
                "size": size,
                "memory_mb": f"{memory_mb:.2f}",
                "preview": preview
            }
        else:
            # 其他格式需要加载才能获取信息
            return {"message": "Load file to see details"}

    def load_data(self, y_file: str, A_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        加载数据文件

        Args:
            y_file: y向量文件路径
            A_file: A矩阵文件路径（可选）

        Returns:
            (y, A) 元组，如果A_file为None则A为None

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 格式不支持或数据无效
            MemoryError: 超过内存限制
        """
        # 加载y
        y_format = self.detect_format(y_file)

        if y_format == "npy":
            y = self.load_npy_file(y_file)
        elif y_format == "mat":
            data = self.load_mat_file(y_file)
            # 假设.mat文件中有'y'键
            if 'y' not in data:
                raise ValueError("No 'y' variable found in .mat file")
            y = data['y'].flatten()
        elif y_format == "csv":
            y = self.load_csv_file(y_file)
            if y.ndim == 2:
                y = y.flatten()
        elif y_format == "h5":
            data = self.load_h5_file(y_file)
            if 'y' not in data:
                raise ValueError("No 'y' dataset found in .h5 file")
            y = data['y'].flatten()

        # 验证y的数据类型
        self.validate_data_type(y)

        # 如果没有A文件，只返回y
        if A_file is None:
            return y, None

        # 加载A
        A_format = self.detect_format(A_file)

        if A_format == "npy":
            A = self.load_npy_file(A_file)
        elif A_format == "mat":
            data = self.load_mat_file(A_file)
            if 'A' not in data:
                raise ValueError("No 'A' variable found in .mat file")
            A = data['A']
        elif A_format == "csv":
            A = self.load_csv_file(A_file)
        elif A_format == "h5":
            data = self.load_h5_file(A_file)
            if 'A' not in data:
                raise ValueError("No 'A' dataset found in .h5 file")
            A = data['A']

        # 验证A的数据类型
        self.validate_data_type(A)

        # 验证维度
        self.validate_dimensions(y, A)

        # 检查内存限制
        self.check_memory_limit(A.shape, A.dtype)

        logger.info(f"Successfully loaded data: y{y.shape}, A{A.shape}")

        return y, A
