import numpy as np
import os
import pytest
from src.data_processing import load_npz_data, format_for_model


# --- 1. 测试 .npz 数据加载与合并 ---
def test_load_npz_data(tmp_path):
    """
    测试是否能正确扫描文件夹并合并多个 .npz 文件。
    使用 tmp_path 创建临时虚拟数据进行测试。
    """
    # 在临时文件夹创建两个模拟的 npz 文件
    d = tmp_path / "eeg_subfolder"
    d.mkdir()

    # 模拟数据：每个文件 10 个样本，每个样本 3000 点
    file1 = d / "test1.npz"
    np.savez(file1, x=np.random.rand(10, 3000), y=np.zeros(10))

    file2 = d / "test2.npz"
    np.savez(file2, x=np.random.rand(5, 3000), y=np.ones(5))

    # 执行加载
    X, y = load_npz_data(str(d))

    # 断言：合并后应该是 15 个样本
    assert X.shape == (15, 3000), "合并后的特征维度不正确"
    assert y.shape == (15,), "合并后的标签维度不正确"
    assert len(np.unique(y)) == 2, "标签合并错误"


# --- 2. 测试数据格式化 (Conv1D 适配) ---
def test_format_for_model():
    """
    验证数据是否被正确扩展为 (Samples, 3000, 1)
    以及标签是否转为了 One-hot 编码。
    """
    # 模拟原始输入：(20, 3000)
    X_raw = np.random.rand(20, 3000)
    y_raw = np.array([0, 1, 2, 3, 4] * 4)  # 5类标签

    # 执行格式化
    X, y_one_hot = format_for_model(X_raw, y_raw, num_classes=5)

    # 断言 1：特征维度必须是三维 (Samples, Time, 1)
    assert len(X.shape) == 3, "特征应该是三维的"
    assert X.shape[2] == 1, "最后一个维度应该是 1 (单通道)"

    # 断言 2：标签必须是 One-hot 编码 (Samples, 5)
    assert y_one_hot.shape == (20, 5), "One-hot 编码后的维度不正确"
    assert np.all(np.sum(y_one_hot, axis=1) == 1), "One-hot 编码逻辑错误"


# --- 3. 测试空路径异常 ---
def test_load_npz_invalid_path():
    """
    确保路径不存在时会抛出错误。
    """
    with pytest.raises(FileNotFoundError):
        load_npz_data("non_existent_path_123")