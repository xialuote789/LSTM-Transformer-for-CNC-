"""
**Data Preprocessing and Data Loader**
**Functions:**
1. Data loading and preprocessing
2. Sliding window segmentation
3. Data normalization
4. Handling class imbalance
5. Creating PyTorch datasets

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class CNCDataPreprocessor:
    """CNC数据预处理器"""

    def __init__(self, data_path, window_size=2000, stride=1000,
                 test_size=0.2, val_size=0.15, random_state=42):
        """
        Args:
            data_path: 数据路径
            window_size: 滑动窗口大小（2000 = 1秒 @ 2kHz）
            stride: 滑动步长（1000 = 0.5秒重叠）
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        """
        self.data_path = Path(data_path)
        self.window_size = window_size
        self.stride = stride
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # 数据存储
        self.raw_data = []
        self.raw_labels = []
        self.file_info = []

        # 预处理后的数据
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # 标准化器
        self.scaler = None

    def load_raw_data(self, max_files_per_class=None, verbose=True):
        """加载原始数据"""
        if verbose:
            print("\n" + "="*80)
            print("加载原始数据")
            print("="*80)

        good_count = 0
        bad_count = 0
        total_samples = 0

        # 遍历所有文件
        for machine_dir in self.data_path.iterdir():
            if not machine_dir.is_dir():
                continue

            for process_dir in machine_dir.iterdir():
                if not process_dir.is_dir():
                    continue

                # 加载good数据
                good_path = process_dir / "good"
                if good_path.exists():
                    good_files = list(good_path.glob("*.h5"))
                    if max_files_per_class:
                        good_files = good_files[:max_files_per_class]

                    for h5_file in tqdm(good_files,
                                      desc=f"加载 {machine_dir.name}/{process_dir.name}/good"):
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                keys = list(f.keys())
                                data = f['vibration_data' if 'vibration_data' in keys else keys[0]][:]

                                self.raw_data.append(data)
                                self.raw_labels.append(0)  # 0 = good
                                self.file_info.append({
                                    'machine': machine_dir.name,
                                    'process': process_dir.name,
                                    'label': 'good',
                                    'file': h5_file.name,
                                    'shape': data.shape
                                })
                                good_count += 1
                                total_samples += len(data)
                        except Exception as e:
                            print(f"  ⚠️ 加载失败: {h5_file.name} - {e}")

                # 加载bad数据
                bad_path = process_dir / "bad"
                if bad_path.exists():
                    bad_files = list(bad_path.glob("*.h5"))
                    if max_files_per_class:
                        bad_files = bad_files[:max_files_per_class]

                    for h5_file in tqdm(bad_files,
                                      desc=f"加载 {machine_dir.name}/{process_dir.name}/bad"):
                        try:
                            with h5py.File(h5_file, 'r') as f:
                                keys = list(f.keys())
                                data = f['vibration_data' if 'vibration_data' in keys else keys[0]][:]

                                self.raw_data.append(data)
                                self.raw_labels.append(1)  # 1 = bad
                                self.file_info.append({
                                    'machine': machine_dir.name,
                                    'process': process_dir.name,
                                    'label': 'bad',
                                    'file': h5_file.name,
                                    'shape': data.shape
                                })
                                bad_count += 1
                                total_samples += len(data)
                        except Exception as e:
                            print(f"  ⚠️ 加载失败: {h5_file.name} - {e}")

        if verbose:
            print(f"\n✓ 加载完成:")
            print(f"  - 正常文件: {good_count}")
            print(f"  - 异常文件: {bad_count}")
            print(f"  - 总文件数: {good_count + bad_count}")
            print(f"  - 总采样点: {total_samples:,}")
            print(f"  - 类别比例: 1:{good_count/max(bad_count, 1):.1f} (异常:正常)")

        return self.raw_data, self.raw_labels

    def create_sliding_windows(self, verbose=True):
        """创建滑动窗口"""
        if verbose:
            print("\n" + "="*80)
            print("创建滑动窗口")
            print("="*80)

        windows = []
        labels = []
        window_info = []

        for idx, (data, label) in enumerate(zip(self.raw_data, self.raw_labels)):
            # 计算可以创建的窗口数
            n_windows = (len(data) - self.window_size) // self.stride + 1

            for i in range(n_windows):
                start = i * self.stride
                end = start + self.window_size
                window = data[start:end]

                windows.append(window)
                labels.append(label)
                window_info.append({
                    'file_idx': idx,
                    'window_idx': i,
                    'start': start,
                    'end': end
                })

        windows = np.array(windows, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if verbose:
            print(f"✓ 窗口创建完成:")
            print(f"  - 总窗口数: {len(windows)}")
            print(f"  - 窗口形状: {windows[0].shape}")
            print(f"  - 正常窗口: {np.sum(labels == 0)}")
            print(f"  - 异常窗口: {np.sum(labels == 1)}")
            print(f"  - 窗口比例: 1:{np.sum(labels == 0)/max(np.sum(labels == 1), 1):.1f} (异常:正常)")

        return windows, labels, window_info

    def normalize_data(self, X_train, X_val, X_test, method='standard'):
        """数据标准化"""
        print("\n" + "="*80)
        print(f"数据标准化 (方法: {method})")
        print("="*80)

        # 选择标准化方法
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"未知的标准化方法: {method}")

        # 保存原始形状
        train_shape = X_train.shape
        val_shape = X_val.shape
        test_shape = X_test.shape

        # 展平数据进行标准化
        X_train_flat = X_train.reshape(-1, 3)
        X_val_flat = X_val.reshape(-1, 3)
        X_test_flat = X_test.reshape(-1, 3)

        # 拟合并转换
        X_train_normalized = self.scaler.fit_transform(X_train_flat)
        X_val_normalized = self.scaler.transform(X_val_flat)
        X_test_normalized = self.scaler.transform(X_test_flat)

        # 恢复原始形状
        X_train_normalized = X_train_normalized.reshape(train_shape)
        X_val_normalized = X_val_normalized.reshape(val_shape)
        X_test_normalized = X_test_normalized.reshape(test_shape)

        print(f"✓ 标准化完成")
        print(f"  - 训练集均值: {self.scaler.mean_}")
        print(f"  - 训练集标准差: {self.scaler.scale_}")

        return X_train_normalized, X_val_normalized, X_test_normalized

    def split_data(self, windows, labels, verbose=True):
        """分割数据集"""
        if verbose:
            print("\n" + "="*80)
            print("数据集分割")
            print("="*80)

        # 第一次分割：训练+验证 vs 测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            windows, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels  # 保持类别比例
        )

        # 第二次分割：训练 vs 验证
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )

        if verbose:
            print(f"✓ 数据集分割完成:")
            print(f"  - 训练集: {len(X_train)} 样本 ({len(X_train)/len(windows)*100:.1f}%)")
            print(f"    • 正常: {np.sum(y_train == 0)}, 异常: {np.sum(y_train == 1)}")
            print(f"  - 验证集: {len(X_val)} 样本 ({len(X_val)/len(windows)*100:.1f}%)")
            print(f"    • 正常: {np.sum(y_val == 0)}, 异常: {np.sum(y_val == 1)}")
            print(f"  - 测试集: {len(X_test)} 样本 ({len(X_test)/len(windows)*100:.1f}%)")
            print(f"    • 正常: {np.sum(y_test == 0)}, 异常: {np.sum(y_test == 1)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess(self, max_files_per_class=None, normalize_method='standard'):
        """完整的预处理流程"""
        print("\n" + "="*80)
        print("开始完整预处理流程")
        print("="*80)

        # 1. 加载原始数据
        self.load_raw_data(max_files_per_class=max_files_per_class)

        # 2. 创建滑动窗口
        windows, labels, window_info = self.create_sliding_windows()

        # 3. 分割数据集
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(windows, labels)

        # 4. 数据标准化
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test,
                                                     method=normalize_method)

        # 保存处理后的数据
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print("\n" + "="*80)
        print("预处理完成！")
        print("="*80)

        return (self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test)

    def save_preprocessed_data(self, save_path):
        """保存预处理后的数据"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存数据
        np.save(save_path / 'X_train.npy', self.X_train)
        np.save(save_path / 'X_val.npy', self.X_val)
        np.save(save_path / 'X_test.npy', self.X_test)
        np.save(save_path / 'y_train.npy', self.y_train)
        np.save(save_path / 'y_val.npy', self.y_val)
        np.save(save_path / 'y_test.npy', self.y_test)

        # 保存标准化器
        with open(save_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"✓ 数据已保存到: {save_path}")

class CNCDataset(Dataset):
    """PyTorch数据集类"""

    def __init__(self, X, y, transform=None):
        """
        Args:
            X: 特征数据 (样本数, 窗口大小, 3)
            y: 标签 (样本数,)
            transform: 数据增强
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test,
                       batch_size=32, num_workers=2, balance_training=True):
    """创建数据加载器"""
    print("\n" + "="*80)
    print("创建数据加载器")
    print("="*80)

    # 创建数据集
    train_dataset = CNCDataset(X_train, y_train)
    val_dataset = CNCDataset(X_val, y_val)
    test_dataset = CNCDataset(X_test, y_test)

    # 处理类别不平衡（仅训练集）
    train_sampler = None
    if balance_training:
        # 计算每个样本的权重
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sample_weights = torch.DoubleTensor(sample_weights)

        # 创建加权采样器
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"✓ 使用加权采样器平衡训练数据")
        print(f"  - 类别权重: {class_weights}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✓ 数据加载器创建完成:")
    print(f"  - 批量大小: {batch_size}")
    print(f"  - 训练批次: {len(train_loader)}")
    print(f"  - 验证批次: {len(val_loader)}")
    print(f"  - 测试批次: {len(test_loader)}")

    return train_loader, val_loader, test_loader

def visualize_preprocessed_data(X_train, y_train, num_samples=4):
    """可视化预处理后的数据"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*3))

    # 随机选择样本
    indices = np.random.choice(len(X_train), num_samples, replace=False)

    for i, idx in enumerate(indices):
        sample = X_train[idx]
        label = y_train[idx]
        label_text = "Normal" if label == 0 else "Anomaly"

        # 绘制三个轴
        for j, axis_name in enumerate(['X-axis', 'Y-axis', 'Z-axis']):
            axes[i, j].plot(sample[:, j], linewidth=0.5)
            axes[i, j].set_title(f'Sample {idx} - {label_text} - {axis_name}')
            axes[i, j].grid(True, alpha=0.3)

            if i == num_samples - 1:
                axes[i, j].set_xlabel('Time steps')
            axes[i, j].set_ylabel('Normalized value')

    plt.suptitle('Preprocessed Data Samples', fontsize=16)
    plt.tight_layout()
    plt.show()

# 主函数
def run_module2(data_path, save_path=None, max_files_per_class=10):
    """运行模块2：数据预处理"""

    print("="*80)
    print("模块2：数据预处理与加载器")
    print("="*80)

    # 1. 初始化预处理器
    preprocessor = CNCDataPreprocessor(
        data_path=data_path,
        window_size=2000,  # 1秒 @ 2kHz
        stride=1000,       # 0.5秒重叠
        test_size=0.2,
        val_size=0.15
    )

    # 2. 执行预处理
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(
        max_files_per_class=max_files_per_class,
        normalize_method='standard'
    )

    # 3. 保存预处理数据（可选）
    if save_path:
        preprocessor.save_preprocessed_data(save_path)

    # 4. 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=32,
        balance_training=True  # 使用加权采样处理不平衡
    )

    # 5. 可视化样本
    print("\n可视化预处理后的数据...")
    visualize_preprocessed_data(X_train, y_train, num_samples=4)

    print("\n" + "="*80)
    print("模块2完成！")
    print("="*80)

    return preprocessor, train_loader, val_loader, test_loader

# 测试运行
if __name__ == "__main__":
    # Google Colab路径
    data_path = "/content/drive/MyDrive/CNC_Machining/data"
    save_path = "/content/drive/MyDrive/CNC_preprocessed"

    # 运行预处理（使用较少文件进行测试）
    preprocessor, train_loader, val_loader, test_loader = run_module2(
        data_path=data_path,
        save_path=save_path,
        max_files_per_class=5  # 先用少量数据测试
    )