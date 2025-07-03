import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

def parse_data_from_string(data_string):
    """从字符串解析数据（模拟文件读取）"""
    lines = data_string.strip().split('\n')
    all_data = []
    
    for line in lines:
        if line.strip() and not line.startswith('%'):
            parts = line.split()
            if len(parts) >= 8:
                try:
                    freq = float(parts[0])
                    alpha = float(parts[1])
                    delta = float(parts[2])
                    f1dot = float(parts[3])
                    f2dot = float(parts[4])
                    nc = float(parts[5])
                    twoF = float(parts[6])
                    twoFr = float(parts[7])
                    all_data.append([freq, f1dot, f2dot, twoFr])
                except ValueError:
                    continue
    
    return np.array(all_data)

def read_data_from_file(output_file):
    """从文件读取数据的函数"""
    with open(output_file, 'r') as f:
        lines = f.readlines()

    # Parse all data points
    all_data = []
    for line in lines:
        if line.strip() and not line.startswith('%'):
            parts = line.split()
            if len(parts) >= 8:
                try:
                    freq = float(parts[0])
                    alpha = float(parts[1])
                    delta = float(parts[2])
                    f1dot = float(parts[3])
                    f2dot = float(parts[4])
                    nc = float(parts[5])
                    twoF = float(parts[6])
                    twoFr = float(parts[7])
                    all_data.append([freq, f1dot, f2dot, twoFr])
                except ValueError:
                    continue
    
    return np.array(all_data)

class RobustInterpolator3D:
    """
    改进的三维插值器类，解决数值精度问题
    """
    
    def __init__(self, data, normalization_method='standard', fallback_to_1d=True):
        """
        初始化插值器
        
        参数:
        data: shape (n, 4) 的数组，列为 [freq, f1dot, f2dot, 2Fr]
        normalization_method: 'standard', 'minmax', 'robust', 或 None
        fallback_to_1d: 如果3D插值失败，是否回退到1D插值
        """
        self.data = data
        self.freq = data[:, 0].copy()
        self.f1dot = data[:, 1].copy()
        self.f2dot = data[:, 2].copy()
        self.twoFr = data[:, 3].copy()
        
        # 保存原始数据统计信息
        self.freq_stats = {'mean': np.mean(self.freq), 'std': np.std(self.freq), 
                          'min': np.min(self.freq), 'max': np.max(self.freq)}
        self.f1dot_stats = {'mean': np.mean(self.f1dot), 'std': np.std(self.f1dot),
                           'min': np.min(self.f1dot), 'max': np.max(self.f1dot)}
        self.f2dot_stats = {'mean': np.mean(self.f2dot), 'std': np.std(self.f2dot),
                           'min': np.min(self.f2dot), 'max': np.max(self.f2dot)}
        
        self.normalization_method = normalization_method
        self.fallback_to_1d = fallback_to_1d
        self.use_3d_interpolation = True
        
        # 数据归一化
        self._normalize_data()
        
        # 分析数据特征
        self._analyze_data_characteristics()
        
        # 尝试建立3D插值器
        self._setup_interpolation()
        
        print(f"频率范围: {self.freq_stats['min']:.10f} - {self.freq_stats['max']:.10f}")
        print(f"f1dot范围: {self.f1dot_stats['min']:.2e} - {self.f1dot_stats['max']:.2e}")
        print(f"f2dot范围: {self.f2dot_stats['min']:.2e} - {self.f2dot_stats['max']:.2e}")
        print(f"2Fr范围: {np.min(self.twoFr):.2f} - {np.max(self.twoFr):.2f}")
        
    def _normalize_data(self):
        """数据归一化处理"""
        if self.normalization_method is None:
            self.freq_norm = self.freq
            self.f1dot_norm = self.f1dot
            self.f2dot_norm = self.f2dot
            return
            
        # 计算偏差（相对于均值）
        self.df_original = self.freq - self.freq_stats['mean']
        self.df1_original = self.f1dot - self.f1dot_stats['mean']
        self.df2_original = self.f2dot - self.f2dot_stats['mean']
        
        if self.normalization_method == 'standard':
            # 标准化（零均值，单位方差）
            self.freq_norm = (self.freq - self.freq_stats['mean']) / max(self.freq_stats['std'], 1e-15)
            self.f1dot_norm = (self.f1dot - self.f1dot_stats['mean']) / max(self.f1dot_stats['std'], 1e-15)
            self.f2dot_norm = (self.f2dot - self.f2dot_stats['mean']) / max(self.f2dot_stats['std'], 1e-15)
            
        elif self.normalization_method == 'minmax':
            # 最小-最大归一化到 [-1, 1]
            freq_range = max(self.freq_stats['max'] - self.freq_stats['min'], 1e-15)
            f1dot_range = max(self.f1dot_stats['max'] - self.f1dot_stats['min'], 1e-15)
            f2dot_range = max(self.f2dot_stats['max'] - self.f2dot_stats['min'], 1e-15)
            
            self.freq_norm = 2 * (self.freq - self.freq_stats['min']) / freq_range - 1
            self.f1dot_norm = 2 * (self.f1dot - self.f1dot_stats['min']) / f1dot_range - 1
            self.f2dot_norm = 2 * (self.f2dot - self.f2dot_stats['min']) / f2dot_range - 1
            
        elif self.normalization_method == 'robust':
            # 鲁棒归一化（使用中位数和四分位数）
            freq_median = np.median(self.freq)
            f1dot_median = np.median(self.f1dot)
            f2dot_median = np.median(self.f2dot)
            
            freq_mad = np.median(np.abs(self.freq - freq_median))
            f1dot_mad = np.median(np.abs(self.f1dot - f1dot_median))
            f2dot_mad = np.median(np.abs(self.f2dot - f2dot_median))
            
            self.freq_norm = (self.freq - freq_median) / max(freq_mad, 1e-15)
            self.f1dot_norm = (self.f1dot - f1dot_median) / max(f1dot_mad, 1e-15)
            self.f2dot_norm = (self.f2dot - f2dot_median) / max(f2dot_mad, 1e-15)
    
    def _analyze_data_characteristics(self):
        """分析数据特征，判断是否适合3D插值"""
        # 计算各维度的变异系数
        cv_freq = self.freq_stats['std'] / abs(self.freq_stats['mean']) if self.freq_stats['mean'] != 0 else 0
        cv_f1dot = self.f1dot_stats['std'] / abs(self.f1dot_stats['mean']) if self.f1dot_stats['mean'] != 0 else 0
        cv_f2dot = self.f2dot_stats['std'] / abs(self.f2dot_stats['mean']) if self.f2dot_stats['mean'] != 0 else 0
        
        # 判断哪个维度变化最大
        variations = {'freq': cv_freq, 'f1dot': cv_f1dot, 'f2dot': cv_f2dot}
        self.dominant_dimension = max(variations, key=variations.get)
        
        print(f"数据特征分析:")
        print(f"  频率变异系数: {cv_freq:.2e}")
        print(f"  f1dot变异系数: {cv_f1dot:.2e}")
        print(f"  f2dot变异系数: {cv_f2dot:.2e}")
        print(f"  主导维度: {self.dominant_dimension}")
        
    def _setup_interpolation(self):
        """设置插值方法"""
        try:
            # 尝试3D插值（使用归一化数据）
            self.points_norm = np.column_stack((self.freq_norm, self.f1dot_norm, self.f2dot_norm))
            
            # 检查是否存在重复点
            unique_points, inverse_indices = np.unique(self.points_norm, axis=0, return_inverse=True)
            if len(unique_points) < len(self.points_norm):
                print(f"警告: 发现 {len(self.points_norm) - len(unique_points)} 个重复点，将进行合并")
                # 对重复点的2Fr值取平均
                unique_values = np.zeros(len(unique_points))
                for i in range(len(unique_points)):
                    mask = inverse_indices == i
                    unique_values[i] = np.mean(self.twoFr[mask])
                
                self.points_norm = unique_points
                self.values_norm = unique_values
            else:
                self.values_norm = self.twoFr
            
            # 测试griddata是否可用
            test_point = np.mean(self.points_norm, axis=0).reshape(1, -1)
            test_result = griddata(self.points_norm, self.values_norm, test_point, 
                                 method='linear', fill_value=np.nan)
            
            if np.isnan(test_result[0]):
                raise ValueError("3D插值返回NaN")
                
            print("✓ 3D插值设置成功")
            
        except Exception as e:
            print(f"⚠ 3D插值设置失败: {e}")
            if self.fallback_to_1d:
                self._setup_1d_fallback()
            else:
                raise
    
    def _setup_1d_fallback(self):
        """设置1D插值回退方案"""
        self.use_3d_interpolation = False
        
        # 基于主导维度进行1D插值
        if self.dominant_dimension == 'freq':
            self.x_1d = self.freq
        elif self.dominant_dimension == 'f1dot':
            self.x_1d = self.f1dot
        else:  # f2dot
            self.x_1d = self.f2dot
            
        # 排序以便插值
        sort_indices = np.argsort(self.x_1d)
        self.x_1d_sorted = self.x_1d[sort_indices]
        self.y_1d_sorted = self.twoFr[sort_indices]
        
        print(f"✓ 回退到1D插值（基于 {self.dominant_dimension}）")
    
    def interpolate(self, freq_new, f1dot_new, f2dot_new, method='linear'):
        """
        插值函数
        
        参数:
        freq_new, f1dot_new, f2dot_new: 新的坐标点
        method: 插值方法 ('linear', 'nearest', 'cubic')
        
        返回:
        插值得到的 2Fr 值
        """
        if self.use_3d_interpolation:
            return self._interpolate_3d(freq_new, f1dot_new, f2dot_new, method)
        else:
            return self._interpolate_1d(freq_new, f1dot_new, f2dot_new)
    
    def _interpolate_3d(self, freq_new, f1dot_new, f2dot_new, method='linear'):
        """3D插值"""
        # 归一化输入
        if self.normalization_method == 'standard':
            freq_norm = (freq_new - self.freq_stats['mean']) / max(self.freq_stats['std'], 1e-15)
            f1dot_norm = (f1dot_new - self.f1dot_stats['mean']) / max(self.f1dot_stats['std'], 1e-15)
            f2dot_norm = (f2dot_new - self.f2dot_stats['mean']) / max(self.f2dot_stats['std'], 1e-15)
        elif self.normalization_method == 'minmax':
            freq_range = max(self.freq_stats['max'] - self.freq_stats['min'], 1e-15)
            f1dot_range = max(self.f1dot_stats['max'] - self.f1dot_stats['min'], 1e-15)
            f2dot_range = max(self.f2dot_stats['max'] - self.f2dot_stats['min'], 1e-15)
            
            freq_norm = 2 * (freq_new - self.freq_stats['min']) / freq_range - 1
            f1dot_norm = 2 * (f1dot_new - self.f1dot_stats['min']) / f1dot_range - 1
            f2dot_norm = 2 * (f2dot_new - self.f2dot_stats['min']) / f2dot_range - 1
        else:
            freq_norm = freq_new
            f1dot_norm = f1dot_new
            f2dot_norm = f2dot_new
        
        # 准备查询点
        if np.isscalar(freq_new):
            xi = np.array([[freq_norm, f1dot_norm, f2dot_norm]])
        else:
            xi = np.column_stack((freq_norm.flatten(), f1dot_norm.flatten(), f2dot_norm.flatten()))
        
        # 执行插值
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = griddata(self.points_norm, self.values_norm, xi, 
                            method=method, fill_value=np.nan)
        
        return result
    
    def _interpolate_1d(self, freq_new, f1dot_new, f2dot_new):
        """1D插值回退方案"""
        if self.dominant_dimension == 'freq':
            x_new = freq_new
        elif self.dominant_dimension == 'f1dot':
            x_new = f1dot_new
        else:
            x_new = f2dot_new
            
        return np.interp(x_new, self.x_1d_sorted, self.y_1d_sorted)
    
    def interpolate_by_deviation(self, df_new, df1_new, df2_new, method='linear'):
        """
        基于偏差进行插值
        """
        freq_new = self.freq_stats['mean'] + df_new
        f1dot_new = self.f1dot_stats['mean'] + df1_new
        f2dot_new = self.f2dot_stats['mean'] + df2_new
        
        return self.interpolate(freq_new, f1dot_new, f2dot_new, method)

def plot_robust_3d_scatter(interpolator, show_data_points=True, n_sample_points=100):
    """改进的3D散点图绘制函数"""
    fig = plt.figure(figsize=(20, 6))
    
    if interpolator.use_3d_interpolation:
        # 3D插值表面 - 使用归一化坐标
        ax1 = fig.add_subplot(141, projection='3d')
        
        # 创建三维网格
        if interpolator.normalization_method:
            x_range = np.linspace(np.min(interpolator.freq_norm), np.max(interpolator.freq_norm), 20)
            z_range = np.linspace(np.min(interpolator.f2dot_norm), np.max(interpolator.f2dot_norm), 20)
        else:
            x_range = np.linspace(np.min(interpolator.freq), np.max(interpolator.freq), 20)
            z_range = np.linspace(np.min(interpolator.f2dot), np.max(interpolator.f2dot), 20)
            
        X_3d, Z_3d = np.meshgrid(x_range, z_range)
        Y_3d = np.zeros_like(X_3d)  # 固定中间维度
        
        # 反归一化进行插值
        if interpolator.normalization_method == 'standard':
            freq_test = X_3d * interpolator.freq_stats['std'] + interpolator.freq_stats['mean']
            f1dot_test = Y_3d * interpolator.f1dot_stats['std'] + interpolator.f1dot_stats['mean']
            f2dot_test = Z_3d * interpolator.f2dot_stats['std'] + interpolator.f2dot_stats['mean']
        elif interpolator.normalization_method == 'minmax':
            freq_range = interpolator.freq_stats['max'] - interpolator.freq_stats['min']
            f1dot_range = interpolator.f1dot_stats['max'] - interpolator.f1dot_stats['min']
            f2dot_range = interpolator.f2dot_stats['max'] - interpolator.f2dot_stats['min']
            
            freq_test = (X_3d + 1) * freq_range / 2 + interpolator.freq_stats['min']
            f1dot_test = (Y_3d + 1) * f1dot_range / 2 + interpolator.f1dot_stats['min']
            f2dot_test = (Z_3d + 1) * f2dot_range / 2 + interpolator.f2dot_stats['min']
        else:
            freq_test = X_3d
            f1dot_test = Y_3d  
            f2dot_test = Z_3d
        
        # 计算插值表面
        try:
            W_3d = interpolator.interpolate(freq_test, f1dot_test, f2dot_test)
            W_3d = W_3d.reshape(X_3d.shape)
            
            # 绘制表面
            surface = ax1.plot_surface(X_3d, Z_3d, W_3d, cmap='viridis', alpha=0.7)
            
            # 显示数据点
            if show_data_points:
                n_total = len(interpolator.freq)
                indices = np.random.choice(n_total, min(n_sample_points, n_total), replace=False)
                if interpolator.normalization_method:
                    ax1.scatter(interpolator.freq_norm[indices], interpolator.f2dot_norm[indices], 
                               interpolator.twoFr[indices], c=interpolator.twoFr[indices], 
                               cmap='viridis', s=20, alpha=0.8)
                else:
                    ax1.scatter(interpolator.freq[indices], interpolator.f2dot[indices], 
                               interpolator.twoFr[indices], c=interpolator.twoFr[indices], 
                               cmap='viridis', s=20, alpha=0.8)
            
            ax1.set_xlabel('频率 (归一化)' if interpolator.normalization_method else '频率')
            ax1.set_ylabel('f2dot (归一化)' if interpolator.normalization_method else 'f2dot')
            ax1.set_zlabel('2Fr')
            ax1.set_title('3D插值表面')
            
        except Exception as e:
            ax1.text(0.5, 0.5, 0.5, f"3D可视化失败:\n{str(e)}", 
                    transform=ax1.transAxes, ha='center', va='center')
            ax1.set_title('3D插值 - 可视化失败')
    
    else:
        # 1D插值显示
        ax1 = fig.add_subplot(141)
        ax1.plot(interpolator.x_1d_sorted, interpolator.y_1d_sorted, 'b-', linewidth=2)
        
        if show_data_points:
            indices = np.random.choice(len(interpolator.x_1d), min(n_sample_points, len(interpolator.x_1d)), replace=False)
            ax1.scatter(interpolator.x_1d[indices], interpolator.twoFr[indices], 
                       c='red', s=20, alpha=0.7)
        
        ax1.set_xlabel(f'{interpolator.dominant_dimension}')
        ax1.set_ylabel('2Fr')
        ax1.set_title(f'1D插值 (基于 {interpolator.dominant_dimension})')
        ax1.grid(True, alpha=0.3)
    
    # 添加统计信息和使用说明
    ax2 = fig.add_subplot(142)
    ax2.axis('off')
    
    stats_text = f"""插值器信息:

数据点数量: {len(interpolator.data):,}
插值方法: {"3D" if interpolator.use_3d_interpolation else "1D"}
归一化方法: {interpolator.normalization_method or "无"}

数据范围:
  freq:  {interpolator.freq_stats['min']:.6f} - {interpolator.freq_stats['max']:.6f}
  f1dot: {interpolator.f1dot_stats['min']:.2e} - {interpolator.f1dot_stats['max']:.2e}
  f2dot: {interpolator.f2dot_stats['min']:.2e} - {interpolator.f2dot_stats['max']:.2e}
  2Fr:   {np.min(interpolator.twoFr):.0f} - {np.max(interpolator.twoFr):.0f}

状态: ✓ 可用
"""
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))
    
    # 示例用法
    ax3 = fig.add_subplot(143)
    ax3.axis('off')
    
    usage_text = """使用示例:

# 创建鲁棒插值器
interpolator = RobustInterpolator3D(
    data, 
    normalization_method='standard',
    fallback_to_1d=True
)

# 进行插值
result = interpolator.interpolate(
    freq_new=151.5, 
    f1dot_new=-1e-10, 
    f2dot_new=0.0
)

# 或使用偏差插值
result = interpolator.interpolate_by_deviation(
    df_new=0.0, 
    df1_new=0.0, 
    df2_new=1e-19
)

print(f"插值结果: {result}")
"""
    
    ax3.text(0.05, 0.95, usage_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))
    
    # 性能测试
    ax4 = fig.add_subplot(144)
    
    # 测试插值性能
    n_test = 1000
    test_freq = np.random.uniform(interpolator.freq_stats['min'], interpolator.freq_stats['max'], n_test)
    test_f1dot = np.random.uniform(interpolator.f1dot_stats['min'], interpolator.f1dot_stats['max'], n_test)
    test_f2dot = np.random.uniform(interpolator.f2dot_stats['min'], interpolator.f2dot_stats['max'], n_test)
    
    import time
    start_time = time.time()
    test_results = []
    for i in range(min(100, n_test)):  # 测试前100个点
        result = interpolator.interpolate(test_freq[i], test_f1dot[i], test_f2dot[i])
        if not np.isnan(result):
            test_results.append(result)
    
    end_time = time.time()
    
    ax4.hist(test_results, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('插值结果 (2Fr)')
    ax4.set_ylabel('频次')
    ax4.set_title(f'插值测试结果分布\n({len(test_results)}/100 成功)')
    ax4.grid(True, alpha=0.3)
    
    # 添加性能信息
    avg_time = (end_time - start_time) / min(100, n_test) * 1000
    ax4.text(0.02, 0.98, f'平均插值时间: {avg_time:.2f} ms', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# 测试函数
def test_robust_interpolator():
    """测试鲁棒插值器的功能"""
    print("=== 鲁棒插值器测试 ===\n")
    
    # 创建模拟数据（类似你的实际数据）
    np.random.seed(42)
    n_points = 1000
    
    # 模拟你的数据特征：频率变化很小，f2dot是主要变量
    freq_base = 151.5
    freq = freq_base + np.random.normal(0, 1e-6, n_points)
    
    f1dot_base = -1e-10
    f1dot = f1dot_base + np.random.normal(0, 1e-12, n_points)
    
    f2dot = np.random.uniform(-2.5e-18, 2.5e-18, n_points)
    
    # 创建一个与f2dot相关的2Fr值（加一些噪声）
    twoFr = 1e6 + 1e8 * (f2dot / 2.5e-18)**2 + np.random.normal(0, 1e5, n_points)
    
    data = np.column_stack([freq, f1dot, f2dot, twoFr])
    
    print(f"测试数据创建完成: {n_points} 个点")
    print(f"freq范围: {np.min(freq):.10f} - {np.max(freq):.10f}")
    print(f"f1dot范围: {np.min(f1dot):.2e} - {np.max(f1dot):.2e}")  
    print(f"f2dot范围: {np.min(f2dot):.2e} - {np.max(f2dot):.2e}")
    print(f"2Fr范围: {np.min(twoFr):.0f} - {np.max(twoFr):.0f}\n")
    
    # 测试不同的归一化方法
    methods = ['standard', 'minmax', 'robust', None]
    
    for method in methods:
        print(f"--- 测试归一化方法: {method} ---")
        try:
            interpolator = RobustInterpolator3D(data, normalization_method=method)
            
            # 测试插值
            test_result = interpolator.interpolate(freq_base, f1dot_base, 0.0)
            print(f"插值测试: {test_result}")
            
            # 测试偏差插值
            deviation_result = interpolator.interpolate_by_deviation(0.0, 0.0, 1e-19)
            print(f"偏差插值测试: {deviation_result}")
            
        except Exception as e:
            print(f"失败: {e}")
        
        print()
    
    # 返回一个成功的插值器用于可视化
    return RobustInterpolator3D(data, normalization_method='standard')

# 主程序
if __name__ == "__main__":
    # 如果有实际数据文件，取消注释下面的行
    data = read_data_from_file('LAL_example_data/LALSemiCoherentF0F1F2_analytical/dats/semicoh_results.dat')
    interpolator = RobustInterpolator3D(data, normalization_method='standard')
    
    # 使用测试数据
    # interpolator = test_robust_interpolator()
    
    # 绘制结果
    plot_robust_3d_scatter(interpolator)
    
    print("\n=== 解决方案总结 ===")
    print("1. 数据归一化: 解决不同维度数值范围差异巨大的问题")
    print("2. 重复点处理: 检测并合并重复的数据点")
    print("3. 自动回退: 如果3D插值失败，自动使用1D插值")
    print("4. 多种归一化方法: standard, minmax, robust")
    print("5. 鲁棒性改进: 添加错误处理和数值稳定性检查")
    print("\n推荐使用 'standard' 归一化方法处理你的数据。")