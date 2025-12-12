import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gaussian_kde # 引入 gaussian_kde 和 norm
import warnings

# --- 关键导入：保持原有逻辑 ---
# 假设 plot1 模块和其中的函数在您的环境中可用
# from plot1 import load_and_preprocess_data, process_mid_price_from_depth
# --------------------------------

# ---------------------- 关键优化：全局参数设置 ----------------------
plt.rcParams.update({
    # 1. 提高默认显示DPI
    'figure.dpi': 150,
    'savefig.dpi': 300, 
    
    # 2. 开启抗锯齿
    'text.antialiased': True,
    'axes.linewidth': 1,
    'lines.antialiased': True,
    
    # 3. 使用矢量字体
    'font.family': ['SimHei', 'Times New Roman'],
    'font.sans-serif': ['SimHei', 'Times New Roman'],
    
    # 4. 解决负号显示问题
    'axes.unicode_minus': False,
    
    # 5. 微调字体大小
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

# ---------------------- 1. 数据加载和预处理 (您的原有代码) ----------------------
try:
    fundamental_df = pd.read_excel('data\\ETH1.xlsx', sheet_name='1110')
except FileNotFoundError:
    print("错误：无法找到 'data\\ETH1.xlsx' 文件。使用模拟数据代替。")
    # --- 替代模拟数据 ---
    dates = pd.to_datetime(pd.date_range(start='2023-01-01 09:00:00', periods=1000, freq='10s'))
    np.random.seed(42)
    price_series_values = 100 + np.cumsum(np.random.randn(1000) * 0.1)
    price_series = pd.Series(price_series_values, index=dates)
    mid_price_minutely_list = [
        pd.Series(price_series_values * (1 + np.random.randn(1000) * 0.001), index=dates)
    ]
    # ----------------------
else:
    # 您的原始数据处理逻辑
    fundamental_df = fundamental_df[fundamental_df['Dates'] != 'Dates']
    fundamental_df['Dates'] = fundamental_df['Dates'].str.replace("'", "", regex=False)
    fundamental_df['Dates'] = fundamental_df['Dates'].str.replace(".000000", "", regex=False)
    fundamental_df['Dates'] = pd.to_datetime(fundamental_df['Dates'])
    price_series = pd.Series(fundamental_df['Price'].values, index=fundamental_df['Dates'])

    # 模拟数据加载逻辑 (需要 plot1.py 中的函数支持)
    try:
        from plot import load_and_preprocess_data, process_mid_price_from_depth
        paths = [r'log\rmsc04_two_hour\SNAPSHOT_AGENT.bz2'] 
        labels = ['Hybrid'] 
        mid_price_minutely_list = []

        for path in paths:
            raw_data = load_and_preprocess_data(path)
            mid_price_data_df = process_mid_price_from_depth(raw_data)
            
            if not pd.api.types.is_datetime64_any_dtype(mid_price_data_df.index):
                mid_price_data_df.index = pd.to_datetime(mid_price_data_df.index)
            
            possible_mid_price_cols = ['mid_price', 'middle_price', 'mid', '中间价', '均价']
            mid_price_col = None
            for col in possible_mid_price_cols:
                if col in mid_price_data_df.columns:
                    mid_price_col = col
                    break
            
            if mid_price_col is None:
                numeric_cols = mid_price_data_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) == 1:
                    mid_price_col = numeric_cols[0]
                    print(f"自动识别中间价列：{mid_price_col}")
                else:
                    raise ValueError(f"请指定中间价列名，可用的数值列：{numeric_cols}")
            
            aggregation_method = 'mean'
            # 聚合到10秒
            mid_price_minutely = mid_price_data_df[mid_price_col].resample('10s').agg(aggregation_method).dropna()
            mid_price_minutely_list.append(mid_price_minutely)
            
            print(f"\n路径 {path} 按10秒{aggregation_method}聚合后的数据形状：", mid_price_minutely.shape)
            
    except ImportError:
        print("警告：无法导入 'plot1.py' 中的函数。使用模拟数据代替模拟结果。")
        # 避免报错，如果无法加载模拟数据，则使用替代模拟数据
        if 'mid_price_minutely_list' not in locals() or not mid_price_minutely_list:
            dates = price_series.index
            np.random.seed(42)
            mid_price_minutely_list = [
                pd.Series(price_series.values * (1 + np.random.randn(len(dates)) * 0.001), index=dates)
            ]


# ---------------------- 2. 绘图：价格对比图 (您的原有代码) ----------------------

fig1, ax1 = plt.subplots(1, 1, figsize=(5, 4)) 

color_history = '#2E86AB'
color_simulated = '#F18F01' 

# 绘制Hybrid模拟曲线
mid_price_data = mid_price_minutely_list[0]
ax1.plot(mid_price_data.index, mid_price_data.values,
        color=color_simulated, linewidth=0.5, alpha=1,
        label=f'Simulated (Hybrid)', antialiased=True)

# 绘制历史曲线
ax1.plot(price_series.index, price_series.values, 
        color=color_history, linewidth=0.5, alpha=1,
        label='History', antialiased=True)

# 子图样式设置
ax1.set_xlabel('Time')
ax1.set_ylabel('Mid-price（USDT）')
ax1.set_title('ETH Price Comparison')

# 显示四周边框
ax1.spines['top'].set_visible(True) 
ax1.spines['right'].set_visible(True) 
ax1.spines['left'].set_linewidth(0.6)
ax1.spines['bottom'].set_linewidth(0.6)
ax1.spines['top'].set_linewidth(0.6)
ax1.spines['right'].set_linewidth(0.6)

# 网格优化
ax1.grid(True, alpha=0.4, linestyle=':', linewidth=1.2)

# 图例设置 (右上角)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
          framealpha=1.0)

# x轴日期显示优化
fig1.autofmt_xdate()

# 缩小y轴范围
all_y_data = np.concatenate([
    price_series.values,
    mid_price_minutely_list[0].values
])
y_min = all_y_data.min() * 0.99
y_max = all_y_data.max() * 1.01
ax1.set_ylim(y_min, y_max)

plt.tight_layout()

# 保存价格对比图
plt.savefig(
    'ETH价格_Hybrid模拟中间价对比图_高清版.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.savefig('ETH价格_Hybrid对比图_矢量版.svg', format='svg', bbox_inches='tight')


# ---------------------- 3. 绘图：收益率 CDF 对比图 (KDE 平滑 & 标准正态基准) ----------------------
# --- 定义平滑 CDF 函数 (已修正) ---
def calculate_smooth_cdf(data, x_range, bandwidth='scott'):
    """使用高斯核密度估计计算平滑的 CDF。"""
    
    # 强制将数据转换为标准的 float64 NumPy 数组并移除所有 NaN 值
    data = np.asarray(data, dtype=np.float64)
    data = data[~np.isnan(data)] 
    
    # 检查数据点是否足够
    if len(data) < 2:
        warnings.warn("数据点少于 2 个，无法计算 KDE-CDF。")
        return np.zeros_like(x_range)
    
    # 1. 计算核密度估计 
    kde = gaussian_kde(data, bandwidth) 
    
    # 2. 对 KDE 进行积分，得到平滑 CDF
    cdf_values = np.array([kde.integrate_box_1d(-np.inf, x) for x in x_range])
    
    # 确保 CDF 严格从 0 延伸到 1 (处理积分误差)
    cdf_values[cdf_values < 0] = 0
    cdf_values[cdf_values > 1] = 1
    
    return cdf_values


# --- 3.1 收益率计算与标准化 ---
# 模拟收益率 (百分比)
simulated_returns = mid_price_minutely_list[0].pct_change().dropna() * 100

# 核心修正：对模拟收益率进行标准化 (Z-score)
mu_sim = simulated_returns.mean()
sigma_sim = simulated_returns.std()

# 检查标准差，避免除以零
if sigma_sim == 0:
    print("警告：模拟收益率标准差为零，无法标准化。")
    standardized_returns = simulated_returns - mu_sim
else:
    standardized_returns = (simulated_returns - mu_sim) / sigma_sim

# --- 3.2 绘图初始化 ---
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5)) 

color_normal_cdf = '#2E86AB'  
color_simulated_cdf = '#F18F01' 


# --- 3.3 确定 X 轴范围 (以标准差单位为主，例如 [-5, 5]) ---
# 标准化后的数据范围大致在 -4 到 4 之间，我们使用 [-5, 5] 作为绘图范围
x_min_plot = -5
x_max_plot = 5 
x_plot = np.linspace(x_min_plot, x_max_plot, 1000)


# --- 3.4 绘制 Standardized Simulated Returns KDE-CDF ---
# 使用标准化后的收益率数据
smooth_cdf_sim_standardized = calculate_smooth_cdf(standardized_returns, x_plot)
ax2.plot(x_plot, smooth_cdf_sim_standardized, 
        color=color_simulated_cdf, 
        linewidth=1.5, 
        label='Simulated Historcal Return', 
        antialiased=True)


# --- 3.5 绘制 标准正态分布 CDF (Standard Normal N(0, 1)) ---
# 标准正态分布的 CDF 值：mu=0, sigma=1
cdf_standard_normal = norm.cdf(x_plot, loc=0, scale=1) 
ax2.plot(x_plot, cdf_standard_normal, 
        color=color_normal_cdf, 
        linestyle='--', 
        linewidth=1.5, 
        label=f'Normal Reference ($\mu=0, \sigma=1$)', 
        alpha=0.8, 
        antialiased=True)


# --- 图表样式设置 ---
ax2.set_title('CDF of Returns 20251110 ETH')
# ax2.set_xlabel('Returns (Standard Deviation Units)') # 横坐标单位发生变化
# ax2.set_ylabel('Cumulative Probability')

ax2.tick_params(direction='in', which='both')
ax2.grid(True, alpha=0.5, linestyle=':', linewidth=0.8)

ax2.set_ylim(0, 1.05) 
ax2.set_xlim(x_min_plot, x_max_plot)

# 边框和图例
ax2.spines['top'].set_visible(True) 
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_linewidth(0.6)
ax2.spines['bottom'].set_linewidth(0.6)
ax2.spines['top'].set_linewidth(0.6)
ax2.spines['right'].set_linewidth(0.6)

ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
          framealpha=1.0)

plt.tight_layout()

# 保存CDF对比图
filename_cdf = 'ETH标准化收益率KDE_CDF对比图'
plt.savefig(f'{filename_cdf}_高清版.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{filename_cdf}_矢量版.svg', format='svg', bbox_inches='tight')

# 可选：显示所有图片
plt.show()