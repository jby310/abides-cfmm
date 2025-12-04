import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import warnings

# --- 关键导入：保持原有逻辑 ---
# 假设 plot1 模块和其中的函数在您的环境中可用
# from plot1 import load_and_preprocess_data, process_mid_price_from_depth
# 由于我无法访问 plot1.py，此处只是一个占位符，请确保您环境中 plot1.py 存在。
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
    'font.family': ['Times New Roman', 'SimHei'],
    'font.sans-serif': ['Times New Roman', 'SimHei'],
    
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
        from plot1 import load_and_preprocess_data, process_mid_price_from_depth
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


# ---------------------- 3. 绘图：价格 CDF 对比图 (已修正延伸逻辑) ----------------------

# 获取价格数据
historical_prices = price_series.values
simulated_prices = mid_price_minutely_list[0].values

# --- CDF 绘图 ---
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5)) 

color_history_cdf = '#2E86AB'  
color_simulated_cdf = '#F18F01' 

# --- 模拟价格 CDF (Hybrid) ---
data_sim = np.sort(simulated_prices)
cdf_sim = np.arange(1, len(data_sim) + 1) / len(data_sim)
ax2.plot(data_sim, cdf_sim, 
        color=color_simulated_cdf, 
        linewidth=1.2, 
        label='Simulated (Hybrid) Price ECDF', 
        antialiased=True)

# --- 历史价格 CDF (增加延伸逻辑) ---
data_hist = np.sort(historical_prices)
cdf_hist = np.arange(1, len(data_hist) + 1) / len(data_hist)

# 获取整个图表的 X 轴范围 (由 Simulated 数据主导)
x_min_global = data_sim.min()
x_max_global = data_sim.max()

# 检查是否需要延伸
if data_hist.min() > x_min_global or data_hist.max() < x_max_global:
    # 延伸 X 轴数据
    extended_data_hist = np.concatenate([
        [x_min_global],        # 起点 X 值设为全局最小值
        data_hist,
        [x_max_global]         # 终点 X 值设为全局最大值
    ])
    # 延伸 Y 轴数据
    extended_cdf_hist = np.concatenate([
        [0.0],                 # 赋予起点 Y 值 0
        cdf_hist,
        [1.0]                  # 赋予终点 Y 值 1
    ])
else:
    # 如果历史范围足够宽，则无需延伸
    extended_data_hist = data_hist
    extended_cdf_hist = cdf_hist

# 绘制延伸后的历史 CDF
ax2.plot(extended_data_hist, extended_cdf_hist, 
        color=color_history_cdf, 
        linewidth=1.2, 
        label='Historical Price ECDF', 
        antialiased=True)


# --- 图表样式设置 ---
ax2.set_title('Empirical CDF of ETH Mid-price Comparison')
ax2.set_xlabel('Mid-price (USDT)') 
ax2.set_ylabel('Cumulative Probability')

ax2.tick_params(direction='in', which='both')
ax2.grid(True, alpha=0.5, linestyle=':', linewidth=0.8)

ax2.set_ylim(0, 1.05) 

# 设置 X 轴范围，确保覆盖全局范围
x_min_plot = min(historical_prices.min(), simulated_prices.min())
x_max_plot = max(historical_prices.max(), simulated_prices.max())
ax2.set_xlim(x_min_plot * 0.99, x_max_plot * 1.01)

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
filename_cdf = 'ETH价格ECDF对比图_相似度'
plt.savefig(f'{filename_cdf}_高清版.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{filename_cdf}_矢量版.svg', format='svg', bbox_inches='tight')

plt.show()