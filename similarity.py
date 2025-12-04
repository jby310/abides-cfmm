import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 关键优化：全局参数设置（解决模糊核心） ----------------------
plt.rcParams.update({
    # 1. 提高默认显示DPI（Windows默认100，Mac默认72，提升到150-200）
    'figure.dpi': 150,
    'savefig.dpi': 300,  # 保存图片的DPI（保持高分辨率）
    
    # 2. 开启抗锯齿（字体/线条边缘平滑）
    'text.antialiased': True,
    'axes.linewidth': 1,
    'lines.antialiased': True,
    
    # 3. 使用矢量字体（避免点阵字体模糊）
    'font.family': ['Times New Roman', 'SimHei'],  # 优先矢量字体
    'font.sans-serif': ['Times New Roman', 'SimHei'],
    
    # 4. 解决负号显示问题
    'axes.unicode_minus': False,
    
    # 5. 微调字体大小（避免过小导致模糊）
    'font.size': 8,  # 全局基础字体大小
    'axes.labelsize': 9,  # 坐标轴标签字体大小
    'axes.titlesize': 11,  # 子图标题字体大小
    'legend.fontsize': 8,  # 图例字体大小
    'xtick.labelsize': 7,  # x轴刻度字体大小
    'ytick.labelsize': 7,  # y轴刻度字体大小
})

# ---------------------- 原有数据处理代码（修改：只保留Hybrid场景） ----------------------
fundamental_df = pd.read_excel('data\\ETH1.xlsx', sheet_name='1110')
fundamental_df = fundamental_df[fundamental_df['Dates'] != 'Dates']
fundamental_df['Dates'] = fundamental_df['Dates'].str.replace("'", "", regex=False)
fundamental_df['Dates'] = fundamental_df['Dates'].str.replace(".000000", "", regex=False)
fundamental_df['Dates'] = pd.to_datetime(fundamental_df['Dates'])
price_series = pd.Series(fundamental_df['Price'].values, index=fundamental_df['Dates'])

from plot1 import load_and_preprocess_data, process_mid_price_from_depth

# 修改：只保留Hybrid场景的路径和标签
paths = [r'log\rmsc04_two_hour\SNAPSHOT_AGENT.bz2']  # 仅Hybrid路径
labels = ['Hybrid']  # 仅Hybrid标签
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
    mid_price_minutely = mid_price_data_df[mid_price_col].resample('10s').agg(aggregation_method).dropna()
    mid_price_minutely_list.append(mid_price_minutely)
    
    print(f"\n路径 {path} 按30秒{aggregation_method}聚合后的数据形状：", mid_price_minutely.shape)
    print("聚合后的数据前5行：")
    print(mid_price_minutely.head())

# ---------------------- 绘图代码（修改：单图+四周边框） ----------------------
# 修改：改为单图（1行1列），调整画布比例更协调
fig, ax = plt.subplots(1, 1, figsize=(5, 4))  # 宽10，高6（单图更美观）

color_history = '#2E86AB'
color_simulated = '#F18F01'  # 仅Hybrid对应的颜色


# 绘制Hybrid模拟曲线（列表仅1个元素，直接取第0个）
mid_price_data = mid_price_minutely_list[0]
ax.plot(mid_price_data.index, mid_price_data.values,
        color=color_simulated, linewidth=0.5, alpha=1,
        label=f'Simulated (Hybrid)', antialiased=True)

# 绘制曲线（保持原有样式）
ax.plot(price_series.index, price_series.values, 
        color=color_history, linewidth=0.5, alpha=1,
        label='History', antialiased=True)

# 子图样式设置
# ax.set_xlabel('Time', fontweight='bold')
ax.set_xlabel('Time')
# ax.set_ylabel('Mid-price（USD）', fontweight='bold')
ax.set_ylabel('Mid-price（USDT）')
# ax.set_title('ETH Price Comparison', fontweight='bold')
ax.set_title('ETH Price Comparison')

# 修改：显示四周边框（恢复top和right spine）
ax.spines['top'].set_visible(True)  # 显示上边框
ax.spines['right'].set_visible(True)  # 显示右边框
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)
ax.spines['top'].set_linewidth(0.6)  # 统一边框宽度
ax.spines['right'].set_linewidth(0.6)

# 网格优化
ax.grid(True, alpha=0.4, linestyle=':', linewidth=1.2)

# 图例设置 (右上角)

ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
          framealpha=1.0)

# x轴日期显示优化
fig.autofmt_xdate()

# ---------------------- 缩小y轴范围（保持原有设置） ----------------------
all_y_data = np.concatenate([
    price_series.values,
    mid_price_minutely_list[0].values  # 仅Hybrid数据
])
y_min = all_y_data.min() * 1.0
y_max = all_y_data.max() * 0.998
ax.set_ylim(y_min, y_max)

# 调整布局
plt.tight_layout()

# ---------------------- 保存设置 ----------------------
plt.savefig(
    'ETH价格_Hybrid模拟中间价对比图_高清版.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)

# 保存矢量图
plt.savefig('ETH价格_Hybrid对比图_矢量版.svg', format='svg', bbox_inches='tight')

# 可选：显示图片
plt.show()