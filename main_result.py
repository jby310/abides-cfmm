import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_dummy_data():
    """创建一个模拟的实验结果DataFrame用于演示"""
    np.random.seed(42)
    k_values = np.logspace(9, 12, 5)  # 1e9, 1e9.75, ..., 1e12
    fee_values = np.array([0.0001, 0.001, 0.01, 0.1])
    slippage_values = np.array([0.01, 0.05])
    
    data = []
    for k in k_values:
        for fee in fee_values:
            for slippage in slippage_values:
                # 模拟指标变化
                spread = (fee * 0.01) + (1 / np.log10(k)) * 0.001 + np.random.normal(0, 0.0005)
                volume = np.log10(k) * 1000 + (1 / fee) * 10 + np.random.normal(0, 100)
                depth = np.log10(k) * 500 + np.random.normal(0, 50)
                
                # 假设 spread_mean < 0 更好，volume/depth_mean > 0 更好
                data.append({
                    'k': k,
                    'fee': fee,
                    'max_slippage': slippage,
                    'spread_mean': -spread * 10,  # 负值表示价差缩小
                    'volume_mean': volume / 10000,
                    'depth_mean': depth / 100
                })
    return pd.DataFrame(data)

# 假设你的数据加载或生成如下：
df_results = pd.read_excel('实验数据2.xlsx', sheet_name='seed5') # 实际使用时

# --- 筛选数据 ---
FIXED_SLIPPAGE_VALUE = 0.01
df_filtered = df_results[
    np.isclose(df_results['max_slippage'], FIXED_SLIPPAGE_VALUE, atol=1e-6)
].copy()

# 检查筛选结果
if df_filtered.empty:
    print(f"❌ 错误：在数据中未找到 max_slippage = {FIXED_SLIPPAGE_VALUE} 的记录。")
    exit()

print(f"✅ 已筛选出 {len(df_filtered)} 条 max_slippage = {FIXED_SLIPPAGE_VALUE} 的数据。")

def plot_multiple_heatmaps(df_filtered):
    """
    Plots heatmaps for the three metrics (spread_mean, volume_mean, depth_mean) 
    against k and fee. Uses English labels.
    """
    
    indicators = {
        'spread_mean': 'Spread Change ($\Delta$ Spread)',
        'volume_mean': 'Volume Change ($\Delta$ Volume)',
        'depth_mean': 'Depth Change ($\Delta$ Depth)'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.suptitle(f'Heatmaps of Metric Changes (Fixed Max Slippage={FIXED_SLIPPAGE_VALUE})', fontsize=18, y=1.05)
    
    for i, (col, title) in enumerate(indicators.items()):
        
        # Pivot the data into a matrix of k (rows) and fee (columns)
        heatmap_data = df_filtered.pivot_table(
            index='k', 
            columns='fee', 
            values=col, 
            aggfunc='mean' # Use mean to handle multiple seeds/runs per parameter pair
        )
        
        # Format k-axis labels using scientific notation (log space)
        k_labels = [f'{x:.2e}' for x in heatmap_data.index]
        fee_labels = [f'{x:.4f}' for x in heatmap_data.columns]
        
        # Plot the heatmap
        # Use diverging cmap ('coolwarm') for Spread (since negative is better, positive is worse)
        # Use sequential cmap ('viridis', 'magma') for Volume/Depth (since positive is better)
        cmap_choice = 'coolwarm' if col == 'spread_mean' else ('magma' if col == 'volume_mean' else 'viridis')
        
        sns.heatmap(
            heatmap_data, 
            ax=axes[i], 
            annot=True, 
            fmt=".3f", 
            cmap=cmap_choice,
            linewidths=.5, 
            linecolor='black',
            cbar_kws={'label': f'Average {title}'}
        )
        
        # Set axes labels and title
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel('Fee', fontsize=12)
        axes[i].set_ylabel(r'Pool Size ($\mathbf{k}$)', fontsize=12)
        
        # Ensure labels are displayed correctly
        axes[i].set_yticklabels(k_labels, rotation=0)
        axes[i].set_xticklabels(fee_labels, rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    
    # Save and display image
    filename = f'heatmaps_slippage_{FIXED_SLIPPAGE_VALUE:.2f}_EN.png'
    plt.savefig(filename, dpi=300)
    # plt.show()
    print(f"\nHeatmaps saved to {filename}")
    # Triggering the image tag here to show the resulting visualization.
    # 


# --- Execute Plotting ---
plot_multiple_heatmaps(df_filtered)