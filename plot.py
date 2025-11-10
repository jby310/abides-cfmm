import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_and_preprocess_data(file_path):
    """加载并预处理快照数据"""
    # 读取数据并重置索引
    df = pd.read_pickle(file_path)
    df = df.reset_index()
    
    # 转换时间格式并设置为索引
    df['EventTime'] = pd.to_datetime(df['EventTime'])
    df = df.set_index('EventTime').sort_index()
    
    # 筛选1小时数据（从第二条记录开始）
    if len(df) >= 2:  # 确保有足够数据
        start_time = df.index[1]
        end_time = start_time + pd.Timedelta(hours=1)
        return df[(df.index >= start_time) & (df.index <= end_time)]
    return pd.DataFrame()  # 空数据处理


def process_liquidity_events(df):
    """处理流动性事件数据并按分钟聚合，计算买卖方最小值"""
    if df.empty:
        return pd.DataFrame(columns=['bid_liquidity', 'ask_liquidity', 'min_liquidity'])
        
    liquidity_events = df[df['EventType'] == 'LIQUIDITY_1PCT']
    if liquidity_events.empty:
        return pd.DataFrame(columns=['bid_liquidity', 'ask_liquidity', 'min_liquidity'])
    
    # 提取流动性数据并计算最小值
    liquidity_data = pd.DataFrame({
        'bid_liquidity': [event['bid_liquidity'] for event in liquidity_events['Event']],
        'ask_liquidity': [event['ask_liquidity'] for event in liquidity_events['Event']]
    }, index=liquidity_events.index)
    liquidity_data['min_liquidity'] = liquidity_data[['bid_liquidity', 'ask_liquidity']].min(axis=1)
    
    # 按分钟聚合
    return liquidity_data.resample('1min').mean().dropna()


def process_spread_events(df):
    """处理价差事件数据并按分钟聚合"""
    if df.empty:
        return pd.DataFrame(columns=['spread'])
        
    spread_events = df[df['EventType'] == 'SPREAD']
    if spread_events.empty:
        return pd.DataFrame(columns=['spread'])
    
    spread_data = pd.DataFrame({
        'spread': [event['spread'] for event in spread_events['Event']]
    }, index=spread_events.index)
    
    return spread_data.resample('1min').mean().dropna()


def process_volume_events(df):
    """处理交易量事件数据并按分钟聚合"""
    if df.empty:
        return pd.DataFrame(columns=['volume', 'cumulative_volume'])
        
    volume_events = df[df['EventType'] == 'VOLUME']
    if volume_events.empty:
        return pd.DataFrame(columns=['volume', 'cumulative_volume'])
    
    volume_data = pd.DataFrame({
        'volume': [event['volume'] for event in volume_events['Event']]
    }, index=volume_events.index)
    
    volume_data_min = volume_data.resample('1min').sum().dropna()
    volume_data_min['cumulative_volume'] = volume_data_min['volume'].cumsum()
    return volume_data_min


def plot_comparison_metrics(data_dict):
    """绘制两个数据源的对比图表"""
    # 创建画布和子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    fig.suptitle('Market Metrics Comparison (rmsc03 vs rmsc04)', fontsize=16)
    
    # 定义两个数据源的样式
    styles = {
        'rmsc03': {'color': 'blue', 'marker': 'o', 'alpha': 0.7},
        'rmsc04': {'color': 'orange', 'marker': 's', 'alpha': 0.7}
    }
    
    # 1. 流动性对比图表
    for name, data in data_dict.items():
        liq = data['liquidity']
        if not liq.empty:
            # axes[0].plot(liq.index, liq['bid_liquidity'], 
            #              label=f'{name} Bid Liquidity', **styles[name])
            # axes[0].plot(liq.index, liq['ask_liquidity'], 
            #              label=f'{name} Ask Liquidity', linestyle='--',** styles[name])
            axes[0].plot(liq.index, liq['min_liquidity'], 
                         label=f'{name} Min Liquidity', linestyle=':', **styles[name])
    
    axes[0].set_ylabel('Liquidity (Shares)', fontsize=12)
    axes[0].legend(fontsize=10, loc='upper left')
    axes[0].grid(alpha=0.3)
    axes[0].set_title('Liquidity Comparison', fontsize=14)
    
    # 2. 价差对比图表
    bar_width = 0.3/24/60  # 缩小柱状图宽度避免重叠
    for i, (name, data) in enumerate(data_dict.items()):
        spread = data['spread']
        if not spread.empty:
            # 偏移位置避免柱状图重叠
            axes[1].bar(spread.index + pd.Timedelta(minutes=i*2), 
                        spread['spread'], width=bar_width,
                        label=f'{name} Average Spread', color=styles[name]['color'], 
                        alpha=styles[name]['alpha'])
    
    axes[1].set_ylabel('Spread ($)', fontsize=12)
    axes[1].legend(fontsize=10, loc='upper left')
    axes[1].grid(alpha=0.3)
    axes[1].set_title('Spread Comparison', fontsize=14)
    
    # 3. 交易量对比图表（双轴）
    for name, data in data_dict.items():
        volume = data['volume']
        if not volume.empty:
            # 左侧轴：累积交易量
            axes[2].plot(volume.index, volume['cumulative_volume'],
                         label=f'{name} Cumulative Volume',** styles[name])
            
            # 右侧轴：每分钟交易量（共享一个副轴）
            if not hasattr(axes[2], 'twin_ax'):
                axes[2].twin_ax = axes[2].twinx()
            axes[2].twin_ax.bar(volume.index + pd.Timedelta(minutes=1 if name == 'rmsc04' else 0),
                                volume['volume'], width=bar_width,
                                label=f'{name} Volume per Minute', 
                                color=styles[name]['color'], alpha=0.4)
    
    axes[2].set_ylabel('Cumulative Volume (Shares)', fontsize=12)
    axes[2].twin_ax.set_ylabel('Volume per Minute (Shares)', fontsize=12)
    axes[2].legend(fontsize=10, loc='upper left')
    axes[2].twin_ax.legend(fontsize=10, loc='upper right')
    axes[2].grid(alpha=0.3)
    axes[2].set_title('Volume Comparison', fontsize=14)
    
    # 设置x轴格式
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    
    # 调整布局并保存图表
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('market_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 定义两个数据源路径
    data_paths = {
        'rmsc03': r'log\rmsc03_two_hour\SNAPSHOT_AGENT.bz2',
        'rmsc04': r'log\rmsc04_two_hour\SNAPSHOT_AGENT.bz2'
    }
    
    # 批量处理数据
    data_dict = {}
    for name, path in data_paths.items():
        raw_data = load_and_preprocess_data(path)
        data_dict[name] = {
            'liquidity': process_liquidity_events(raw_data),
            'spread': process_spread_events(raw_data),
            'volume': process_volume_events(raw_data)
        }
    
    # 绘制对比图表
    plot_comparison_metrics(data_dict)