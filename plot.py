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


def plot_mid_price_comparison(merged, ax, styles):
    """绘制中间价预测与原始值对比图"""
    # 预测值曲线
    ax.plot(merged.index, merged['mid_price_1_pred'], label='Hybrid', color=styles[name]['color'], linewidth=1.5)
    # 原始值曲线
    ax.plot(merged.index, merged['mid_price_1_origin'], label='Original', color='blue', linewidth=1.5, alpha=styles[name]['alpha'])

    # 图表美化
    ax.set_xlabel('Timestamp', fontsize=10)
    ax.set_ylabel('Mid Price', fontsize=10)
    ax.set_title('Predicted vs Original Mid Price', fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(linestyle='--', alpha=0.3)


def plot_comparison_metrics(data_dict, merged):
    """绘制所有对比图表，使用2x2子图布局"""
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=False)
    fig.suptitle('Market Metrics & Price Comparison', fontsize=16)
    
    # 定义两个数据源的样式
    styles = {
        'rmsc03': {'color': 'blue', 'marker': 'o', 'alpha': 0.7},
        'rmsc04': {'color': 'orange', 'marker': 's', 'alpha': 0.7}
    }
    bar_width = 0.3/24/60  # 柱状图宽度
    
    # 1. 流动性对比图表 (0,0位置)
    ax1 = axes[0, 0]
    for name, data in data_dict.items():
        liq = data['liquidity']
        if not liq.empty:
            # # 显示bid流动性（实线）
            # ax1.plot(liq.index, liq['bid_liquidity'], 
            #              label=f'{name} Bid Liquidity', **styles[name])
            # # 显示ask流动性（虚线）
            # ax1.plot(liq.index, liq['ask_liquidity'], 
            #              label=f'{name} Ask Liquidity', linestyle='--',** styles[name])
            # 显示最小流动性（点线）
            ax1.plot(liq.index, liq['min_liquidity'], 
                         label=f'{name} Min Liquidity', linestyle=':', **styles[name])


        # liq = data['liquidity']
        # if not liq.empty:
        #     ax1.plot(liq.index, liq['min_liquidity'], 
        #              label=f'{name} Min Liquidity', linestyle=':', **styles[name])
    
    ax1.set_ylabel('Liquidity (Shares)', fontsize=10)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_title('Liquidity Comparison', fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 价差对比图表 (0,1位置)
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(data_dict.items()):
        spread = data['spread']
        if not spread.empty:
            ax2.bar(spread.index + pd.Timedelta(minutes=i*2), 
                    spread['spread'], width=bar_width,
                    label=f'{name} Average Spread', color=styles[name]['color'], 
                    alpha=styles[name]['alpha'])
    
    ax2.set_ylabel('Spread ($)', fontsize=10)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.set_title('Spread Comparison', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 交易量对比图表 (1,0位置)
    ax3 = axes[1, 0]
    for name, data in data_dict.items():
        volume = data['volume']
        if not volume.empty:
            # 左侧轴：累积交易量
            ax3.plot(volume.index, volume['cumulative_volume'],
                     label=f'{name} Cumulative Volume',** styles[name])
            
            # 右侧轴：每分钟交易量
            if not hasattr(ax3, 'twin_ax'):
                ax3.twin_ax = ax3.twinx()
            ax3.twin_ax.bar(volume.index + pd.Timedelta(minutes=1 if name == 'rmsc04' else 0),
                            volume['volume'], width=bar_width,
                            label=f'{name} Volume per Minute', 
                            color=styles[name]['color'], alpha=0.4)
    
    ax3.set_ylabel('Cumulative Volume (Shares)', fontsize=10)
    ax3.twin_ax.set_ylabel('Volume per Minute (Shares)', fontsize=10)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.twin_ax.legend(fontsize=8, loc='upper right')
    ax3.grid(alpha=0.3)
    ax3.set_title('Volume Comparison', fontsize=12)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 中间价对比图表 (1,1位置)
    plot_mid_price_comparison(merged, axes[1, 1], styles)
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 调整布局并保存图表
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('market_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 定义数据源路径
    data_paths = {
        'rmsc03': r'log\rmsc03_two_hour\SNAPSHOT_AGENT.bz2',
        'rmsc04': r'log\rmsc04_two_hour\SNAPSHOT_AGENT.bz2'
    }
    
    # 批量处理市场指标数据
    data_dict = {}
    for name, path in data_paths.items():
        raw_data = load_and_preprocess_data(path)
        data_dict[name] = {
            'liquidity': process_liquidity_events(raw_data),
            'spread': process_spread_events(raw_data),
            'volume': process_volume_events(raw_data)
        }
    
    # 处理中间价数据
    origin = pd.read_csv(r'log\rmsc03_two_hour\mid_price.csv')
    pred = pd.read_csv(r'log\rmsc04_two_hour\mid_price.csv')
    
    # 转换时间格式并设置为索引
    pred['timestamp'] = pd.to_datetime(pred['timestamp'])
    pred.set_index('timestamp', inplace=True)
    
    origin['timestamp'] = pd.to_datetime(origin['timestamp'])
    origin.set_index('timestamp', inplace=True)
    
    # 外连接合并并排序
    merged = pred.join(origin, lsuffix='_pred', rsuffix='_origin', how='outer')
    merged.sort_index(inplace=True)
    
    # 绘制所有对比图表
    plot_comparison_metrics(data_dict, merged)