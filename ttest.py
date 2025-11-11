from scipy.stats import ttest_rel
import numpy as np
import pandas as pd

def second_level_did(hybrid_path, original_path,
                     metrics=('spread', 'depth', 'volume'),
                     tol='500ms'):
    """
    逐秒差分 + 配对 t 检验（同一秒配对）
    参数
    ----
    hybrid_path   : str  -- Hybrid 日志 pickle 路径
    original_path : str  -- Original 日志 pickle 路径
    metrics       : tuple-- 需要检验的指标名（与 Event 字典 key 对应）
    tol           : str  -- merge_asof 容差，默认 ±500 ms
    返回
    ----
    delta_df    : DataFrame-- 逐秒差值（列=Δmetric）
    mean_deltas : Series   -- 三指标差值的均值
    t_stats     : Series   -- 三指标配对 t 值
    p_vals      : Series   -- 三指标配对 p 值
    """

    # 1. 读数据 → 统一秒级索引
    def _load_second_df(path):
        df = pd.read_pickle(path).reset_index()
        df['EventTime'] = pd.to_datetime(df['EventTime']).dt.round('1s')
        return df.set_index('EventTime').sort_index()

    df_h = _load_second_df(hybrid_path)
    df_o = _load_second_df(original_path)

    # 2. 按秒聚合（过滤非字典类型的Event）
    agg_h = df_h.groupby('EventTime').agg(
        spread=('Event', lambda x: np.nanmean([
            e['spread'] for e in x 
            if isinstance(e, dict) and 'spread' in e
        ])),
        depth=('Event', lambda x: np.nanmean([
            e['bid_liquidity'] + e['ask_liquidity'] 
            for e in x 
            if isinstance(e, dict) and 'bid_liquidity' in e and 'ask_liquidity' in e
        ])),
        volume=('Event', lambda x: np.nansum([
            e['volume'] for e in x 
            if isinstance(e, dict) and 'volume' in e
        ]))
    )
    agg_o = df_o.groupby('EventTime').agg(
        spread=('Event', lambda x: np.nanmean([
            e['spread'] for e in x 
            if isinstance(e, dict) and 'spread' in e
        ])),
        depth=('Event', lambda x: np.nanmean([
            e['bid_liquidity'] + e['ask_liquidity'] 
            for e in x 
            if isinstance(e, dict) and 'bid_liquidity' in e and 'ask_liquidity' in e
        ])),
        volume=('Event', lambda x: np.nansum([
            e['volume'] for e in x 
            if isinstance(e, dict) and 'volume' in e
        ]))
    )

    # 3. 秒级对齐
    merged = pd.merge_asof(
        agg_h.add_prefix('h_'),
        agg_o.add_prefix('o_'),
        left_index=True, right_index=True,
        direction='nearest', tolerance=pd.Timedelta(tol)
    ).dropna()

    # 4. 逐秒差分
    delta_df = pd.DataFrame({
        f'Δ{m}': merged[f'h_{m}'] - merged[f'o_{m}']
        for m in metrics
    })

    # 5. 计算差值均值
    mean_deltas = delta_df.mean()

    # 6. 配对 t 检验
    t_stats, p_vals = {}, {}
    for m in metrics:
        t_stats[m], p_vals[m] = ttest_rel(merged[f'h_{m}'], merged[f'o_{m}'])

    return delta_df, mean_deltas, pd.Series(t_stats), pd.Series(p_vals)

if __name__ == "__main__":
    # 调用函数并获取均值信息
    delta_df, mean_deltas, t_stats, p_vals = second_level_did(
        hybrid_path=r'log\rmsc04_two_hour\SNAPSHOT_AGENT.bz2',
        original_path=r'log\rmsc03_two_hour\SNAPSHOT_AGENT.bz2'
    )
    
    # 打印结果（包含均值）
    print('=== 逐秒配对 t 检验 ===')
    for m in ['spread', 'depth', 'volume']:
        print(
            f'Δ{m.capitalize():<7}: '
            f'均值={mean_deltas[f"Δ{m}"]:.4f}, '
            f't={t_stats[m]:6.2f}, '
            f'p={p_vals[m]:.3g}'
        )
    
    delta_df.to_csv('second_did_delta.csv')