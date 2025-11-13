import argparse
import json
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

def second_level_did(hybrid_path, original_path,
                     metrics=('spread', 'depth', 'volume'),
                     tol='500ms'):
    """
    逐秒差分 + 配对 t 检验（同一秒配对）
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

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行t检验分析')
    parser.add_argument('--output_dir', type=str, help='输出目录', default='ttest_results')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建文件路径
    hybrid_path = f'log/rmsc04_two_hour/SNAPSHOT_AGENT.bz2'
    original_path = 'log/rmsc03_two_hour/SNAPSHOT_AGENT.bz2'
    
    try:
        # 调用函数并获取结果
        delta_df, mean_deltas, t_stats, p_vals = second_level_did(
            hybrid_path=hybrid_path,
            original_path=original_path
        )
        
        # 准备结果数据
        result = {
            'spread_mean': mean_deltas['Δspread'],
            'spread_t': t_stats['spread'],
            'spread_p': p_vals['spread'],
            'depth_mean': mean_deltas['Δdepth'],
            'depth_t': t_stats['depth'],
            'depth_p': p_vals['depth'],
            'volume_mean': mean_deltas['Δvolume'],
            'volume_t': t_stats['volume'],
            'volume_p': p_vals['volume']
        }
        
        # 保存结果到JSON文件
        with open(os.path.join(args.output_dir, 'ttest_results.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        # 保存详细的差值数据
        delta_df.to_csv(os.path.join(args.output_dir, 'second_did_delta.csv'))
        
        # 打印结果（可选）
        print('=== 逐秒配对 t 检验 ===')
        for m in ['spread', 'depth', 'volume']:
            print(
                f'Δ{m.capitalize():<7}: '
                f'均值={result[f"{m}_mean"]:.4f}, '
                f't={result[f"{m}_t"]:6.2f}, '
                f'p={result[f"{m}_p"]:.3g}'
            )
        
        print(f"结果已保存到 {args.output_dir}")
        
    except Exception as e:
        print(f"t检验分析失败: {e}")
        # 保存错误信息
        error_result = {
            'error': str(e),
            'spread_mean': 0,
            'spread_t': 0,
            'spread_p': 1,
            'depth_mean': 0,
            'depth_t': 0,
            'depth_p': 1,
            'volume_mean': 0,
            'volume_t': 0,
            'volume_p': 1
        }
        with open(os.path.join(args.output_dir, 'ttest_results.json'), 'w') as f:
            json.dump(error_result, f, indent=2)

if __name__ == "__main__":
    main()