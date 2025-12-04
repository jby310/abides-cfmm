import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tempfile
import json
import random

# 设置中文显示和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量用于控制是否跳过初始通用对照组
# 第一次运行时设置为 False，运行后可以设置为 True 跳过初始通用对照组（仅当agents_values长度为1时）
IS_CONTROL_GROUP_RUN = False 
# 存储已运行的对照组的参数，以便动态跳过
_CONTROL_GROUP_RUN_PARAMS = {} 


class ExperimentRunner:
    def __init__(self, base_cmd, param_ranges):
        self.base_cmd = base_cmd
        self.results = []
        self.control_group_run = IS_CONTROL_GROUP_RUN 
        self.results_file = 'experiment_results_grid_search_v2_crypto.csv' # 更改结果文件名以区分
        self.param_ranges = param_ranges
        self.existing_results = self.load_results_from_csv()
        self.control_group_params_run = set() # 记录已运行的对照组参数: (num_hybrid_agents, fundamental_file_path, seed)
        
    def replace_parameter(self, cmd, param_name, param_value):
        """通用参数替换函数"""
        # param_value 必须是字符串或能转为字符串
        param_value_str = str(param_value)
        
        # 处理 -param_name value 形式
        pattern1 = rf'({param_name})\s+\S+'
        replacement1 = r'\1 ' + param_value_str
        cmd = re.sub(pattern1, replacement1, cmd)
        
        # 处理 param_name=value 形式
        pattern2 = rf'({param_name}=)\S+'
        replacement2 = r'\1' + param_value_str
        cmd = re.sub(pattern2, replacement2, cmd)
        
        return cmd
        
    def _get_asset_config(self, fundamental_file_path):
        """根据 fundamental_file_path 返回对应的 r_bar 和 -d 值"""
        path = fundamental_file_path.strip()
        if 'ETH1.xlsx' in path:
            return {'r_bar': 3611.0, '-d': 'ETH'}
        elif 'BIT.xlsx' in path:
            return {'r_bar': 113994.6305, '-d': 'BIT'}
        else:
            # 默认值
            return {'r_bar': 3611.0, '-d': '20251028'}

    def run_control_group(self, num_agents_value, fundamental_file_value, seed_value):
        """
        运行对照组，现在接受参数以匹配实验组。
        对照组只需替换 num_hybrid_agents, fundamental_file_path, seed, r_bar, -d。
        """
        # 确定对照组的唯一标识 (num_hybrid_agents, fundamental_file_path, seed)
        control_key = (int(num_agents_value), fundamental_file_value.strip(), int(seed_value))
        
        # 检查是否已运行
        if control_key in self.control_group_params_run:
            print(f"对照组 (agents={num_agents_value}, fund={fundamental_file_value.strip()}, seed={seed_value}) 已运行，跳过...")
            return True
        
        # 获取资产配置
        asset_config = self._get_asset_config(fundamental_file_value)
        r_bar_value = asset_config['r_bar']
        d_value = asset_config['-d']
        
        try:
            print(f"运行对照组: agents={num_agents_value}, fund={fundamental_file_value.strip()}, seed={seed_value}, r_bar={r_bar_value}, -d={d_value}")
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            control_cmd = lines[0] # 第一个命令是对照组模板
            
            # 替换对照组所需的参数
            control_cmd = self.replace_parameter(control_cmd, '--num-hybrid-agents', int(num_agents_value))
            control_cmd = self.replace_parameter(control_cmd, '--fundamental-file-path', fundamental_file_value)
            control_cmd = self.replace_parameter(control_cmd, '-s', seed_value)
            control_cmd = self.replace_parameter(control_cmd, '--r-bar', r_bar_value)
            control_cmd = self.replace_parameter(control_cmd, '-d', d_value)

            
            temp_dir = tempfile.gettempdir()
            batch_filename = os.path.join(temp_dir, 'run_control.bat')
            
            batch_content = f"""@echo off
{control_cmd}
"""
            
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            result_code = os.system(f'"{batch_filename}"')
            
            if result_code != 0:
                print(f"对照组执行失败，返回码: {result_code}")
                return False
                
            self.control_group_params_run.add(control_key)
            print("对照组运行完成")
            return True
                
        except Exception as e:
            print(f"对照组运行异常: {e}")
            return False
            
    def append_result_to_csv(self, result):
        """将单次实验结果追加到CSV文件"""
        try:
            # 确保 k, fee, max_slippage 有足够的精度进行比较
            result_for_comparison = result.copy()
            result_for_comparison['k'] = round(result_for_comparison['k'], 6)
            result_for_comparison['fee'] = round(result_for_comparison['fee'], 6)
            result_for_comparison['max_slippage'] = round(result_for_comparison['max_slippage'], 6)

            result_df = pd.DataFrame([result])
            
            if os.path.exists(self.results_file):
                existing_df = self.load_results_from_csv()
                
                # 对现有结果进行四舍五入以匹配浮点数比较逻辑
                existing_df_rounded = existing_df.copy()
                existing_df_rounded['k'] = existing_df_rounded['k'].round(6)
                existing_df_rounded['fee'] = existing_df_rounded['fee'].round(6)
                existing_df_rounded['max_slippage'] = existing_df_rounded['max_slippage'].round(6)
                
                # 检查当前结果是否已存在（精确匹配所有参数）
                is_duplicate = existing_df_rounded[
                    (existing_df_rounded['k'] == result_for_comparison['k']) & 
                    (existing_df_rounded['fee'] == result_for_comparison['fee']) & 
                    (existing_df_rounded['max_slippage'] == result_for_comparison['max_slippage']) & 
                    (existing_df_rounded['num_hybrid_agents'] == result_for_comparison['num_hybrid_agents']) & 
                    (existing_df_rounded['fundamental_file_path'] == result_for_comparison['fundamental_file_path']) & 
                    (existing_df_rounded['seed'] == result_for_comparison['seed'])
                ].shape[0] > 0
                
                if is_duplicate:
                    # print("此组合结果已存在，跳过写入CSV。")
                    return True

                result_df.to_csv(self.results_file, mode='a', header=False, index=False)
            else:
                result_df.to_csv(self.results_file, index=False)
            
            # print(f"结果已追加到 {self.results_file}")
            return True
        except Exception as e:
            print(f"写入CSV文件失败: {e}")
            return False

    def determine_effect_type(self, spread_mean, depth_mean, volume_mean, spread_p, depth_p, volume_p):
        """内部函数：根据 T 检验结果判断效应类型 (显著性水平 alpha = 0.05)"""
        alpha = 0.05
        
        # 共生效应：价差不显著变化 或 略微变小，同时深度和成交量显著增加
        # 略微变小：我们允许价差均值在统计学上不显著（p>alpha）或者变动非常小（绝对值小于0.002）
        is_spread_stable = (spread_p > alpha) or (abs(spread_mean) <= 0.002)
        is_depth_increase = (depth_p < alpha) and (depth_mean > 0)
        is_volume_increase = (volume_p < alpha) and (volume_mean > 0)
        
        if is_spread_stable and is_depth_increase and is_volume_increase:
            return "共生效应"
        
        # 挤出效应：价差显著变大（即显著变差），同时深度和成交量均没有显著增加
        # 显著变大（显著变差）：p < alpha 且 mean > 0.005
        is_spread_worse = (spread_p < alpha) and (spread_mean > 0.005)
        is_depth_not_better = (depth_p > alpha) or (depth_mean <= 0)
        is_volume_not_better = (volume_p > alpha) or (volume_mean <= 0)
        
        if is_spread_worse and is_depth_not_better and is_volume_not_better:
            return "挤出效应"
        
        # 混合效应 (恶化价差/增加流动性)：价差显著变大，但深度或成交量也显著增加
        if is_spread_worse and (is_depth_increase or is_volume_increase):
             return "混合效应 (恶化价差/增加流动性)"
        
        # 其他所有情况都归类为混合效应（可能包含不显著变化，或仅一两个指标显著变化等）
        return "混合效应"


    def run_single_experiment(self, k_value, fee_value, slippage_value, num_agents_value, fundamental_file_value, seed_value):
        """运行单次实验（实验组）并提取 T 检验结果和效应类型"""
        # --- 1. 运行对照组 (必须在实验组之前运行) ---
        if not self.run_control_group(num_agents_value, fundamental_file_value, seed_value):
            return None # 如果对照组运行失败，则跳过本次实验

        try:
            print(f"运行实验组: k={k_value:.2e}, fee={fee_value:.4f}, slippage={slippage_value:.3f}, agents={num_agents_value}, fund={fundamental_file_value.strip()}, seed={seed_value}")
            
            result_dir = "ttest_results"
            os.makedirs(result_dir, exist_ok=True)
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            cmd2 = lines[1] # 第二个命令是实验组模板
            
            # --- 2. 替换实验组参数 ---
            
            # 获取资产配置并替换 r_bar 和 -d
            asset_config = self._get_asset_config(fundamental_file_value)
            r_bar_value = asset_config['r_bar']
            d_value = asset_config['-d']
            
            cmd2 = self.replace_parameter(cmd2, '-k', int(k_value))
            cmd2 = self.replace_parameter(cmd2, '--fee', fee_value)
            cmd2 = self.replace_parameter(cmd2, '--max-slippage', slippage_value)
            cmd2 = self.replace_parameter(cmd2, '--num-hybrid-agents', int(num_agents_value))
            cmd2 = self.replace_parameter(cmd2, '--fundamental-file-path', fundamental_file_value)
            cmd2 = self.replace_parameter(cmd2, '-s', seed_value)
            # 替换 r_bar 和 -d
            cmd2 = self.replace_parameter(cmd2, '--r-bar', r_bar_value)
            cmd2 = self.replace_parameter(cmd2, '-d', d_value)
            
            cmd3 = f"python ttest.py --output_dir {result_dir}"
            
            temp_dir = tempfile.gettempdir()
            batch_filename = os.path.join(temp_dir, 'run_experiment.bat')
            
            batch_content = f"""@echo off
{cmd2}
{cmd3}
"""
            
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            # --- 3. 运行实验和T检验 ---
            result_code = os.system(f'"{batch_filename}"')
            
            if result_code != 0:
                print(f"实验组执行失败，返回码: {result_code}")
                return None
                
            # --- 4. 提取和处理结果 ---
            result_file = os.path.join(result_dir, 'ttest_results.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    ttest_result = json.load(f)
                
                spread_mean = ttest_result.get('spread_mean', np.nan)
                depth_mean = ttest_result.get('depth_mean', np.nan)
                volume_mean = ttest_result.get('volume_mean', np.nan)
                spread_p = ttest_result.get('spread_p', np.nan)
                depth_p = ttest_result.get('depth_p', np.nan)
                volume_p = ttest_result.get('volume_p', np.nan)

                # 计算效应类型
                effect_type = self.determine_effect_type(
                    spread_mean, depth_mean, volume_mean, 
                    spread_p, depth_p, volume_p
                )
                
                result = {
                    'k': float(k_value), 
                    'fee': float(fee_value), 
                    'max_slippage': float(slippage_value),
                    'num_hybrid_agents': int(num_agents_value), 
                    'fundamental_file_path': fundamental_file_value.strip(),
                    'seed': int(seed_value),
                    'effect_type': effect_type, 
                    **ttest_result
                }
                print(f"成功提取结果: ΔSpread={result['spread_mean']:.4f}, ΔDepth={result['depth_mean']:.2f}, ΔVolume={result['volume_mean']:.2f}, 效应: {effect_type}")
                
                self.append_result_to_csv(result)
                
                return result
            else:
                print(f"未找到结果文件: {result_file}")
                return None
                
        except Exception as e:
            print(f"实验组运行异常: {e}")
            return None

    def grid_search_sampling(self, param_ranges):
        """根据指定的范围和步长进行网格采样。"""
        print("\n=== 开始生成网格采样参数组合 ===")
        
        # 1. k值 (对数步进: 10^9, 10^11, 10^13, 10^15)
        k_values = [10**i for i in range(9, 16, 2)]
        
        # 2. fee值 (线性步进: 0.001, 0.004, 0.007, 0.010)
        fee_start = 0.001
        fee_stop = 0.010 
        fee_step = 0.003
        fee_values = [round(fee_start + i * fee_step, 4) for i in range(int(round((fee_stop - fee_start) / fee_step)) + 1)]
        fee_values = sorted(list(set([v for v in fee_values if v >= fee_start and v <= fee_stop])))

        # 3. max_slippage值 (线性步进: 0.01, 0.04, 0.07, 0.10)
        slippage_start = 0.01
        slippage_stop = 0.10 
        slippage_step = 0.03
        slippage_values = [round(slippage_start + i * slippage_step, 4) for i in range(int(round((slippage_stop - slippage_start) / slippage_step)) + 1)]
        slippage_values = sorted(list(set([v for v in slippage_values if v >= slippage_start and v <= slippage_stop])))
        
        # 4. num_hybrid_agents (100, 125, 150, 175, 200) - 从 param_ranges 获取
        agents_start = param_ranges.get('num_hybrid_agents', (100, 200))[0]
        agents_stop = param_ranges.get('num_hybrid_agents', (100, 200))[1]
        agents_step = 25 # 默认步长为25 (100, 125, 150, 175, 200)
        # agents_values = list(range(agents_start, agents_stop + agents_step, agents_step))
        agents_values = [100]
        
        # 5. fundamental_file_path (列表)
        fundamental_values = param_ranges.get('fundamental_file_path', [])
        
        # 6. seed值 (列表)
        seed_values = [int(s) for s in param_ranges.get('seed', [])]
        
        print(f"k值: {[f'{v:.0e}' for v in k_values]}")
        print(f"fee值: {[f'{v:.4f}' for v in fee_values]}")
        print(f"max_slippage值: {[f'{v:.2f}' for v in slippage_values]}")
        print(f"num_hybrid_agents值: {agents_values}")
        print(f"fundamental_file_path值: {fundamental_values}")
        print(f"seed值: {seed_values}")

        # 生成所有组合（六重循环）
        combinations = []
        for k in k_values:
            for fee in fee_values:
                for slippage in slippage_values:
                    for agents in agents_values:
                        for fund in fundamental_values:
                            for seed in seed_values:
                                combinations.append({
                                    'k': k,
                                    'fee': fee,
                                    'max_slippage': slippage,
                                    'num_hybrid_agents': agents,
                                    'fundamental_file_path': fund.strip(),
                                    'seed': seed
                                })
                        
        print(f"\n共生成 {len(combinations)} 个网格采样组合")
        return combinations

    def filter_existing_combinations(self, combinations, existing_df):
        """过滤掉已存在（精确匹配）的组合"""
        if existing_df.empty:
            return combinations

        required_cols = ['k', 'fee', 'max_slippage', 'num_hybrid_agents', 'fundamental_file_path', 'seed']
        if not all(col in existing_df.columns for col in required_cols):
            print("警告：现有结果中缺少关键参数列，无法进行精确过滤。将运行所有组合。")
            return combinations

        # 定义需要进行四舍五入的列（数值型列）
        numerical_cols = ['k', 'fee', 'max_slippage']
        # 定义不需要四舍五入的列（整数/字符串型列）
        other_cols = ['num_hybrid_agents', 'fundamental_file_path', 'seed']

        # 将现有结果转换为集合以便快速查找
        existing_tuples = set()
        for _, row in existing_df.iterrows():
            # 1. 对数值列进行四舍五入
            numerical_part = tuple(round(row[col], 6) for col in numerical_cols)
            # 2. 获取其他列的值
            other_part = tuple(row[col] for col in other_cols)
            # 3. 组合成完整的元组
            existing_tuples.add(numerical_part + other_part)

        filtered_combinations = []
        for combo in combinations:
            # 创建当前组合的元组（与上面相同的方式构造）
            current_tuple = (
                round(combo['k'], 6), 
                round(combo['fee'], 6), 
                round(combo['max_slippage'], 6), 
                combo['num_hybrid_agents'],
                combo['fundamental_file_path'], 
                combo['seed']
            )
            
            # 检查当前组合是否已存在
            if current_tuple not in existing_tuples:
                filtered_combinations.append(combo)
        
        num_excluded = len(combinations) - len(filtered_combinations)
        print(f"网格采样过滤后: 排除 {num_excluded} 个已有组合，剩余 {len(filtered_combinations)} 个新组合需要运行")
        return filtered_combinations
    
    def run_grid_search(self, param_ranges):
        """使用网格采样运行参数扫描"""
        self.results = []
        
        all_combinations = self.grid_search_sampling(param_ranges)
        
        # 确定 agents_values 的长度
        agents_values = [c['num_hybrid_agents'] for c in all_combinations]
        agents_values_count = len(set(agents_values))
        
        # 如果 num_hybrid_agents 只有一个值，则只在开始运行一次通用对照组（如果未运行过）
        if agents_values_count == 1 and not IS_CONTROL_GROUP_RUN:
            # 此时只需运行一个代表性的对照组（使用第一个组合的 agents, fund, seed）
            representative_combo = all_combinations[0]
            if not self.run_control_group(
                representative_combo['num_hybrid_agents'], 
                representative_combo['fundamental_file_path'], 
                representative_combo['seed']
            ):
                print("通用对照组运行失败，终止实验")
                return pd.DataFrame()
            # 标记为已运行，避免 run_single_experiment 再次运行
            global IS_CONTROL_GROUP_RUN
            IS_CONTROL_GROUP_RUN = True
        
        # 过滤已运行的组合
        param_combinations_to_run = self.filter_existing_combinations(
            all_combinations, self.existing_results
        )
        
        if not param_combinations_to_run:
            print("所有网格采样组合均已运行，无需重复实验。")
            return self.existing_results
        
        print(f"\n开始运行 {len(param_combinations_to_run)} 个新实验...")
        print(f"历史结果总数: {len(self.existing_results)}")
        
        successful_experiments = 0
        for i, params in enumerate(param_combinations_to_run):
            print(f"\n=== 新实验 {i+1}/{len(param_combinations_to_run)} (总进度: {len(self.existing_results) + i + 1}/{len(all_combinations)}) ===")
            
            # 如果 agents_values_count > 1，则 run_single_experiment 会在内部为每个组合运行对照组
            # 如果 agents_values_count == 1，则对照组已在前面运行，run_single_experiment 内部会跳过
            result = self.run_single_experiment(
                params['k'], 
                params['fee'], 
                params['max_slippage'], 
                params['num_hybrid_agents'], 
                params['fundamental_file_path'],
                params['seed']
            )
            
            if result:
                self.results.append(result)
                successful_experiments += 1
        
        print(f"\n新实验完成: {successful_experiments}/{len(param_combinations_to_run)} 个新实验成功")
        
        # 合并新旧结果
        if not self.existing_results.empty:
            all_results = pd.concat([self.existing_results, pd.DataFrame(self.results)], ignore_index=True)
        else:
            all_results = pd.DataFrame(self.results)
        
        return all_results

    def load_results_from_csv(self):
        """从CSV文件加载已有结果"""
        if os.path.exists(self.results_file):
            try:
                # 尝试以 UTF-8 编码读取，如果失败则尝试 GBK
                try:
                    df = pd.read_csv(self.results_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(self.results_file, encoding='gbk')

                if not df.empty:
                    # 确保参数列的类型正确
                    if 'k' in df.columns: df['k'] = df['k'].astype(float)
                    if 'fee' in df.columns: df['fee'] = df['fee'].astype(float)
                    if 'max_slippage' in df.columns: df['max_slippage'] = df['max_slippage'].astype(float)
                    if 'num_hybrid_agents' in df.columns: df['num_hybrid_agents'] = df['num_hybrid_agents'].astype(int) 
                    if 'seed' in df.columns: df['seed'] = df['seed'].astype(int)
                    
                print(f"从 {self.results_file} 加载了 {len(df)} 条已有结果")
                return df
            except Exception as e:
                print(f"加载CSV文件失败: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

def main():
    # 命令行参数模板。
    # 注意：-d, --num-hybrid-agents, -s, --fundamental-file-path, --r-bar 
    # 现在在运行前会被动态替换。
    base_cmd = """python -u abides.py -c rmsc03 -t BIT -d 20251028 -s 5678 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:40:00 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python -u abides.py -c rmsc04 -t BIT -d 20251028 -s 5678 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 1000000000 --fee 0.003 --max-slippage 0.01 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python ttest.py"""
    
    # 定义网格采样的范围
    param_ranges = {
        'k': (1e9, 1e15), 
        'fee': (0.001, 0.010), 
        'max_slippage': (0.01, 0.10), 
        # num_hybrid_agents: (起始值, 终止值)，步长在 grid_search_sampling 中定义为 25
        'num_hybrid_agents' : (100, 200), 
        # seed: 列表，值在 grid_search_sampling 中转换为 int
        'seed': ['5678', '91011', '12314'], 
        # fundamental_file_path: 列表，末尾的空格已被 .strip() 处理
        'fundamental_file_path': ['data/ETH1.xlsx ', 'data/BIT.xlsx '],
    }
    
    runner = ExperimentRunner(base_cmd, param_ranges)
    
    print("开始基于网格采样的参数扫描实验...")
    print(f"参数范围: {param_ranges}")
    
    # 运行网格采样
    results_df = runner.run_grid_search(param_ranges)
    
    if results_df.empty:
        print("没有可用的实验结果")
        return
    
    # 最终结果概览
    print("\n=== 实验最终结果概览 ===")
    if 'effect_type' in results_df.columns:
        print(results_df['effect_type'].value_counts())
        print("\n=== 各效应类型的参数组合统计 ===")
        print(results_df.groupby('effect_type').agg({
            'k': ['min', 'max', 'mean'],
            'fee': ['min', 'max', 'mean'],
            'max_slippage': ['min', 'max', 'mean'],
            'num_hybrid_agents': ['min', 'max', 'mean'],
        }))
    else:
        print("CSV中缺少 'effect_type' 列。")

    print("\n实验完成。所有结果（包含效应类型）已保存到 CSV 文件。")

if __name__ == "__main__":
    main()