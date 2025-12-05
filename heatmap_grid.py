import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tempfile
import json
import random

# 全局设置（保持不变）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ExperimentRunner:
    def __init__(self, base_cmd, param_ranges):
        self.base_cmd = base_cmd
        self.results = []
        # 对照组结果和状态管理
        self.control_results_file = 'control_group_status_v3_crypto.json'
        self.control_group_status = self.load_control_status()
        
        self.results_file = 'experiment_results_grid_search_v3_crypto.csv' 
        self.param_ranges = param_ranges
        self.existing_results = self.load_results_from_csv()
        
    def load_control_status(self):
        """从 JSON 文件加载已成功运行的对照组状态"""
        if os.path.exists(self.control_results_file):
            try:
                with open(self.control_results_file, 'r') as f:
                    # 使用 frozenset 来存储参数组合的哈希值
                    # 注意：加载时需要处理 fundamental_file_path 的空格
                    data = json.load(f)
                    # 确保 fundamental_file_path 始终被 strip()
                    status_set = set()
                    for d in data:
                        if 'fundamental_file_path' in d:
                            d['fundamental_file_path'] = d['fundamental_file_path'].strip()
                        status_set.add(frozenset(d.items()))
                    return status_set
            except Exception as e:
                print(f"加载对照组状态失败: {e}")
                return set()
        return set()

    def save_control_status(self, params):
        """将成功运行的对照组参数组合保存到 JSON 文件"""
        # 确保保存前进行 strip()
        params_to_save = params.copy()
        params_to_save['fundamental_file_path'] = params_to_save['fundamental_file_path'].strip()
        
        # 将 frozenset 转换回 list/dict 以便 JSON 存储
        # 先将当前状态（已清理空格）转换为列表
        current_list = [dict(s) for s in self.control_group_status]
        
        # 检查是否已存在（理论上 run_control_group 已经检查过）
        params_frozen = frozenset(params_to_save.items())
        if params_frozen not in self.control_group_status:
            current_list.append(params_to_save)
            self.control_group_status.add(params_frozen)

        try:
            with open(self.control_results_file, 'w') as f:
                json.dump(current_list, f, indent=4)
        except Exception as e:
            print(f"保存对照组状态失败: {e}")
            
    def get_control_params(self, combination):
        """从实验组组合中提取对照组所需的参数"""
        return {
            'num_hybrid_agents': combination['num_hybrid_agents'], 
            'fundamental_file_path': combination['fundamental_file_path'].strip(), 
            'seed': combination['seed']
        }

    def replace_parameter(self, cmd, param_name, param_value):
        """通用参数替换函数"""
        # 处理 -param_name value 形式
        pattern1 = rf'{re.escape(param_name)}\s+\S+'
        replacement1 = f'{param_name} {param_value}'
        cmd = re.sub(pattern1, replacement1, cmd)
        
        # 处理 param_name=value 形式
        pattern2 = rf'{re.escape(param_name)}=\S+'
        replacement2 = f'{param_name}={param_value}'
        cmd = re.sub(pattern2, replacement2, cmd)
        
        return cmd
        
    def run_control_group(self, num_agents_value, fundamental_file_value, seed_value):
        """运行对照组（如果尚未运行）"""
        
        control_params = {
            'num_hybrid_agents': int(num_agents_value),
            'fundamental_file_path': fundamental_file_value.strip(),
            'seed': int(seed_value)
        }
        control_params_frozen = frozenset(control_params.items())

        if control_params_frozen in self.control_group_status:
            # print(f"对照组已运行，跳过: {control_params}")
            return True
            
        try:
            print(f"\n--- 运行新的对照组: {control_params} ---")
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            if not lines:
                print("错误: base_cmd为空")
                return False
                
            control_cmd = lines[0]

            # 替换核心参数
            control_cmd = self.replace_parameter(control_cmd, '--num-hybrid-agents', int(num_agents_value))
            control_cmd = self.replace_parameter(control_cmd, '--fundamental-file-path', fundamental_file_value)
            control_cmd = self.replace_parameter(control_cmd, '-s', seed_value)

            # 设置 r_bar
            r_bar = 0.0
            fund_path_stripped = fundamental_file_value.strip()
            if fund_path_stripped == 'data/ETH1.xlsx':
                r_bar = 3611.0
            elif fund_path_stripped == 'data/BIT.xlsx':
                r_bar = 113994.6305
                
            control_cmd = self.replace_parameter(control_cmd, '--r-bar', r_bar)
            
            
            temp_dir = tempfile.gettempdir()
            batch_filename = os.path.join(temp_dir, 'run_control.bat')
            
            batch_content = f"""@echo off
{control_cmd}
"""
            
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            # 使用 os.system 执行命令
            result_code = os.system(f'"{batch_filename}"')
            
            if result_code != 0:
                print(f"对照组执行失败，返回码: {result_code}")
                return False
                
            # 成功运行后记录状态
            self.save_control_status(control_params)
            print("对照组运行完成")
            return True
                
        except Exception as e:
            print(f"对照组运行异常: {e}")
            return False

    def append_result_to_csv(self, result):
        """将单次实验结果追加到CSV文件"""
        try:
            result_df = pd.DataFrame([result])
            
            if os.path.exists(self.results_file):
                # 重新加载以确保最新的 existing_df
                existing_df = self.load_results_from_csv()
                
                # 检查当前结果是否已存在（精确匹配所有参数）
                is_duplicate = existing_df[
                    (existing_df['k'].round(6) == round(result['k'], 6)) & 
                    (existing_df['fee'].round(6) == round(result['fee'], 6)) & 
                    (existing_df['max_slippage'].round(6) == round(result['max_slippage'], 6)) & 
                    (existing_df['num_hybrid_agents'] == result['num_hybrid_agents']) & 
                    (existing_df['fundamental_file_path'] == result['fundamental_file_path'].strip()) & 
                    (existing_df['seed'] == result['seed'])
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
        """内部函数：根据 T 检验结果判断效应类型"""
        # 定义判断标准 (假设显著性水平 alpha = 0.05)
        
        # 共生效应：价差不显著变化 或 略微变小，同时深度和成交量显著增加
        is_spread_stable = (spread_p > 0.05) or (abs(spread_mean) <= 0.002)
        is_depth_increase = (depth_p < 0.05) and (depth_mean > 0)
        is_volume_increase = (volume_p < 0.05) and (volume_mean > 0)
        
        if is_spread_stable and is_depth_increase and is_volume_increase:
            return "共生效应"
        
        # 挤出效应：价差显著变大，同时深度或成交量没有显著增加
        is_spread_worse = (spread_p < 0.05) and (spread_mean > 0.005)
        is_depth_not_better = (depth_p > 0.05) or (depth_mean <= 0)
        is_volume_not_better = (volume_p > 0.05) or (volume_mean <= 0)
        
        if is_spread_worse and is_depth_not_better and is_volume_not_better:
            return "挤出效应"
        elif is_spread_worse and (is_depth_increase or is_volume_increase):
             # 价差显著变大，但深度或成交量也显著增加
             return "混合效应 (恶化价差/增加流动性)"
        else:
            return "混合效应"

    def run_single_experiment(self, k_value, fee_value, slippage_value, num_agents_value, fundamental_file_value, seed_value):
        """运行单次实验（实验组）并提取 T 检验结果和效应类型"""
        try:
            print(f"-> 运行实验组: k={k_value:.2e}, fee={fee_value:.4f}, slippage={slippage_value:.3f}")
            
            result_dir = "ttest_results"
            os.makedirs(result_dir, exist_ok=True)
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            cmd2 = lines[1]
            
            # 替换所有参数
            cmd2 = self.replace_parameter(cmd2, '-k', int(k_value))
            cmd2 = self.replace_parameter(cmd2, '--fee', fee_value)
            cmd2 = self.replace_parameter(cmd2, '--max-slippage', slippage_value)
            cmd2 = self.replace_parameter(cmd2, '--num-hybrid-agents', int(num_agents_value))
            cmd2 = self.replace_parameter(cmd2, '--fundamental-file-path', fundamental_file_value)
            cmd2 = self.replace_parameter(cmd2, '-s', seed_value)

            # 设置 r_bar
            r_bar = 0.0
            fund_path_stripped = fundamental_file_value.strip()
            if fund_path_stripped == 'data/ETH1.xlsx':
                r_bar = 3611.0
            elif fund_path_stripped == 'data/BIT.xlsx':
                r_bar = 113994.6305
                
            cmd2 = self.replace_parameter(cmd2, '--r-bar', r_bar)
                
            cmd3 = f"python ttest.py --output_dir {result_dir}"
            
            temp_dir = tempfile.gettempdir()
            batch_filename = os.path.join(temp_dir, 'run_experiment.bat')
            
            batch_content = f"""@echo off
{cmd2}
{cmd3}
"""
            
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            result_code = os.system(f'"{batch_filename}"')
            
            if result_code != 0:
                print(f"实验组执行失败，返回码: {result_code}")
                return None
                
            result_file = os.path.join(result_dir, 'ttest_results.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    ttest_result = json.load(f)
                
                # --- 效应计算开始 ---
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
                # --- 效应计算结束 ---
                
                result = {
                    'k': float(k_value), 
                    'fee': float(fee_value), 
                    'max_slippage': float(slippage_value),
                    'num_hybrid_agents': int(num_agents_value), 
                    'fundamental_file_path': fundamental_file_value.strip(),
                    'seed': int(seed_value),
                    'effect_type': effect_type, # 新增效应类型
                    **ttest_result
                }
                print(f"-> 结果: ΔSpread={result['spread_mean']:.4f}, ΔDepth={result['depth_mean']:.2f}, ΔVolume={result['volume_mean']:.2f}, 效应: {effect_type}")
                
                # 将结果保存到CSV
                self.append_result_to_csv(result)
                
                return result
            else:
                print(f"未找到结果文件: {result_file}")
                return None
                
        except Exception as e:
            print(f"实验组运行异常: {e}")
            return None

    def grid_search_sampling(self, param_ranges):
        """
        根据指定的范围和步长进行网格采样。
        执行顺序调整为: 
        Agents -> Fund_Path -> Seed -> K -> Fee -> Slippage
        以减少对照组 (Agents, Fund_Path, Seed) 的重复运行。
        """
        print("\n=== 开始生成网格采样参数组合 ===")
        
        # 1. k值 (对数步进: 10^9, 10^11, 10^13, 10^15)
        k_values = [10**i for i in range(9, 13, 1)]
        # k_values = [10**11]
        
        # 2. fee值 (线性步进: 0.001, 0.004, 0.007, 0.010)
        fee_start = 0.001
        fee_stop = 0.010 
        fee_step = 0.003
        fee_values = [round(fee_start + i * fee_step, 4) for i in range(int(round((fee_stop - fee_start) / fee_step)) + 1)]
        fee_values = sorted(list(set([v for v in fee_values if v >= fee_start and v <= fee_stop])))
        # fee_values = [0.003]

        # 3. max_slippage值 (线性步进: 0.01, 0.04, 0.07, 0.10)
        slippage_start = 0.01
        slippage_stop = 0.10 
        slippage_step = 0.03
        slippage_values = [round(slippage_start + i * slippage_step, 4) for i in range(int(round((slippage_stop - slippage_start) / slippage_step)) + 1)]
        slippage_values = sorted(list(set([v for v in slippage_values if v >= slippage_start and v <= slippage_stop])))
        # slippage_values = [0.05]
        
        # 4. num_hybrid_agents 
        agents_values = param_ranges.get('num_hybrid_agents', []) 
        if not agents_values: agents_values = [100]

        # 5. fundamental_file_path (列表)
        fundamental_values = param_ranges.get('fundamental_file_path', [])
        
        # 6. seed值 (列表)
        seed_values = param_ranges.get('seed', [])
        
        print(f"Agents值: {agents_values}")
        print(f"Fund_Path值: {[f.strip() for f in fundamental_values]}")
        print(f"Seed值: {seed_values}")
        print(f"k值: {[f'{v:.0e}' for v in k_values]}")
        print(f"fee值: {[f'{v:.4f}' for v in fee_values]}")
        print(f"max_slippage值: {[f'{v:.2f}' for v in slippage_values]}")
        
        # 生成所有组合（调整后的六重循环）
        combinations = []
        # 外层循环：对照组参数
        for agents in agents_values:
            for fund in fundamental_values:
                for seed in seed_values:
                    # 内层循环：实验组参数
                    for k in k_values:
                        for fee in fee_values:
                            for slippage in slippage_values:
                                combinations.append({
                                    'k': k,
                                    'fee': fee,
                                    'max_slippage': slippage,
                                    'num_hybrid_agents': agents,
                                    'fundamental_file_path': fund, # 保持空格以便替换原始cmd
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
            # 1. 对数值列进行四舍五入 (统一到小数点后 6 位)
            numerical_part = tuple(round(row[col], 6) for col in numerical_cols)
            # 2. 获取其他列的值 (fundamental_file_path 已经过 strip() 存储)
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
                combo['fundamental_file_path'].strip(), # 注意这里需要 strip() 来匹配已存储的结果
                combo['seed']
            )
            
            # 检查当前组合是否已存在
            if current_tuple not in existing_tuples:
                filtered_combinations.append(combo)
        
        num_excluded = len(combinations) - len(filtered_combinations)
        print(f"网格采样过滤后: 排除 {num_excluded} 个已有组合，剩余 {len(filtered_combinations)} 个新组合需要运行")
        return filtered_combinations
    
    def run_grid_search(self, param_ranges):
        """使用网格采样运行参数扫描，并在需要时运行对照组"""
        self.results = []
        
        all_combinations = self.grid_search_sampling(param_ranges)
        
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
        
        # 使用当前对照组参数的元组来跟踪当前的对照组状态
        current_control_params = None 
        
        for i, params in enumerate(param_combinations_to_run):
            
            # 提取对照组所需参数
            control_params = self.get_control_params(params)
            
            # 检查是否需要运行新的对照组
            if current_control_params is None or current_control_params != control_params:
                # 需要运行新的对照组
                control_run_success = self.run_control_group(
                    params['num_hybrid_agents'], 
                    params['fundamental_file_path'], 
                    params['seed']
                )
                
                if not control_run_success:
                    print(f"致命错误: 对照组运行失败，终止当前对照组下的所有实验组。对照组参数: {control_params}")
                    # 由于循环顺序已优化，这里可以跳过当前对照组下的所有剩余组合（如果它们是连续的）
                    # 但为了安全，我们只跳过当前的实验组。
                    current_control_params = control_params # 更新状态，但不标记为成功运行
                    continue
                
                # 更新当前成功运行的对照组参数
                current_control_params = control_params 
                print(f"\n=== 运行新对照组下的第 {i+1} 个实验 ===")
            else:
                # 沿用上一个对照组的结果，不打印对照组运行信息，节省屏幕输出
                print(f"\n=== 沿用对照组 (Agents={control_params['num_hybrid_agents']}, Fund={control_params['fundamental_file_path']}, Seed={control_params['seed']}) 的第 {i+1} 个实验 ===")

            # 运行实验组
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
                # 显式指定 dtype 避免警告和类型不匹配
                dtype_spec = {
                    'k': float, 
                    'fee': float, 
                    'max_slippage': float,
                    'num_hybrid_agents': int,
                    'seed': int,
                    'fundamental_file_path': str
                }
                df = pd.read_csv(self.results_file, dtype=dtype_spec)
                
                if not df.empty:
                    # 确保 fundamental_file_path 已清理空格，以便与新的组合参数匹配
                    df['fundamental_file_path'] = df['fundamental_file_path'].astype(str).str.strip()
                    
                print(f"从 {self.results_file} 加载了 {len(df)} 条已有结果")
                return df
            except Exception as e:
                print(f"加载CSV文件失败: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

def main():
    # 命令行参数模板。
    # -c rmsc03 是对照组 (Control Group)
    # -c rmsc04 是实验组 (Experiment Group)
    base_cmd = """python -u abides.py -c rmsc03 -t ETH -d 20251028 -s 5678 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:35:00 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python -u abides.py -c rmsc04 -t ETH -d 20251028 -s 5678 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:35:00 -k 1000000000 --fee 0.003 --max-slippage 0.01 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python ttest.py"""
    
    # 定义网格采样的范围
    param_ranges = {
        'k': (1e9, 1e13), 
        'fee': (0.001, 0.010), 
        'max_slippage': (0.01, 0.10), 
        'num_hybrid_agents' : [100, 150, 200], # 增加 agents 以测试循环顺序优化
        'seed': [5678], 
        'fundamental_file_path': ['data/ETH1.xlsx '], # 保持尾随空格
    }

    
    runner = ExperimentRunner(base_cmd, param_ranges)
    
    print("开始基于网格采样的参数扫描实验（优化对照组运行顺序）...")
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
    else:
        print("CSV中缺少 'effect_type' 列。")

    print("\n实验完成。所有结果（包含效应类型）已保存到 CSV 文件，对照组状态已保存到 JSON 文件。")

if __name__ == "__main__":
    main()