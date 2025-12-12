import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tempfile
import json
import random
from scipy.stats import qmc

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

IS_CONTROL_GROUP_RUN = True

# --- 新增/保留的参数网格生成函数 ---
# 这是一个独立的函数，用于生成所有可能的参数组合网格点
def generate_parameter_grid(bounds, density):
    """
    Generate a fine parameter grid (k, fee, max_slippage).
    Removed 'seed_encoded' feature.
    """
    # 使用对数空间生成 k 值
    k_log = np.linspace(np.log10(bounds['k'][0]), np.log10(bounds['k'][1]), density)
    k_values = 10 ** k_log
    
    # 使用线性空间生成 fee 值
    fee_values = np.linspace(bounds['fee'][0], bounds['fee'][1], density)
    
    # 创建网格数据列表
    grid_data = []
    for k in k_values:
        for fee in fee_values:
            # 这里的 max_slippage 是固定值，直接使用 bounds 中的值
            grid_data.append({
                'k': k,
                'fee': fee,
                'max_slippage': bounds['max_slippage'][0], # max_slippage 边界通常是一个 (min, max) 元组，即使 min=max
            })
    
    # 返回包含所有网格点组合的 DataFrame
    return pd.DataFrame(grid_data), k_values, fee_values

GRID_DENSITY = 50 # 用于生成参数网格的密度

class ExperimentRunner:
    def __init__(self, base_cmd):
        self.base_cmd = base_cmd
        self.results = []
        self.control_group_run = IS_CONTROL_GROUP_RUN
        self.results_file = 'experiment_results.csv'
        self.existing_results = self.load_results_from_csv()
        self.fixed_seeds = [12315]  # 固定的seed值
        
    def replace_parameter(self, cmd, param_name, param_value):
        """通用参数替换函数"""
        pattern1 = rf'{param_name}\s+\S+'
        replacement1 = f'{param_name} {param_value}'
        cmd = re.sub(pattern1, replacement1, cmd)
        
        pattern2 = rf'{param_name}=\S+'
        replacement2 = f'{param_name}={param_value}'
        cmd = re.sub(pattern2, replacement2, cmd)
        
        return cmd
        
    def run_control_group(self):
        """运行对照组（只需运行一次）"""
        if self.control_group_run:
            print("对照组已运行，跳过...")
            return True
            
        try:
            print("运行对照组...")
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            control_cmd = lines[0]
            
            batch_content = f"""@echo off
{control_cmd}
"""
            
            with open('run_control.bat', 'w') as f:
                f.write(batch_content)
            
            result_code = os.system('run_control.bat')
            
            if result_code != 0:
                print(f"对照组执行失败，返回码: {result_code}")
                return False
                
            self.control_group_run = True
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
                result_df.to_csv(self.results_file, mode='a', header=False, index=False)
            else:
                result_df.to_csv(self.results_file, index=False)
            
            print(f"结果已追加到 {self.results_file}")
            return True
        except Exception as e:
            print(f"写入CSV文件失败: {e}")
            return False
    
    def run_single_experiment(self, k_value, fee_value, slippage_value, seed_value):
        """运行单次实验（实验组）并提取t检验结果"""
        # ... (与原代码保持一致) ...
        try:
            print(f"运行实验组: k={k_value:.2e}, fee={fee_value:.4f}, slippage={slippage_value:.3f}, seed={seed_value}")
            
            result_dir = "ttest_results"
            os.makedirs(result_dir, exist_ok=True)
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            cmd2 = lines[1]
            
            cmd2 = self.replace_parameter(cmd2, '-k', int(k_value))
            cmd2 = self.replace_parameter(cmd2, '--fee', fee_value)
            cmd2 = self.replace_parameter(cmd2, '--max-slippage', slippage_value)
            cmd2 = self.replace_parameter(cmd2, '-s', seed_value)
            
            cmd3 = f"python ttest.py"
            
            batch_content = f"""@echo off
{cmd2}
{cmd3}
"""
            
            batch_filename = os.path.join(result_dir, 'run_experiment.bat')
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            result_code = os.system(f'"{batch_filename}"')
            
            if result_code != 0:
                print(f"实验组执行失败: k={k_value:.2e}, fee={fee_value:.4f}, slippage={slippage_value:.3f}, seed={seed_value}, 返回码: {result_code}")
                return None
                
            result_file = os.path.join(result_dir, 'ttest_results.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    ttest_result = json.load(f)
                
                result = {
                    'k': k_value,
                    'fee': fee_value,
                    'max_slippage': slippage_value,
                    'seed': seed_value,
                    **ttest_result
                }
                print(f"成功提取结果: ΔSpread={result['spread_mean']:.4f}, ΔDepth={result['depth_mean']:.2f}, ΔVolume={result['volume_mean']:.2f}")
                
                self.append_result_to_csv(result)
                
                return result
            else:
                print(f"未找到结果文件: {result_file}")
                return None
                
        except Exception as e:
            print(f"实验组运行异常: k={k_value:.2e}, fee={fee_value:.4f}, slippage={slippage_value:.3f}, seed={seed_value}: {e}")
            return None

    # --- 修改后的采样函数：网格随机均匀采样 ---
    def grid_random_sampling(self, parameter_grid_df, n_samples=32, exclude_combinations=None):
        """
        从生成的参数网格点中进行随机均匀采样（k, fee, max_slippage）
        并与固定的 seed 值组合。
        """
        # 网格中的所有 'k', 'fee', 'max_slippage' 组合
        grid_combinations = parameter_grid_df[['k', 'fee', 'max_slippage']].to_dict('records')
        
        if not grid_combinations:
            print("参数网格为空，无法采样。")
            return []
            
        print(f"参数网格共有 {len(grid_combinations)} 个点。")
        print(f"需要生成 {n_samples} 个样本，固定seed值: {self.fixed_seeds}")
        
        combinations = []
        
        # 统计已排除的组合（用于去重）
        excluded_set = set()
        if exclude_combinations is not None and not exclude_combinations.empty:
            for _, row in exclude_combinations.iterrows():
                # 使用接近的字符串表示作为 key，因为浮点数直接比较有风险
                key = (f"{row['k']:.8e}", f"{row['fee']:.8f}", f"{row['max_slippage']:.8f}", row['seed'])
                excluded_set.add(key)
        
        # 使用随机选择进行采样，直到达到 n_samples
        # 注意：这里是从网格点中**随机均匀采样**，允许重复采样网格点（如果 n_samples 很大）
        # 且与 seed 的组合也可能重复，但我们会检查历史结果。
        
        # 首先生成足够的 (k, fee, max_slippage) 网格点 + seed 组合
        for _ in range(n_samples * 2): # 多采样一些以应对去重，最大尝试次数为 2*n_samples
            
            # 随机选择一个网格点
            grid_point = random.choice(grid_combinations)
            
            # 随机选择一个固定的 seed
            seed_value = random.choice(self.fixed_seeds)
            
            combination = {
                'k': grid_point['k'],
                'fee': grid_point['fee'],
                'max_slippage': grid_point['max_slippage'],
                'seed': seed_value
            }
            
            # 检查是否是历史重复组合
            k_str = f"{combination['k']:.8e}"
            fee_str = f"{combination['fee']:.8f}"
            slippage_str = f"{combination['max_slippage']:.8f}"
            current_key = (k_str, fee_str, slippage_str, combination['seed'])
            
            # 使用 set 进行高效查重
            is_duplicate = current_key in excluded_set
            if not is_duplicate:
                # 检查是否与当前批次重复
                for existing_combo in combinations:
                    if (abs(existing_combo['k'] - combination['k']) < 1e-10 and
                        abs(existing_combo['fee'] - combination['fee']) < 1e-10 and
                        abs(existing_combo['max_slippage'] - combination['max_slippage']) < 1e-10 and
                        existing_combo['seed'] == combination['seed']):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                combinations.append(combination)
                # 将新组合添加到排除集，以防止后续在同一批次中重复采样
                excluded_set.add(current_key) 
                if len(combinations) >= n_samples:
                    break

        print(f"成功生成 {len(combinations)} 个不重复的新组合（从网格中随机采样）")
        
        # 这里的验证函数针对连续采样设计，对于网格点采样，覆盖率检查意义不大，但保留统计信息
        # self.validate_sampling_quality(combinations, parameter_bounds_for_validation) 
        
        return combinations[:n_samples]

    # 原有的 latin_hypercube_sampling 和 generate_additional_combinations 函数被删除或替换，
    # 但为了完整性，原有的 validate_sampling_quality 被保留并进行适应性修改。

    def validate_sampling_quality(self, combinations, continuous_params):
        """验证连续参数空间的采样质量（现用于网格点采样后的统计）"""
        if not combinations:
            print("没有可验证的组合")
            return
            
        df = pd.DataFrame(combinations)
        print("\n=== 网格随机采样统计 ===")
        
        # 连续参数空间用于统计（即使是网格点，也检查其范围）
        for param, bounds in continuous_params.items():
            values = df[param]
            lower, upper = bounds
            
            if lower == upper:
                # 针对固定值
                print(f"{param}: 固定值 {lower:.2e}")
            else:
                # 针对采样的网格点范围
                # 注意：这里显示的范围可能小于 generate_parameter_grid 定义的整个范围
                coverage = (values.max() - values.min()) / (upper - lower) * 100 if upper > lower else 100.0
                print(f"{param}: 线性空间覆盖 {coverage:.1f}% ({values.min():.4e} - {values.max():.4e})")
        
        # 显示seed分布
        seed_counts = df['seed'].value_counts().sort_index()
        print(f"\nseed分布: {dict(seed_counts)}")
        
        # 显示采样分布统计
        print("\n采样分布统计:")
        for param in continuous_params.keys():
            values = df[param]
            print(f"{param}: 均值={values.mean():.4e}, 标准差={values.std():.4e}, 范围=[{values.min():.4e}, {values.max():.4e}]")
        
        print("=== 统计完成 ===\n")
        
    # --- 修改后的运行函数 ---
    def run_grid_sampling(self, parameter_grid_df, param_bounds_for_validation, n_samples=64):
        """使用网格随机采样运行参数扫描（seed固定）"""
        self.results = []
        
        if not self.run_control_group():
            print("对照组运行失败，终止实验")
            return pd.DataFrame()
        
        # 生成网格随机样本，排除已有组合
        param_combinations = self.grid_random_sampling(
            parameter_grid_df, n_samples, self.existing_results
        )
        
        # 对采样的点进行统计验证 (使用 param_bounds_for_validation 作为边界信息)
        self.validate_sampling_quality(param_combinations, param_bounds_for_validation)
        
        if not param_combinations:
            print("没有新的参数组合需要实验")
            return self.existing_results
        
        print(f"开始运行 {len(param_combinations)} 个新实验...")
        print(f"已有 {len(self.existing_results)} 条历史结果")
        
        successful_experiments = 0
        for i, params in enumerate(param_combinations):
            print(f"\n=== 新实验 {i+1}/{len(param_combinations)} ===")
            result = self.run_single_experiment(
                params['k'], 
                params['fee'], 
                params['max_slippage'], 
                params['seed']
            )
            if result:
                self.results.append(result)
                successful_experiments += 1
        
        print(f"\n新实验完成: {successful_experiments}/{len(param_combinations)} 个新实验成功")
        
        # 合并新旧结果
        if not self.existing_results.empty:
            all_results = pd.concat([self.existing_results, pd.DataFrame(self.results)], ignore_index=True)
        else:
            all_results = pd.DataFrame(self.results)
        
        return all_results

    # ... (load_results_from_csv 和 analyze_coexistence_effect 保持不变) ...

    def load_results_from_csv(self):
        """从CSV文件加载已有结果"""
        if os.path.exists(self.results_file):
            try:
                df = pd.read_csv(self.results_file)
                if not df.empty:
                    df['k'] = df['k'].astype(float)
                    df['fee'] = df['fee'].astype(float)
                    df['max_slippage'] = df['max_slippage'].astype(float)
                    df['seed'] = df['seed'].astype(int)
                print(f"从 {self.results_file} 加载了 {len(df)} 条已有结果")
                return df
            except Exception as e:
                print(f"加载CSV文件失败: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def analyze_coexistence_effect(self, results_df):
        """分析共生/挤出效应"""
        if results_df.empty:
            print("没有结果数据可供分析")
            return
            
        print("\n=== 共生/挤出效应分析 ===")
        
        for idx, row in results_df.iterrows():
            k, fee, slippage, seed = row['k'], row['fee'], row['max_slippage'], row['seed']
            
            spread_mean = row['spread_mean']
            depth_mean = row['depth_mean'] 
            volume_mean = row['volume_mean']
            
            spread_p = row['spread_p']
            depth_p = row['depth_p']
            volume_p = row['volume_p']
            
            if (spread_p > 0.05 or abs(spread_mean) <= 0.002) and depth_p < 0.05 and volume_p < 0.05 and depth_mean > 0 and volume_mean > 0:
                effect_type = "共生效应"
            elif spread_p < 0.05 and abs(spread_mean) > 0.005 and (depth_p > 0.05 or depth_mean <= 0 or volume_p > 0.05 or volume_mean <= 0):
                effect_type = "挤出效应"
            else:
                effect_type = "混合效应"
            
            print(f"k={k:.2e}, fee={fee:.4f}, slippage={slippage:.3f}, seed={seed}: {effect_type}")
            print(f"  价差: Δ={spread_mean:.4f}, p={spread_p:.3g}")
            print(f"  深度: Δ={depth_mean:.2f}, p={depth_p:.3g}")
            print(f"  成交量: Δ={volume_mean:.2f}, p={volume_p:.3g}")
            print()

# --- 主函数修改：调用网格生成和网格随机采样 ---
def main():
    base_cmd = """python -u abides.py -c rmsc03 -t ETH -d 20251110 -s 12315 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:40:00 --num-hybrid-agents 100 --fundamental-file-path data/ETH1.xlsx --r-bar 3611.0
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 12315 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 1000000000 --fee 0.003 --max-slippage 0.05 --num-hybrid-agents 100 --fundamental-file-path data/ETH1.xlsx --r-bar 3611.0
python ttest.py"""
    
    # 定义连续参数空间的上下界
    param_bounds = {
        'k': (1e10, 1e11),     # 资金池规模: 1e10 到 1e11
        'fee': (0.001, 0.01),  # 手续费: 0.001 到 0.01
        'max_slippage': (0.05, 0.05), # 最大滑点: 固定为 0.05
    }
    
    runner = ExperimentRunner(base_cmd)
    
    # 1. 生成参数网格
    grid_df, k_values, fee_values = generate_parameter_grid(param_bounds, GRID_DENSITY)
    print(f"已生成包含 {len(grid_df)} 个点的参数网格 (k: {len(k_values)} 个点, fee: {len(fee_values)} 个点)")

    print("开始参数网格随机采样实验...")
    print(f"参数空间边界: {param_bounds}")
    print(f"固定seed值: {runner.fixed_seeds}")
    
    n_samples = 128
    
    # 2. 调用网格随机采样运行
    # 传入 grid_df 作为采样空间，param_bounds 作为验证时的边界信息
    results_df = runner.run_grid_sampling(grid_df, param_bounds, n_samples)
    
    if results_df.empty:
        print("没有可用的实验结果")
        return
    
    # 分析效应
    runner.analyze_coexistence_effect(results_df)
    
    print("实验完成")

if __name__ == "__main__":
    main()