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

class ExperimentRunner:
    def __init__(self, base_cmd):
        self.base_cmd = base_cmd
        self.results = []
        self.control_group_run = IS_CONTROL_GROUP_RUN
        self.results_file = 'experiment_results.csv'
        self.existing_results = self.load_results_from_csv()
        self.fixed_seeds = [1234, 121314, 91011, 5678]  # 固定的seed值
        
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
            
            cmd3 = f"python ttest.py --output_dir {result_dir}"
            
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

    def latin_hypercube_sampling(self, param_bounds, n_samples=32, exclude_combinations=None):
        """使用scipy.stats.qmc进行连续参数空间的拉丁超立方采样（seed固定为四个值）"""
        # 从参数边界中移除seed，因为我们将单独处理它
        continuous_params = {k: v for k, v in param_bounds.items() if k != 'seed'}
        param_names = list(continuous_params.keys())
        
        print(f"使用连续参数空间拉丁超立方采样生成 {n_samples} 个样本")
        print(f"连续参数边界: {continuous_params}")
        print(f"固定seed值: {self.fixed_seeds}")
        
        # 计算每个seed需要分配多少个样本
        samples_per_seed = max(1, n_samples // len(self.fixed_seeds))
        remaining_samples = n_samples % len(self.fixed_seeds)
        
        combinations = []
        
        # 为每个固定seed生成样本
        for i, seed_value in enumerate(self.fixed_seeds):
            # 分配样本数量
            current_samples = samples_per_seed
            if i < remaining_samples:
                current_samples += 1
            
            if current_samples <= 0:
                continue
                
            print(f"为seed={seed_value}生成{current_samples}个样本")
            
            # 创建拉丁超立方采样器
            sampler = qmc.LatinHypercube(d=len(param_names), seed=42+i)
            sample = sampler.random(n=current_samples)
            
            # 将[0,1]区间的样本映射到实际参数值
            for j in range(current_samples):
                combination = {'seed': seed_value}
                for k, param in enumerate(param_names):
                    lower_bound, upper_bound = continuous_params[param]
                    sample_value = sample[j, k]
                    
                    if param == 'k' or param == 'fee':
                        # k值和fee都使用对数均匀采样
                        log_lower = np.log10(lower_bound)
                        log_upper = np.log10(upper_bound)
                        log_value = log_lower + sample_value * (log_upper - log_lower)
                        value = 10 ** log_value
                    else:
                        # 其他参数使用线性均匀采样
                        value = lower_bound + sample_value * (upper_bound - lower_bound)
                    
                    combination[param] = value
                combinations.append(combination)
        
        # 如果生成的样本数量不足，用第一个seed补充
        if len(combinations) < n_samples:
            additional_needed = n_samples - len(combinations)
            print(f"需要补充 {additional_needed} 个样本，使用seed={self.fixed_seeds[0]}")
            
            sampler = qmc.LatinHypercube(d=len(param_names), seed=100)
            sample = sampler.random(n=additional_needed)
            
            for j in range(additional_needed):
                combination = {'seed': self.fixed_seeds[0]}
                for k, param in enumerate(param_names):
                    lower_bound, upper_bound = continuous_params[param]
                    sample_value = sample[j, k]
                    
                    if param == 'k' or param == 'fee':
                        log_lower = np.log10(lower_bound)
                        log_upper = np.log10(upper_bound)
                        log_value = log_lower + sample_value * (log_upper - log_lower)
                        value = 10 ** log_value
                    else:
                        value = lower_bound + sample_value * (upper_bound - lower_bound)
                    
                    combination[param] = value
                combinations.append(combination)
        
        # 过滤掉已有的组合（使用容差比较）
        if exclude_combinations is not None and not exclude_combinations.empty:
            filtered_combinations = []
            for combo in combinations:
                is_duplicate = False
                
                for _, existing_row in exclude_combinations.iterrows():
                    # 设置容差阈值
                    k_tolerance = 0.01 * existing_row['k']  # 1%容差
                    fee_tolerance = 0.01 * existing_row['fee']  # 1%容差
                    slippage_tolerance = 0.01 * existing_row['max_slippage']  # 1%容差
                    
                    if (abs(combo['k'] - existing_row['k']) <= k_tolerance and 
                        abs(combo['fee'] - existing_row['fee']) <= fee_tolerance and 
                        abs(combo['max_slippage'] - existing_row['max_slippage']) <= slippage_tolerance and 
                        combo['seed'] == existing_row['seed']):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_combinations.append(combo)
            
            combinations = filtered_combinations
            print(f"过滤后剩余 {len(combinations)} 个新组合")
            
            # 如果过滤后数量不足，补充新组合
            if len(combinations) < n_samples:
                additional_needed = n_samples - len(combinations)
                print(f"需要补充 {additional_needed} 个新组合")
                combinations.extend(self.generate_additional_combinations(
                    param_bounds, additional_needed, exclude_combinations, combinations
                ))
        
        self.validate_sampling_quality(combinations, continuous_params)
        
        return combinations[:n_samples]

    def generate_additional_combinations(self, param_bounds, n_needed, exclude_combinations, existing_combinations):
        """生成额外的非重复组合（连续空间版本，seed固定）"""
        additional_combinations = []
        attempts = 0
        max_attempts = n_needed * 20
        
        continuous_params = {k: v for k, v in param_bounds.items() if k != 'seed'}
        
        while len(additional_combinations) < n_needed and attempts < max_attempts:
            attempts += 1
            
            # 随机选择一个seed
            seed_value = random.choice(self.fixed_seeds)
            new_combo = {'seed': seed_value}
            
            for param, bounds in continuous_params.items():
                lower, upper = bounds
                if param == 'k' or param == 'fee':
                    # k值和fee在对数空间均匀采样
                    log_lower = np.log10(lower)
                    log_upper = np.log10(upper)
                    log_val = random.uniform(log_lower, log_upper)
                    new_combo[param] = 10 ** log_val
                else:
                    new_combo[param] = random.uniform(lower, upper)
            
            # 检查是否重复（使用容差比较）
            is_duplicate = False
            
            if not exclude_combinations.empty:
                for _, existing_row in exclude_combinations.iterrows():
                    k_tolerance = 0.01 * existing_row['k']
                    fee_tolerance = 0.01 * existing_row['fee']
                    slippage_tolerance = 0.01 * existing_row['max_slippage']
                    
                    if (abs(new_combo['k'] - existing_row['k']) <= k_tolerance and 
                        abs(new_combo['fee'] - existing_row['fee']) <= fee_tolerance and 
                        abs(new_combo['max_slippage'] - existing_row['max_slippage']) <= slippage_tolerance and 
                        new_combo['seed'] == existing_row['seed']):
                        is_duplicate = True
                        break
            
            # 检查是否与当前批次重复
            if not is_duplicate:
                for existing_combo in existing_combinations + additional_combinations:
                    k_tolerance = 0.01 * existing_combo['k']
                    fee_tolerance = 0.01 * existing_combo['fee']
                    slippage_tolerance = 0.01 * existing_combo['max_slippage']
                    
                    if (abs(new_combo['k'] - existing_combo['k']) <= k_tolerance and 
                        abs(new_combo['fee'] - existing_combo['fee']) <= fee_tolerance and 
                        abs(new_combo['max_slippage'] - existing_combo['max_slippage']) <= slippage_tolerance and 
                        new_combo['seed'] == existing_combo['seed']):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                additional_combinations.append(new_combo)
        
        return additional_combinations

    def validate_sampling_quality(self, combinations, continuous_params):
        """验证连续参数空间的采样质量"""
        if not combinations:
            print("没有可验证的组合")
            return
            
        df = pd.DataFrame(combinations)
        print("\n=== 连续参数空间采样质量验证 ===")
        
        for param, bounds in continuous_params.items():
            values = df[param]
            lower, upper = bounds
            
            if param == 'k' or param == 'fee':
                # 对k值和fee检查对数分布
                log_values = np.log10(values)
                log_lower = np.log10(lower)
                log_upper = np.log10(upper)
                coverage = (log_values.max() - log_values.min()) / (log_upper - log_lower) * 100
                print(f"{param}: 对数空间覆盖 {coverage:.1f}% ({10**log_values.min():.2e} - {10**log_values.max():.2e})")
            else:
                coverage = (values.max() - values.min()) / (upper - lower) * 100
                print(f"{param}: 线性空间覆盖 {coverage:.1f}% ({values.min():.4f} - {values.max():.4f})")
        
        # 显示seed分布
        seed_counts = df['seed'].value_counts().sort_index()
        print(f"\nseed分布: {dict(seed_counts)}")
        
        # 显示采样分布统计
        print("\n采样分布统计:")
        for param in continuous_params.keys():
            values = df[param]
            if param == 'k' or param == 'fee':
                # 对k值和fee显示对数统计
                log_values = np.log10(values)
                print(f"{param}: 对数均值={log_values.mean():.4f}, 对数标准差={log_values.std():.4f}, 范围=[{10**log_values.min():.2e}, {10**log_values.max():.2e}]")
            else:
                print(f"{param}: 均值={values.mean():.4f}, 标准差={values.std():.4f}, 范围=[{values.min():.4f}, {values.max():.4f}]")
        
        print("=== 验证完成 ===\n")
        
    def run_latin_hypercube_sampling(self, param_bounds, n_samples=64):
        """使用连续参数空间的拉丁超立方采样运行参数扫描（seed固定）"""
        self.results = []
        
        if not self.run_control_group():
            print("对照组运行失败，终止实验")
            return pd.DataFrame()
        
        # 生成拉丁超立方样本，排除已有组合
        param_combinations = self.latin_hypercube_sampling(
            param_bounds, n_samples, self.existing_results
        )
        
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

def main():
    base_cmd = """python -u abides.py -c rmsc03 -t ETH -d 20251110 -s 1235 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:40:00 --fundamental-file-path data/ETH1.xlsx 
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 1235 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 10000000 --fee 0.01 --max-slippage 0.1 --fundamental-file-path data/ETH1.xlsx
python ttest.py"""
    
    # 定义连续参数空间的上下界（不包含seed）
    param_bounds = {
        'k': (1e9, 1e12),        # 资金池规模: 1e9 到 1e12
        'fee': (0.0001, 0.1),     # 手续费: 0.05% 到 10%
        'max_slippage': (0.01, 1.0),  # 最大滑点: 1% 到 100%
    }
    
    runner = ExperimentRunner(base_cmd)
    
    print("开始连续参数空间拉丁超立方采样实验...")
    print(f"参数空间边界: {param_bounds}")
    print(f"固定seed值: {runner.fixed_seeds}")
    
    n_samples = 64
    
    results_df = runner.run_latin_hypercube_sampling(param_bounds, n_samples)
    
    if results_df.empty:
        print("没有可用的实验结果")
        return
    
    # 分析效应
    runner.analyze_coexistence_effect(results_df)
    
    print("实验完成")

if __name__ == "__main__":
    main()