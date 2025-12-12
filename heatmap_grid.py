import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from itertools import product # 用于生成参数的笛卡尔积（网格）
from typing import Dict, Any, List, Tuple

# 设置中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentRunner:
    # 🛠️ __init__ 签名更改：不再接收 k_values
    def __init__(self, base_cmd: str): 
        self.base_cmd = base_cmd
        self.results = []
        self.results_file = 'experiment_results.csv'
        self.existing_results = self.load_results_from_csv()
        
        # 将 seed 视为一个要遍历的离散参数
        self.discrete_seeds = [12315]
        
        # 🛠️ 只关心 seed 生成需要运行对照组的 seed 列表
        self.control_seeds_to_run = self.discrete_seeds
        
        # 🛠️ 记录已运行的对照组 seed
        if not self.existing_results.empty:
            # 假设对照组记录的 fee 和 max_slippage 都是默认值或不重要，只关心 seed
            # 假设对照组有一个特定的标记（例如 fee=10000），用于区分实验组
            control_results = self.existing_results[
                self.existing_results['fee'] == 10000.0 # 假设对照组 fee=10000 是一个独特的标记
            ]
            self.ran_control_seeds = set(
                control_results['seed'].astype(int)
            )
        else:
            self.ran_control_seeds = set()
            
    def replace_parameter(self, cmd: str, param_name: str, param_value: Any) -> str:
        """通用参数替换函数"""
        param_value_str = str(param_value)
        
        # 替换 --param value 形式
        pattern1 = rf'{param_name}\s+\S+'
        replacement1 = f'{param_name} {param_value_str}'
        cmd = re.sub(pattern1, replacement1, cmd)
        
        # 替换 --param=value 形式 (如果存在)
        pattern2 = rf'{param_name}=\S+'
        replacement2 = f'{param_name}={param_value_str}'
        cmd = re.sub(pattern2, replacement2, cmd)
        
        return cmd
        
    # 🛠️ 方法签名更改：删除 k_value 参数，只接受 seed_value
    def run_control_group_with_seed(self, seed_value: int) -> bool:
        """运行对照组 (rmsc03) 并将 -s 参数替换为指定值"""
        
        # 🛠️ 检查 seed 是否已运行/记录
        if seed_value in self.ran_control_seeds:
            print(f"✅ 对照组 (seed={seed_value}) 已运行或已记录，跳过...")
            # 对照组已运行，返回 True
            return True 
            
        try:
            print(f"🚀 运行对照组 (rmsc03) with seed={seed_value}...")
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            control_cmd_base = lines[0] # rmsc03

            # 🛠️ 替换对照组命令中的 -s 参数
            control_cmd = self.replace_parameter(control_cmd_base, '-s', seed_value)
            
            # **注意**: 虽然不再循环 k，但为了 t-test 的对比逻辑，rmsc03 必须与 rmsc04 具有相同的 k 设置。
            # 但是，如果 rmsc03 自身不使用 k 参数，我们不需要替换。
            # 鉴于您的 base_cmd 中 rmsc03 没有 -k，我们保持原样。
            
            batch_content = f"""@echo off
{control_cmd}
"""
            
            # 使用特定 seed 的文件名
            batch_filename = f'run_control_s{seed_value}.bat'
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            result_code = os.system(batch_filename)
            
            if result_code != 0:
                print(f"❌ 对照组 (seed={seed_value}) 执行失败，返回码: {result_code}")
                return False
                
            self.ran_control_seeds.add(seed_value) # 🛠️ 添加 seed
            print(f"✅ 对照组 (seed={seed_value}) 运行完成")
            return True
            
        except Exception as e:
            print(f"❌ 对照组 (seed={seed_value}) 运行异常: {e}")
            return False
    
    # ------------------ 其他辅助方法 ------------------

    def append_result_to_csv(self, result: Dict[str, Any]) -> bool:
        """将单次实验结果追加到CSV文件"""
        try:
            result_df = pd.DataFrame([result])
            
            if os.path.exists(self.results_file):
                result_df.to_csv(self.results_file, mode='a', header=False, index=False)
            else:
                result_df.to_csv(self.results_file, index=False)
            
            return True
        except Exception as e:
            print(f"❌ 写入CSV文件失败: {e}")
            return False
    
    def load_results_from_csv(self) -> pd.DataFrame:
        """从CSV文件加载已有结果"""
        if os.path.exists(self.results_file):
            try:
                df = pd.read_csv(self.results_file)
                if not df.empty:
                    # 确保 k, seed 是整数类型，其他是浮点数
                    for col in ['k', 'seed']:
                        if col in df.columns:
                            # 确保 k 和 seed 可以是整数或 -1/10000 这样的占位符
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-2).astype(int) 
                    for col in ['fee', 'max_slippage', 'spread_mean', 'depth_mean', 'volume_mean', 'p_value_spread', 'p_value_depth', 'p_value_volume']:
                        if col in df.columns:
                            df[col] = df[col].astype(float)
                print(f"📋 从 {self.results_file} 加载了 {len(df)} 条已有结果")
                return df
            except Exception as e:
                print(f"❌ 加载CSV文件失败: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
        
    def check_if_already_run(self, k_value: float, fee_value: float, slippage_value: float, seed_value: int) -> bool:
        """检查参数组合是否已存在于历史结果中 (使用容差比较)"""
        # 仅检查实验组参数
        if self.existing_results.empty:
            return False
            
        k_value_int = int(k_value) # k 统一以整数存储/比较
        fee_value = float(fee_value)
        slippage_value = float(slippage_value)
        
        # 容差阈值（使用绝对容差）
        abs_tol_fee = 0.01 * fee_value
        abs_tol_slippage = 0.01 * slippage_value
        
        mask = (
            # k 应该精确匹配或使用 np.isclose 对 int(k) 进行容差比较
            (self.existing_results['k'] == k_value_int) & 
            np.isclose(self.existing_results['fee'], fee_value, atol=abs_tol_fee) &
            np.isclose(self.existing_results['max_slippage'], slippage_value, atol=abs_tol_slippage) &
            (self.existing_results['seed'] == seed_value)
        )
        
        # 排除对照组记录
        mask &= (self.existing_results['fee'] < 1000.0) # 假设实验组 fee 远小于对照组标记 10000.0
        
        return mask.any()
        
    def run_single_experiment(self, k_value: float, fee_value: float, slippage_value: float, seed_value: int) -> Dict[str, Any] | None:
        """运行单次实验（实验组）并提取t检验结果"""
        # ... (与原代码保持一致，但确保参数类型正确)
        try:
            print(f"🚀 运行: k={k_value:.2e}, fee={fee_value:.4f}, slippage={slippage_value:.3f}, seed={seed_value}")
            
            result_dir = "ttest_results"
            os.makedirs(result_dir, exist_ok=True)
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            cmd2 = lines[1] # rmsc04
            
            # 替换参数
            k_int = int(k_value) # k为整数
            cmd2 = self.replace_parameter(cmd2, '-k', k_int) 
            cmd2 = self.replace_parameter(cmd2, '--fee', fee_value)
            cmd2 = self.replace_parameter(cmd2, '--max-slippage', slippage_value)
            cmd2 = self.replace_parameter(cmd2, '-s', seed_value) # 确保实验组使用正确的 seed
            
            # ttest.py 命令不需要 --output_dir，因为我们手动读取 results.json
            cmd3 = "python ttest.py"
            
            # 创建并运行批处理文件
            batch_content = f"""@echo off
{cmd2}
{cmd3}
"""
            
            # 使用临时文件路径
            batch_filename = os.path.join(result_dir, 'run_experiment.bat') 
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            result_code = os.system(f'"{batch_filename}"')
            
            if result_code != 0:
                print(f"❌ 实验组执行失败，返回码: {result_code}")
                return None
                
            # 提取t检验结果 (假设 ttest.py 将结果输出到项目根目录下的 ttest_results.json)
            result_file = 'ttest_results.json' # 假设 ttest.py 输出在根目录
            potential_file = os.path.join(result_dir, 'ttest_results.json') # 备份检查 ttest_results 文件夹
            
            final_result_file = None
            if os.path.exists(result_file):
                final_result_file = result_file
            elif os.path.exists(potential_file):
                final_result_file = potential_file

            if final_result_file:
                with open(final_result_file, 'r') as f:
                    ttest_result = json.load(f)
                
                result = {
                    'k': k_int, # 记录为整数
                    'fee': float(fee_value),
                    'max_slippage': float(slippage_value),
                    'seed': int(seed_value),
                    **ttest_result
                }
                
                self.append_result_to_csv(result)
                return result
            else:
                print(f"⚠️ 未找到结果文件: {result_file} 或 {potential_file}")
                return None
                
        except Exception as e:
            print(f"❌ 实验组运行异常: {e}")
            return None

    # 🛠️ run_discrete_grid_search 逻辑大改
    def run_discrete_grid_search(self, param_values: Dict[str, List[float]]) -> pd.DataFrame:
        """执行离散参数网格搜索，并按 seed 分组运行对照组"""
        self.results = []
        
        # 提取实验参数 (k, fee, slippage)
        k_values = param_values['k']
        experiment_combinations = list(product(
            k_values,
            param_values['fee'], 
            param_values['max_slippage']
        ))
        
        total_combinations = len(experiment_combinations) * len(self.discrete_seeds)
        print(f"开始离散网格搜索实验: 总计 {total_combinations} 种组合...")
        
        new_experiments_count = 0
        experiment_index = 0

        # 1. 只对 seed 循环
        for seed in self.discrete_seeds:
            print(f"\n========================================================")
            print(f"🌐 开始处理 Seed={seed} 的所有参数组合 (共 {len(experiment_combinations)} 种)")
            print(f"========================================================")

            # 2. 运行/检查对照组 (rmsc03)
            # 注意：对照组不依赖 k，只依赖 seed。
            if not self.run_control_group_with_seed(seed): 
                print(f"❌ 对照组 (seed={seed}) 运行失败，跳过该 seed 的所有实验。")
                continue
            
            # 3. 运行实验组 (rmsc04)
            for k_value, fee, slippage in experiment_combinations:
                experiment_index += 1
                k_value = float(k_value)
                
                print(f"\n=== 尝试实验 {experiment_index}/{total_combinations} (k={k_value:.2e}, Fee={fee:.4f}, Slippage={slippage:.3f}, Seed={seed}) ===")
                
                # 检查是否已运行 (同时检查 k, fee, slippage, seed)
                if self.check_if_already_run(k_value, fee, slippage, seed):
                    print(f"👉 组合已存在于历史结果中，跳过")
                    continue
                    
                # 运行实验组，传递 k_value, fee, slippage, seed
                result = self.run_single_experiment(k_value, fee, slippage, seed)
                if result:
                    self.results.append(result)
                    new_experiments_count += 1
                
        print(f"\n✅ 离散网格搜索完成: 成功运行 {new_experiments_count} 个新实验")
        
        # 合并新旧结果 (仅保留最新的、非重复的实验结果)
        if not self.existing_results.empty:
            all_results = pd.concat([self.existing_results, pd.DataFrame(self.results)], ignore_index=True)
        else:
            all_results = pd.DataFrame(self.results)
            
        # 移除重复项，保留最新的（因为新的结果是最后添加的）
        # 筛选出实验组参数列
        exp_cols = ['k', 'fee', 'max_slippage', 'seed']
        # 对照组标记 (fee=10000) 只有 seed 重复，但我们不删除它们，只删除实验组的重复
        
        # 区分实验组和对照组
        exp_group = all_results[all_results['fee'] < 10000.0] 
        control_group = all_results[all_results['fee'] >= 10000.0]
        
        # 删除实验组重复项 (保留最后一次运行的)
        exp_group_unique = exp_group.drop_duplicates(subset=exp_cols, keep='last')
        
        # 合并 (实验组+对照组)
        final_results = pd.concat([exp_group_unique, control_group], ignore_index=True)
        
        print(f"💡 最终结果集大小: {len(final_results)} 条记录")
        return final_results

def main():
    # 原始命令模板，注意 rmsc03 和 rmsc04 中的 -s 和 -k 参数将由脚本动态替换
    base_cmd = """python -u abides.py -c rmsc03 -t ETH -d 20251028 -s 5678 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:40:00 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python -u abides.py -c rmsc04 -t ETH -d 20251028 -s 5678 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 1e12 --fee 0.001 --max-slippage 0.05 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python ttest.py"""
    
    # 离散参数的网格点
    param_values = {
        # 'k': [1e10, 2e10, 3e10, 4e10, 5e10, 6e10, 7e10, 8e10, 9e10],     # 资金池规模 (9个点)
        # 'fee': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],      # 手续费 (9个点)
        'k': [1e12, 2e12, 3e12, 4e12, 5e12, 6e12, 7e12, 8e12, 9e12] + [0.5e12, 1.5e12, 2.5e12, 3.5e12, 4.5e12, 5.5e12, 6.5e12, 7.5e12, 8.5e12, 9.5e12],     # 资金池规模 (19个点)
        'fee': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009] + [0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0095, 0.0100],      # 手续费 (19个点)
        'max_slippage': [0.05], # 最大滑点 (1个点)
    }
    
    # 🛠️ __init__ 不再接收 k_values
    runner = ExperimentRunner(base_cmd)
    
    # 运行离散网格搜索
    results_df = runner.run_discrete_grid_search(param_values)
    
    if results_df.empty:
        print("没有可用的实验结果")
        return
    
    # 将最终结果保存
    results_df.to_csv('final_experiment_results.csv', index=False)
    print("\n🎉 最终结果已保存到 final_experiment_results.csv")


if __name__ == "__main__":
    main()