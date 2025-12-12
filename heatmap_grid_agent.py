import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from itertools import product  # 用于生成参数的笛卡尔积（网格）
from typing import Dict, Any, List, Tuple

# 设置中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentRunner:
    # 🛠️ 新增 num_hybrid_agents 相关处理
    def __init__(self, base_cmd: str):
        self.base_cmd = base_cmd
        self.results = []
        self.results_file = 'experiment_results.csv'
        self.existing_results = self.load_results_from_csv()
        
        # 将 seed 视为一个要遍历的离散参数
        self.discrete_seeds = [12315]
        
        # 🛠️ 只关心 seed 生成需要运行对照组的 seed 列表
        self.control_seeds_to_run = self.discrete_seeds
        
        # 🛠️ 记录已运行的对照组 seed + num_hybrid_agents
        if not self.existing_results.empty:
            # 假设对照组记录的 fee=10000 作为标记
            control_results = self.existing_results[
                self.existing_results['fee'] == 10000.0
            ]
            # 记录已运行的 (seed, num_hybrid_agents) 组合
            self.ran_control_combinations = set(
                zip(control_results['seed'].astype(int), 
                    control_results['num_hybrid_agents'].astype(int))
            )
        else:
            self.ran_control_combinations = set()
            
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
        
    # 🛠️ 新增 num_hybrid_agents 参数，对照组也要替换该参数
    def run_control_group_with_seed(self, seed_value: int, num_hybrid_agents: int) -> bool:
        """运行对照组 (rmsc03) 并替换 -s 和 --num-hybrid-agents 参数"""
        
        # 🛠️ 检查 (seed, num_hybrid_agents) 组合 是否已运行/记录
        combo_key = (seed_value, num_hybrid_agents)
        if combo_key in self.ran_control_combinations:
            print(f"✅ 对照组 (seed={seed_value}, agents={num_hybrid_agents}) 已运行或已记录，跳过...")
            return True 
            
        try:
            print(f"🚀 运行对照组 (rmsc03) with seed={seed_value}, agents={num_hybrid_agents}...")
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            control_cmd_base = lines[0]  # rmsc03

            # 🛠️ 替换对照组命令中的 -s 和 --num-hybrid-agents 参数
            control_cmd = self.replace_parameter(control_cmd_base, '-s', seed_value)
            control_cmd = self.replace_parameter(control_cmd_base, '--num-hybrid-agents', num_hybrid_agents)
            
            batch_content = f"""@echo off
{control_cmd}
"""
            
            # 使用特定 seed 和 agents 的文件名
            batch_filename = f'run_control_s{seed_value}_agents{num_hybrid_agents}.bat'
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            result_code = os.system(batch_filename)
            
            if result_code != 0:
                print(f"❌ 对照组 (seed={seed_value}, agents={num_hybrid_agents}) 执行失败，返回码: {result_code}")
                return False
                
            self.ran_control_combinations.add(combo_key)  # 🛠️ 添加组合键
            # 记录对照组结果到CSV
            control_result = {
                'k': -1,  # 对照组无k值，用-1标记
                'fee': 10000.0,  # 对照组标记
                'max_slippage': -1.0,  # 对照组无滑点，用-1标记
                'seed': seed_value,
                'num_hybrid_agents': num_hybrid_agents,
                'spread_mean': np.nan,
                'depth_mean': np.nan,
                'volume_mean': np.nan,
                'p_value_spread': np.nan,
                'p_value_depth': np.nan,
                'p_value_volume': np.nan
            }
            self.append_result_to_csv(control_result)
            
            print(f"✅ 对照组 (seed={seed_value}, agents={num_hybrid_agents}) 运行完成")
            return True
            
        except Exception as e:
            print(f"❌ 对照组 (seed={seed_value}, agents={num_hybrid_agents}) 运行异常: {e}")
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
                    # 确保字段类型正确
                    int_cols = ['k', 'seed', 'num_hybrid_agents']
                    for col in int_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-2).astype(int)
                    float_cols = ['fee', 'max_slippage', 'spread_mean', 'depth_mean', 'volume_mean', 
                                  'p_value_spread', 'p_value_depth', 'p_value_volume']
                    for col in float_cols:
                        if col in df.columns:
                            df[col] = df[col].astype(float)
                print(f"📋 从 {self.results_file} 加载了 {len(df)} 条已有结果")
                return df
            except Exception as e:
                print(f"❌ 加载CSV文件失败: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
        
    def check_if_already_run(self, k_value: float, fee_value: float, slippage_value: float, 
                             seed_value: int, num_hybrid_agents: int) -> bool:
        """检查参数组合是否已存在于历史结果中"""
        if self.existing_results.empty:
            return False
            
        k_value_int = int(k_value)
        fee_value = float(fee_value)
        slippage_value = float(slippage_value)
        
        # 容差阈值
        abs_tol_fee = 0.01 * fee_value
        abs_tol_slippage = 0.01 * slippage_value
        
        mask = (
            (self.existing_results['k'] == k_value_int) & 
            np.isclose(self.existing_results['fee'], fee_value, atol=abs_tol_fee) &
            np.isclose(self.existing_results['max_slippage'], slippage_value, atol=abs_tol_slippage) &
            (self.existing_results['seed'] == seed_value) &
            (self.existing_results['num_hybrid_agents'] == num_hybrid_agents)
        )
        
        # 排除对照组记录
        mask &= (self.existing_results['fee'] < 1000.0)
        
        return mask.any()
        
    def run_single_experiment(self, k_value: float, fee_value: float, slippage_value: float, 
                              seed_value: int, num_hybrid_agents: int) -> Dict[str, Any] | None:
        """运行单次实验（实验组）并提取t检验结果"""
        try:
            print(f"🚀 运行: k={k_value:.2e}, fee={fee_value:.4f}, slippage={slippage_value:.3f}, "
                  f"seed={seed_value}, agents={num_hybrid_agents}")
            
            result_dir = "ttest_results"
            os.makedirs(result_dir, exist_ok=True)
            
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            cmd2 = lines[1]  # rmsc04
            
            # 替换所有参数
            k_int = int(k_value)
            cmd2 = self.replace_parameter(cmd2, '-k', k_int) 
            cmd2 = self.replace_parameter(cmd2, '--fee', fee_value)
            cmd2 = self.replace_parameter(cmd2, '--max-slippage', slippage_value)
            cmd2 = self.replace_parameter(cmd2, '-s', seed_value)
            cmd2 = self.replace_parameter(cmd2, '--num-hybrid-agents', num_hybrid_agents)
            
            # ttest.py 命令
            cmd3 = "python ttest.py"
            
            # 创建批处理文件
            batch_content = f"""@echo off
{cmd2}
{cmd3}
"""
            
            batch_filename = os.path.join(result_dir, f'run_exp_agents{num_hybrid_agents}_s{seed_value}.bat')
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            result_code = os.system(f'"{batch_filename}"')
            
            if result_code != 0:
                print(f"❌ 实验组执行失败，返回码: {result_code}")
                return None
                
            # 读取t检验结果
            result_file = 'ttest_results.json'
            potential_file = os.path.join(result_dir, 'ttest_results.json')
            
            final_result_file = None
            if os.path.exists(result_file):
                final_result_file = result_file
            elif os.path.exists(potential_file):
                final_result_file = potential_file

            if final_result_file:
                with open(final_result_file, 'r') as f:
                    ttest_result = json.load(f)
                
                result = {
                    'k': k_int,
                    'fee': float(fee_value),
                    'max_slippage': float(slippage_value),
                    'seed': int(seed_value),
                    'num_hybrid_agents': int(num_hybrid_agents),
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

    def run_discrete_grid_search(self, param_values: Dict[str, List[Any]]) -> pd.DataFrame:
        """执行离散参数网格搜索，核心遍历 num_hybrid_agents"""
        self.results = []
        
        # 提取实验参数
        num_hybrid_agents_list = param_values['num_hybrid_agents']
        k_values = param_values['k']
        fee_values = param_values['fee']
        slippage_values = param_values['max_slippage']
        
        # 生成参数组合 (固定k/fee/slippage，主要遍历agents)
        experiment_combinations = list(product(
            k_values,
            fee_values,
            slippage_values,
            num_hybrid_agents_list
        ))
        
        total_combinations = len(experiment_combinations) * len(self.discrete_seeds)
        print(f"开始离散网格搜索实验: 总计 {total_combinations} 种组合...")
        
        new_experiments_count = 0
        experiment_index = 0

        # 1. 遍历seed
        for seed in self.discrete_seeds:
            print(f"\n========================================================")
            print(f"🌐 开始处理 Seed={seed} 的所有参数组合 (共 {len(experiment_combinations)} 种)")
            print(f"========================================================")

            # 2. 遍历所有参数组合
            for k_value, fee, slippage, num_agents in experiment_combinations:
                experiment_index += 1
                k_value = float(k_value)
                
                print(f"\n=== 尝试实验 {experiment_index}/{total_combinations} ===")
                print(f"参数: k={k_value:.2e}, fee={fee:.4f}, slippage={slippage:.3f}, agents={num_agents}, seed={seed}")
                
                # 先运行/检查对照组
                if not self.run_control_group_with_seed(seed, num_agents):
                    print(f"❌ 对照组运行失败，跳过该实验组")
                    continue
                
                # 检查实验组是否已运行
                if self.check_if_already_run(k_value, fee, slippage, seed, num_agents):
                    print(f"👉 实验组组合已存在于历史结果中，跳过")
                    continue
                    
                # 运行实验组
                result = self.run_single_experiment(k_value, fee, slippage, seed, num_agents)
                if result:
                    self.results.append(result)
                    new_experiments_count += 1
                
        print(f"\n✅ 离散网格搜索完成: 成功运行 {new_experiments_count} 个新实验")
        
        # 合并新旧结果
        if not self.existing_results.empty:
            all_results = pd.concat([self.existing_results, pd.DataFrame(self.results)], ignore_index=True)
        else:
            all_results = pd.DataFrame(self.results)
            
        # 去重处理
        exp_cols = ['k', 'fee', 'max_slippage', 'seed', 'num_hybrid_agents']
        exp_group = all_results[all_results['fee'] < 10000.0]
        control_group = all_results[all_results['fee'] >= 10000.0]
        
        # 删除实验组重复项
        exp_group_unique = exp_group.drop_duplicates(subset=exp_cols, keep='last')
        
        # 合并结果
        final_results = pd.concat([exp_group_unique, control_group], ignore_index=True)
        
        print(f"💡 最终结果集大小: {len(final_results)} 条记录")
        return final_results

def main():
    # 原始命令模板
    base_cmd = """python -u abides.py -c rmsc03 -t ETH -d 20251110 -s 1234 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:35:00 --num-hybrid-agents 100 --fundamental-file-path data/ETH1.xlsx --r-bar 3611.0
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 1234 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:35:00 -k 10000000000 --fee 0.003 --max-slippage 0.05 --num-hybrid-agents 100 --fundamental-file-path data/ETH1.xlsx --r-bar 3611.0
python ttest.py"""
    
    # 🛠️ 修改参数配置：核心遍历 num_hybrid_agents (100-200，间隔5)，固定其他参数
    param_values = {
        'k': [5e10],  # 固定k值
        'fee': [0.005],  # 固定手续费
        'max_slippage': [0.05],  # 固定滑点
        'num_hybrid_agents': list(range(100, 201, 5))  # 100到200，间隔5
    }
    
    # 打印参数配置确认
    print("📌 实验参数配置:")
    print(f"   num_hybrid_agents: {param_values['num_hybrid_agents']} (共{len(param_values['num_hybrid_agents'])}个值)")
    print(f"   k: {param_values['k']}")
    print(f"   fee: {param_values['fee']}")
    print(f"   max_slippage: {param_values['max_slippage']}")
    print(f"   seed: [12315]")
    
    # 初始化运行器
    runner = ExperimentRunner(base_cmd)
    
    # 运行离散网格搜索
    results_df = runner.run_discrete_grid_search(param_values)
    
    if results_df.empty:
        print("没有可用的实验结果")
        return
    
    # 保存最终结果
    results_df.to_csv('final_experiment_results.csv', index=False)
    print("\n🎉 最终结果已保存到 final_experiment_results.csv")
    
    # 打印结果概览
    print("\n📊 实验结果概览:")
    exp_results = results_df[results_df['fee'] < 10000.0]
    ctrl_results = results_df[results_df['fee'] >= 10000.0]
    print(f"   实验组记录数: {len(exp_results)}")
    print(f"   对照组记录数: {len(ctrl_results)}")
    if not exp_results.empty:
        print(f"   覆盖的agents数量: {sorted(exp_results['num_hybrid_agents'].unique())}")


if __name__ == "__main__":
    main()