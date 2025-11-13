import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import tempfile
import json
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

IS_CONTROL_GROUP_RUN = True

class ExperimentRunner:
    def __init__(self, base_cmd):
        self.base_cmd = base_cmd
        self.results = []
        self.control_group_run = IS_CONTROL_GROUP_RUN  # 标记对照组是否已运行
        
    def run_control_group(self):
        """运行对照组（只需运行一次）"""
        if self.control_group_run:
            print("对照组已运行，跳过...")
            return True
            
        try:
            print("运行对照组...")
            
            # 提取对照组的命令（第一条命令）
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            control_cmd = lines[0]
            
            # 创建批处理文件内容
            batch_content = f"""@echo off
{control_cmd}
"""
            
            # 写入批处理文件
            with open('run_control.bat', 'w') as f:
                f.write(batch_content)
            
            # 运行批处理文件
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
    
    def run_single_experiment(self, k_value, fee_value):
        """运行单次实验（实验组）并提取t检验结果"""
        try:
            print(f"运行实验组: k={k_value:.0e}, fee={fee_value:.3f}")
            
            # 创建结果目录
            result_dir = "ttest_results"
            os.makedirs(result_dir, exist_ok=True)
            
            # 提取实验组的命令（第二、三条命令）
            lines = [line.strip() for line in self.base_cmd.split('\n') if line.strip()]
            cmd2 = lines[1].replace('-k 100000000', f'-k {int(k_value)}').replace('--fee 0.003', f'--fee {fee_value}')
            
            # 修改ttest命令，添加结果目录参数
            cmd3 = f"python ttest.py --output_dir {result_dir}"
            
            # 创建批处理文件内容
            batch_content = f"""@echo off
{cmd2}
{cmd3}
"""
            
            # 写入批处理文件
            batch_filename = os.path.join(result_dir, 'run_experiment.bat')
            with open(batch_filename, 'w') as f:
                f.write(batch_content)
            
            # 运行批处理文件
            result_code = os.system(f'"{batch_filename}"')
            
            if result_code != 0:
                print(f"实验组执行失败: k={k_value}, fee={fee_value}, 返回码: {result_code}")
                return None
                
            # 从结果目录读取结果
            result_file = os.path.join(result_dir, 'ttest_results.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    ttest_result = json.load(f)
                
                result = {
                    'k': k_value,
                    'fee': fee_value,
                    **ttest_result
                }
                print(f"成功提取结果: ΔSpread={result['spread_mean']:.4f}, ΔDepth={result['depth_mean']:.2f}, ΔVolume={result['volume_mean']:.2f}")
                return result
            else:
                print(f"未找到结果文件: {result_file}")
                return None
                
        except Exception as e:
            print(f"实验组运行异常: k={k_value}, fee={fee_value}: {e}")
            return None
    
    def run_parameter_sweep(self, k_values, fee_values):
        """运行参数扫描"""
        self.results = []
        
        # 先运行对照组（只运行一次）
        if not self.run_control_group():
            print("对照组运行失败，终止实验")
            return pd.DataFrame()
        
        # 然后运行实验组（多次，根据参数）
        total_experiments = len(k_values) * len(fee_values)
        current_experiment = 0
        
        for k in k_values:
            for fee in fee_values:
                current_experiment += 1
                print(f"\n=== 实验 {current_experiment}/{total_experiments} ===")
                result = self.run_single_experiment(k, fee)
                if result:
                    self.results.append(result)
                print(f"完成实验组: k={k:.0e}, fee={fee:.3f}")
        
        return pd.DataFrame(self.results) if self.results else pd.DataFrame()
    
    def analyze_coexistence_effect(self, results_df):
        """分析共生/挤出效应"""
        if results_df.empty:
            print("没有结果数据可供分析")
            return
            
        print("\n=== 共生/挤出效应分析 ===")
        
        for idx, row in results_df.iterrows():
            k, fee = row['k'], row['fee']
            
            spread_mean = row['spread_mean']
            depth_mean = row['depth_mean'] 
            volume_mean = row['volume_mean']
            
            spread_p = row['spread_p']
            depth_p = row['depth_p']
            volume_p = row['volume_p']
            
            # 判断效应类型
            if (spread_p > 0.05 or abs(spread_mean) <= 0.002) and depth_p < 0.05 and volume_p < 0.05 and depth_mean > 0 and volume_mean > 0:
                effect_type = "共生效应"
            elif spread_p < 0.05 and abs(spread_mean) > 0.005 and (depth_p > 0.05 or depth_mean <= 0 or volume_p > 0.05 or volume_mean <= 0):
                effect_type = "挤出效应"
            else:
                effect_type = "混合效应"
            
            print(f"k={k:.0e}, fee={fee:.3f}: {effect_type}")
            print(f"  价差: Δ={spread_mean:.4f}, p={spread_p:.3g}")
            print(f"  深度: Δ={depth_mean:.2f}, p={depth_p:.3g}")
            print(f"  成交量: Δ={volume_mean:.2f}, p={volume_p:.3g}")
            print()
    
    def create_coexistence_heatmap(self, results_df, output_dir="heatmaps"):
        """创建共生/挤出效应热力图"""
        if results_df.empty:
            print("没有结果数据可供创建热力图")
            return
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 计算共生效应得分
        # 正得分表示共生效应，负得分表示挤出效应
        results_df['coexistence_score'] = (
            # 深度和成交量的改善（正值表示改善）
            np.sign(results_df['depth_mean']) * np.log1p(np.abs(results_df['depth_mean'])) +
            np.sign(results_df['volume_mean']) * np.log1p(np.abs(results_df['volume_mean'])) -
            # 价差的变化（负值表示恶化）
            np.sign(results_df['spread_mean']) * np.log1p(np.abs(results_df['spread_mean'] * 1000))
        )
        
        # 创建热力图
        pivot_table = results_df.pivot_table(
            values='coexistence_score', 
            index='k', 
            columns='fee', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                   cbar_kws={'label': '共生效应得分 (正=共生, 负=挤出)'})
        plt.title('共生/挤出效应热力图\n(绿色=共生效应, 红色=挤出效应)')
        plt.xlabel('手续费 (fee)')
        plt.ylabel('资金池规模 (k)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coexistence_effect.png'), dpi=300)
        plt.close()
        
        print(f"保存热力图到 {output_dir}")
        
        # 创建详细指标热力图
        self.create_detailed_heatmaps(results_df, output_dir)
    
    def create_detailed_heatmaps(self, results_df, output_dir):
        """创建详细指标热力图"""
        metrics = [
            ('spread_mean', '价差变化 (ΔSpread)', 'RdYlBu_r'),
            ('depth_mean', '深度变化 (ΔDepth)', 'viridis'),
            ('volume_mean', '成交量变化 (ΔVolume)', 'plasma')
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (metric, title, cmap) in enumerate(metrics):
            pivot_table = results_df.pivot_table(
                values=metric, 
                index='k', 
                columns='fee', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap=cmap,
                       ax=axes[idx], cbar_kws={'label': title})
            axes[idx].set_title(title)
            axes[idx].set_xlabel('手续费 (fee)')
            axes[idx].set_ylabel('资金池规模 (k)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detailed_metrics_heatmaps.png'), dpi=300)
        plt.close()
        
        print(f"保存详细指标热力图")

def main():
    # 基础命令模板
    base_cmd = """python -u abides.py -c rmsc03 -t ETH -d 20251028 -s 1235 -l rmsc03_two_hour --end-time 09:40:00
python -u abides.py -c rmsc04 -t ETH -d 20251028 -s 1235 -l rmsc04_two_hour --end-time 09:40:00 -k 100000000 --fee 0.003
python ttest.py"""
    
    # 定义参数范围（先使用小范围测试）
    k_values = [1e7, 1e8]  # 池规模参数
    fee_values = [0.003, 0.008]  # 手续费参数
    
    # 创建实验运行器
    runner = ExperimentRunner(base_cmd)
    
    # 运行参数扫描
    print("开始参数扫描实验...")
    results_df = runner.run_parameter_sweep(k_values, fee_values)
    
    if results_df.empty:
        print("未收集到任何结果")
        return
    
    # 保存结果
    results_df.to_csv('experiment_results.csv', index=False)
    print("结果已保存到 experiment_results.csv")
    
    # 分析效应
    runner.analyze_coexistence_effect(results_df)
    
    # 创建热力图
    runner.create_coexistence_heatmap(results_df)
    
    print("实验完成")

if __name__ == "__main__":
    main()