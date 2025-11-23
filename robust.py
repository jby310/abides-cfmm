import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

# 创建robust文件夹（如果不存在）
os.makedirs('robust', exist_ok=True)

# 设置绘图风格
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# 定义全局变量
TARGET_COLS = ['spread_mean', 'depth_mean', 'volume_mean']
FEATURE_COLS = ['k', 'fee', 'seed', 'max_slippage']

def load_and_preprocess_data(file_path):
    """Load and preprocess data, properly handling categorical variables"""
    df = pd.read_csv(file_path)
    print("Original data shape:", df.shape)
    print("\nFirst 5 rows of data:")
    print(df[FEATURE_COLS + TARGET_COLS].head())
    
    # 检查必要列
    required_cols = FEATURE_COLS + TARGET_COLS
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in data: {col}")
    
    # 复制数据用于处理
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()
    y['spread_mean'] = abs(y['spread_mean'])
    y['volume_mean'] = abs(y['volume_mean'])
    
    # 将seed转换为分类变量（字符串类型）
    X['seed'] = X['seed'].astype(str)
    
    print("\nFeature data basic information:")
    print("Numerical feature statistics:")
    print(X[['k', 'fee', 'max_slippage']].describe())
    print("\nCategorical feature statistics:")
    print(f"seed unique values: {X['seed'].unique()}")
    print(f"seed distribution:\n{X['seed'].value_counts()}")
    
    print("\nTarget variables basic statistics:")
    print(y.describe())
    
    return X, y, df, FEATURE_COLS

def prepare_features_for_modeling(X):
    """Prepare features for modeling, properly handling categorical variables"""
    X_encoded = X.copy()
    
    # 对分类变量进行编码
    le = LabelEncoder()
    X_encoded['seed_encoded'] = le.fit_transform(X_encoded['seed'])
    
    # 选择用于建模的特征（排除原始的seed分类列）
    modeling_features = ['k', 'fee', 'max_slippage', 'seed_encoded']
    
    print(f"\nFeatures used for modeling: {modeling_features}")
    print(f"seed encoding mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return X_encoded[modeling_features], le

def train_single_xgboost(X, y_target, target_name):
    """Train single XGBoost regression model, handling categorical features"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_target, test_size=0.3, random_state=42, shuffle=True
    )
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        silent=True
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'cv_r2': cross_val_score(model, X, y_target, cv=5, scoring='r2').mean()
    }
    
    print(f"\n=== Target {target_name} Model Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return model, metrics, (X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)

def get_feature_importance(model, feature_names):
    """Extract feature importance"""
    try:
        gain_importance = model.get_booster().get_score(importance_type='gain')
    except:
        gain_importance = dict(zip(feature_names, model.feature_importances_))
    
    gain_vals = [gain_importance.get(f, 0) for f in feature_names]
    gain_vals = np.array(gain_vals) / sum(gain_vals) * 100 if sum(gain_vals) > 0 else np.zeros_like(gain_vals)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Gain Percentage(%)': gain_vals,
        'Importance Rank': np.argsort(np.argsort(-gain_vals)) + 1
    }).sort_values(by='Gain Percentage(%)', ascending=False)
    
    return importance_df

def plot_categorical_effects(X_original, y, le, save_path='robust/categorical_effects.png'):
    """Categorical variable effects visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 获取原始分类标签
    seed_labels = le.classes_
    
    for i, target in enumerate(TARGET_COLS):
        ax = axes[i]
        
        # 为每个seed值计算目标变量的统计量
        seed_stats = []
        for seed_label in seed_labels:
            mask = X_original['seed'] == seed_label
            target_values = y[target][mask]
            if len(target_values) > 0:
                seed_stats.append({
                    'seed': seed_label,
                    'mean': target_values.mean(),
                    'std': target_values.std(),
                    'count': len(target_values)
                })
        
        stats_df = pd.DataFrame(seed_stats)
        
        # 绘制箱线图和小提琴图
        data_for_plot = []
        for seed_label in seed_labels:
            mask = X_original['seed'] == seed_label
            target_values = y[target][mask]
            data_for_plot.extend([(seed_label, val) for val in target_values])
        
        plot_df = pd.DataFrame(data_for_plot, columns=['seed', target])
        
        # 箱线图
        sns.boxplot(data=plot_df, x='seed', y=target, ax=ax, palette='Set2')
        sns.stripplot(data=plot_df, x='seed', y=target, ax=ax, color='black', alpha=0.5, size=3)
        
        ax.set_xlabel('Random Seed')
        ax.set_ylabel(target)
        ax.set_title(f'{target} by Random Seed\n')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_partial_dependence_with_categorical(models, X_modeling, le, target_names, save_path='robust/partial_dependence_categorical.png'):
    """Partial dependence plots - properly handling categorical variables"""
    n_numeric_features = 3  # k, fee, max_slippage
    n_targets = len(target_names)
    
    fig, axes = plt.subplots(n_targets, n_numeric_features, figsize=(5*n_numeric_features, 4*n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    numeric_features = ['k', 'fee', 'max_slippage']
    
    for i, target in enumerate(target_names):
        model = models[target]
        for j, feature in enumerate(numeric_features):
            ax = axes[i, j]
            
            # 计算部分依赖（只对数值特征）
            feature_idx = X_modeling.columns.tolist().index(feature)
            pdp_results = partial_dependence(
                model, X_modeling, features=[feature_idx], grid_resolution=50
            )
            feature_vals = pdp_results['grid_values'][0]
            pdp_vals = pdp_results['average'][0]
            
            ax.plot(feature_vals, pdp_vals, color=colors[i], linewidth=2, label=target)
            ax.fill_between(feature_vals, pdp_vals, alpha=0.2, color=colors[i])
            
            ax.set_xlabel(feature)
            if j == 0:
                ax.set_ylabel(f'Predicted {target}')
            ax.set_title(f'{feature} vs {target}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_prediction_vs_actual(models_dict, results_dict, save_path='robust/prediction_vs_actual.png'):
    """绘制预测值与真实值的对比图 - 上下排布，三列两行"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (target, results) in enumerate(results_dict.items()):
        X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = results
        
        # 训练集（第一行）
        ax_train = axes[0, i]
        ax_train.scatter(y_train, y_train_pred, alpha=0.6, color=colors[i], label='Train')
        
        # 完美预测线
        min_val = min(y_train.min(), y_train_pred.min())
        max_val = max(y_train.max(), y_train_pred.max())
        ax_train.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax_train.set_xlabel('Actual Values')
        ax_train.set_ylabel('Predicted Values')
        ax_train.set_title(f'{target} - Training Set\nR² = {r2_score(y_train, y_train_pred):.3f}')
        ax_train.legend()
        ax_train.grid(True, alpha=0.3)
        
        # 测试集（第二行）
        ax_test = axes[1, i]
        ax_test.scatter(y_test, y_test_pred, alpha=0.6, color=colors[i], label='Test')
        
        # 完美预测线
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        ax_test.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax_test.set_xlabel('Actual Values')
        ax_test.set_ylabel('Predicted Values')
        ax_test.set_title(f'{target} - Test Set\nR² = {r2_score(y_test, y_test_pred):.3f}')
        ax_test.legend()
        ax_test.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curves(models_dict, X_modeling, y, save_path='robust/learning_curves.png'):
    """学习曲线 - 使用左右双坐标轴"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (target, model) in enumerate(models_dict.items()):
        ax1 = axes[i]  # 主坐标轴
        
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_modeling, y[target], cv=5, 
            scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)
        
        # 主坐标轴：MSE
        line2 = ax1.plot(train_sizes, test_scores_mean, 'o-', color=colors[i], linestyle='--', label='Validation MSE', linewidth=2)[0]
        
        ax1.set_xlabel('Training examples')
        ax1.set_ylabel('MSE', color=colors[i])
        ax1.tick_params(axis='y', labelcolor=colors[i])
        ax1.set_title(f'{target} - Learning Curve')
        ax1.grid(True, alpha=0.3)
        
        # 创建第二个y轴用于R²
        ax2 = ax1.twinx()
        
        # 计算R²分数
        train_r2 = [1 - (mse / np.var(y[target])) for mse in train_scores_mean]
        test_r2 = [1 - (mse / np.var(y[target])) for mse in test_scores_mean]
        
        line4 = ax2.plot(train_sizes, test_r2, 'o-', color='red', linestyle='--', label='Validation R²', linewidth=2, alpha=0.7)[0]
        
        ax2.set_ylabel('R² Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        
        # 合并图例
        lines = [line2, line4]
        labels = ['Validation MSE', 'Validation R²']
        ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_combined_feature_analysis(importance_dict, save_path='robust/combined_feature_analysis.png'):
    """组合特征分析图 - 左右两个子图排版，确保特征顺序一致"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 选择学术蓝调配色
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # 深蓝、紫红、橙黄

    # 为每个目标创建水平条形图
    targets = list(importance_dict.keys())
    features = importance_dict[targets[0]]['Feature'].tolist()
    
    # 计算每个特征在所有目标中的平均重要性
    avg_importance = []
    for feature in features:
        importance_sum = 0
        for target in targets:
            importance_df = importance_dict[target]
            importance_val = importance_df[importance_df['Feature'] == feature]['Gain Percentage(%)'].values[0]
            importance_sum += importance_val
        avg_importance.append(importance_sum / len(targets))
    
    # 按平均重要性排序（两个图使用相同的排序）
    sorted_indices = np.argsort(avg_importance)
    sorted_features = [features[i] for i in sorted_indices]
    
    # 绘制左子图：特征重要性比较
    y_pos = np.arange(len(sorted_features))
    bar_height = 0.22  # 稍微调整条形高度
    
    for i, target in enumerate(targets):
        importances = []
        for feature in sorted_features[::-1]:
            importance_df = importance_dict[target]
            importance_val = importance_df[importance_df['Feature'] == feature]['Gain Percentage(%)'].values[0]
            importances.append(importance_val)
        
        # 使用学术配色，添加边框增强质感
        bars = ax1.barh(y_pos + i * bar_height, importances, bar_height, 
                       color=colors[i], alpha=0.85, label=target, linewidth=0.8)
        
        # 添加数值标签（可选）
        for j, (bar, importance) in enumerate(zip(bars, importances)):
            if importance > 5:  # 只在重要性大于5%时显示标签
                ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.1f}%', ha='left', va='center', fontsize=9)
    
    ax1.set_yticks(y_pos + bar_height)
    ax1.set_yticklabels(sorted_features[::-1], fontsize=11)
    ax1.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Importance Comparison Across Targets', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.2, axis='x')
    
    # # 设置边框样式
    # for spine in ax1.spines.values():
    #     spine.set_linewidth(1.2)
    #     spine.set_color('#333333')
    
    # 右子图：特征贡献热力图（使用相同的特征排序）
    contribution_matrix = np.zeros((len(sorted_features), len(targets)))
    for j, target in enumerate(targets):
        importance_df = importance_dict[target]
        for i, feature in enumerate(sorted_features):
            contribution = importance_df[importance_df['Feature'] == feature]['Gain Percentage(%)'].values
            if len(contribution) > 0:
                contribution_matrix[i, j] = contribution[0]
    
    # 创建DataFrame用于热力图
    heatmap_df = pd.DataFrame(
        contribution_matrix,
        index=sorted_features,
        columns=targets
    )
    
    # 使用更高级的热力图配色
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',  # 改为蓝色系，更符合学术风格
        cbar_kws={'label': 'Feature Gain Percentage (%)'},
        ax=ax2,
        linewidths=0.8,
        linecolor='white',
        annot_kws={'size': 10, 'weight': 'bold'}
    )
    
    ax2.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Contribution Heatmap', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return heatmap_df
def plot_correlation_heatmap(X_modeling, y, save_path='robust/correlation_heatmap.png'):
    """特征与目标变量的相关性热力图"""
    # 合并特征和目标变量
    data_for_corr = pd.concat([X_modeling, y], axis=1)
    
    # 计算相关系数矩阵
    corr_matrix = data_for_corr.corr()
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    
    plt.title('Feature-Target Correlation Heatmap', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function: properly handling categorical variable seed"""
    try:
        X_original, y, df, FEATURE_COLS = load_and_preprocess_data('experiment_results.csv')
    except FileNotFoundError:
        print("Error: experiment_results.csv file not found, please check the path!")
        return
    
    # 准备建模特征（编码分类变量）
    X_modeling, label_encoder = prepare_features_for_modeling(X_original)
    
    # 初始化存储字典
    models = {}
    metrics_dict = {}
    results_dict = {}
    importance_dict = {}
    
    # 训练每个目标的模型
    feature_names = X_modeling.columns.tolist()
    for target in TARGET_COLS:
        model, metrics, results = train_single_xgboost(X_modeling, y[target], target)
        models[target] = model
        metrics_dict[target] = metrics
        results_dict[target] = results
        
        importance_df = get_feature_importance(model, feature_names)
        importance_dict[target] = importance_df
        print(f"\n{target} Feature Importance:")
        print(importance_df)
    
    # 保存模型到robust文件夹
    for target, model in models.items():
        try:
            model.save_model(f'robust/{target}_xgboost.model')
        except:
            import pickle
            with open(f'robust/{target}_xgboost.pkl', 'wb') as f:
                pickle.dump(model, f)
    print("\nAll models saved to robust/ folder")
    
    # 可视化
    print("\n=== Generating Enhanced Visualizations ===")
    
    # 原有的可视化
    plot_categorical_effects(X_original, y, label_encoder)
    plot_partial_dependence_with_categorical(models, X_modeling, label_encoder, TARGET_COLS)
    
    # 修改后的可视化
    plot_prediction_vs_actual(models, results_dict)  # 上下排布
    plot_learning_curves(models, X_modeling, y)  # 双坐标轴
    plot_combined_feature_analysis(importance_dict)  # 左右子图排版
    
    # 其他可视化
    plot_correlation_heatmap(X_modeling, y)
    
    # 保存评估指标到robust文件夹
    metrics_summary = pd.DataFrame(metrics_dict).T
    print("\n=== Multi-Target Model Evaluation Summary ===")
    print(metrics_summary.round(4))
    metrics_summary.to_csv('robust/multi_target_metrics_summary.csv', index=True)
    print("Evaluation metrics summary saved to robust/ folder")
    
    print("\n=== All Visualizations Completed ===")
    print("Generated the following visualization files in robust/ folder:")
    print("1. categorical_effects.png - 分类变量效应分析")
    print("2. partial_dependence_categorical.png - 部分依赖图")
    print("3. prediction_vs_actual.png - 预测值vs真实值（上下排布）")
    print("4. learning_curves.png - 学习曲线（双坐标轴）")
    print("5. combined_feature_analysis.png - 组合特征分析（左右子图）")
    print("6. correlation_heatmap.png - 相关性热力图")

if __name__ == "__main__":
    main()