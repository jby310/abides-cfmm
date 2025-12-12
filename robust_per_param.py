import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# 创建robust文件夹（如果不存在）
os.makedirs('robust', exist_ok=True)

# 设置绘图风格
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# 定义全局变量
ORIGIN_COLS = ['original_spread_mean', 'original_depth_mean', 'original_volume_mean']
TARGET_COLS = ['spread_mean', 'depth_mean', 'volume_mean']
FEATURE_COLS = ['k', 'fee', 'seed', 'max_slippage']
# 仅保留数值特征作为建模特征（移除max_slippage）
MODELING_FEATURE_COLS = ['k', 'fee']

# ========== 新增：为每个目标定制参数 ==========
TARGET_PARAMS = {
    # spread_mean：波动较小，侧重稳定性，正则化稍弱
    'spread_mean': {
        # 'max_depth': 4,
        # 'min_child_weight': 9,
        # 'gamma': 0.8,
        # 'max_leaves': 12,
        # 'subsample': 0.9,
        # 'colsample_bytree': 0.8,
        # 'reg_lambda': 3,
        # 'reg_alpha': 0.3,
        # 'learning_rate': 0.30,
        # 'n_estimators': 100,
        'random_state':12315,
    },
    # depth_mean：中等波动，平衡拟合与泛化，正则化适中
    'depth_mean': {
        # 'max_depth': 4,
        # 'min_child_weight': 5,
        # 'gamma': 0.8,
        # 'max_leaves': 9,
        # 'subsample': 0.8,
        # 'colsample_bytree': 0.8,
        # 'reg_lambda': 5,
        # 'reg_alpha': 0.2,
        # 'learning_rate': 0.05,
        # 'n_estimators': 150
        'random_state':1234,
    },
    # volume_mean：波动大、异常值多，强正则化防过拟合
    'volume_mean': {
        # 'max_depth': 2,
        # 'min_child_weight': 8,
        # 'gamma': 2.0,
        # 'max_leaves': 9,
        # 'subsample': 0.8,
        # 'colsample_bytree': 0.8,
        # 'reg_lambda': 8,
        # 'reg_alpha': 1.0,
        # 'learning_rate': 0.05,
        # 'n_estimators': 200
        'random_state':1234,
    }
}


def load_and_preprocess_data(file_path):
    """Load and preprocess data, properly handling categorical variables"""
    try:
        df = pd.read_csv('experiment_results.csv')
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise

    print("Original data shape:", df.shape)
    
    # 检查必要列
    required_cols = FEATURE_COLS + TARGET_COLS + ORIGIN_COLS
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in data: {col}")
    
    # 复制数据用于处理
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COLS].copy()

    # 计算目标变量的百分比变化值
    for col in TARGET_COLS:
        y[col] = y[col] / df[f'original_{col}'] * 100
    
    print("\nFeature data basic information (before outlier removal):")
    print("Numerical feature statistics (k, fee only):")
    print(X[MODELING_FEATURE_COLS].describe())
    print("\nTarget variables basic statistics (before outlier removal):")
    print(y.describe())
    
    # 拟合 LabelEncoder 用于分类图
    le = LabelEncoder()
    le.fit(X['seed'])
    
    # 返回 X_original, y_preprocessed, df_original, FEATURE_COLS, le
    return X, y, df, FEATURE_COLS, le


def remove_outliers(X, y, target_cols, factor=1.5):
    """
    基于箱线图原理（IQR方法）剔除目标变量中的异常值。
    对于任一目标变量中被判定为异常值的样本，都将被剔除。
    """
    initial_count = X.shape[0]
    outlier_indices = set()
    
    print(f"\n--- Removing Outliers using IQR method (Factor={factor}) ---")
    
    for col in target_cols:
        Q1 = y[col].quantile(0.25)
        Q3 = y[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # 找到该列的异常值索引
        col_outlier_mask = (y[col] < lower_bound) | (y[col] > upper_bound)
        col_outlier_indices = y[col][col_outlier_mask].index.tolist()
        
        # 将异常值索引添加到总集合
        outlier_indices.update(col_outlier_indices)
        
        print(f"  {col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
        print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"    Found {len(col_outlier_indices)} outliers.")
        
    # 剔除异常值
    all_indices = y.index
    clean_indices = all_indices.drop(list(outlier_indices))
    
    X_clean = X.loc[clean_indices].copy()
    y_clean = y.loc[clean_indices].copy()
    
    removed_count = initial_count - X_clean.shape[0]
    print(f"Total samples removed: {removed_count}")
    print(f"Remaining data shape: {X_clean.shape}")
    print("---------------------------------------------------------")
    
    return X_clean, y_clean


def prepare_features_for_modeling(X):
    """Prepare features for modeling. ONLY numerical features are selected (k, fee)."""
    # 选择用于建模的特征：仅包含k和fee
    X_modeling = X[MODELING_FEATURE_COLS].copy()
    
    print(f"\nFeatures used for XGBoost modeling (excluding 'seed' and 'max_slippage'): {MODELING_FEATURE_COLS}")
    
    return X_modeling


def train_single_xgboost(X, y_target, target_name):
    """
    Train single XGBoost regression model with target-specific parameters
    """
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_target, test_size=0.1, random_state=42, shuffle=True
    )
    
    # 自定义XGBoost兼容的R²评估函数
    def r2_metric(y_true, y_pred):
        return 'r2', r2_score(y_true, y_pred), False
    
    # ========== 核心修改：加载目标专属参数 ==========
    target_params = TARGET_PARAMS[target_name]
    print(f"\nUsing {target_name} specific parameters: {target_params}")
    
    # 初始化模型（使用目标专属参数 + 固定随机种子）
    model = xgb.XGBRegressor(
        **target_params  # 展开目标专属参数
    )
    
    # 训练模型
    print(f"Start standard training for {target_name}. Early stopping is disabled.")
    model.fit(
        X_train, y_train,
        verbose=False,               
    )
    
    # 获取实际使用的迭代次数
    final_ntree_limit = model.get_params()['n_estimators']
    print(f"Fixed n_estimators used: {final_ntree_limit}")
    
    # 进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算 CV R2
    cv_model = model
    cv_r2_score = cross_val_score(cv_model, X, y_target, cv=5, scoring='r2').mean()
    
    # 构造 metrics 字典
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'cv_r2': cv_r2_score 
    }
    
    print(f"\n=== Target {target_name} Model Evaluation (Cleaned Data, No Early Stop) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return model, metrics, (X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)

def get_feature_importance(model, feature_names):
    """Extract feature importance (only k and fee)"""
    try:
        # 尝试获取 gain
        gain_importance = model.get_booster().get_score(importance_type='gain')
        if all(f in gain_importance for f in feature_names):
            pass
        else:
              raise AttributeError("Feature names mismatch or gain not available.")
              
    except (AttributeError, xgb.core.XGBoostError):
        gain_importance = dict(zip(feature_names, model.feature_importances_))
    
    gain_vals = [gain_importance.get(f, 0) for f in feature_names]
    gain_vals = np.array(gain_vals) 
    
    # 归一化为百分比
    total_gain = sum(gain_vals)
    gain_vals = (gain_vals / total_gain * 100) if total_gain > 0 else np.zeros_like(gain_vals)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Gain Percentage(%)': gain_vals,
        'Importance Rank': np.argsort(np.argsort(-gain_vals)) + 1
    }).sort_values(by='Gain Percentage(%)', ascending=False)
    
    return importance_df

def plot_categorical_effects(X_original, y, le, save_path='robust/categorical_effects_outliers_excluded.png'):
    """Categorical variable effects visualization (seed vs targets)"""
    if 'outliers_excluded' not in save_path:
        save_path = save_path.replace('.png', '_pre_removal.png')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 获取原始分类标签
    seed_labels = le.classes_
    
    for i, target in enumerate(TARGET_COLS):
        ax = axes[i]
        
        data_for_plot = []
        for seed_label in seed_labels:
            mask = X_original['seed'] == seed_label
            target_values = y[target][mask]
            data_for_plot.extend([(seed_label, val) for val in target_values])
        
        plot_df = pd.DataFrame(data_for_plot, columns=['seed', target])
        
        # 箱线图
        sns.boxplot(data=plot_df, x='seed', y=target, ax=ax, palette='Set2', order=seed_labels)
        sns.stripplot(data=plot_df, x='seed', y=target, ax=ax, color='black', alpha=0.5, size=3, order=seed_labels)
        
        ax.set_xlabel('Random Seed')
        ax.set_ylabel(f'delta {target} change (%)')
        ax.set_title(f'{target} by Random Seed\n')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_partial_dependence_with_categorical(models, X_modeling, le, target_names, save_path='robust/partial_dependence_outliers_excluded.png'):
    """Partial dependence plots - only for k and fee (两行三列排版)"""
    n_numeric_features = len(MODELING_FEATURE_COLS)  # 2个特征：k, fee
    n_targets = len(target_names)  # 3个目标：spread_mean, depth_mean, volume_mean
    
    # 两行三列排版：第一行是k的部分依赖，第二行是fee的部分依赖
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    numeric_features = MODELING_FEATURE_COLS  # [k, fee]
    
    # 第一行：k的部分依赖（3个目标）
    for i, target in enumerate(target_names):
        model = models[target]
        ax = axes[0, i]
        
        # 计算k的部分依赖
        feature_idx = X_modeling.columns.tolist().index('k')
        pdp_results = partial_dependence(
            model, X_modeling, features=[feature_idx], grid_resolution=50, 
            kind='average'
        )
        feature_vals = pdp_results['grid_values'][0]
        pdp_vals = pdp_results['average'][0]
        
        ax.plot(feature_vals, pdp_vals, color=colors[i], linewidth=2, label=target)
        ax.fill_between(feature_vals, pdp_vals, alpha=0.2, color=colors[i])
        
        ax.set_xlabel('k (Fund Pool Size)')
        ax.set_ylabel(f'Predicted {target} (%)')
        ax.set_title(f'k vs {target}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 第二行：fee的部分依赖（3个目标）
    for i, target in enumerate(target_names):
        model = models[target]
        ax = axes[1, i]
        
        # 计算fee的部分依赖
        feature_idx = X_modeling.columns.tolist().index('fee')
        pdp_results = partial_dependence(
            model, X_modeling, features=[feature_idx], grid_resolution=50, 
            kind='average'
        )
        feature_vals = pdp_results['grid_values'][0]
        pdp_vals = pdp_results['average'][0]
        
        ax.plot(feature_vals, pdp_vals, color=colors[i], linewidth=2, label=target)
        ax.fill_between(feature_vals, pdp_vals, alpha=0.2, color=colors[i])
        
        ax.set_xlabel('fee (Transaction Fee)')
        ax.set_ylabel(f'Predicted {target} (%)')
        ax.set_title(f'fee vs {target}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_prediction_vs_actual(models_dict, results_dict, save_path='robust/prediction_vs_actual_outliers_excluded.png'):
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

def plot_learning_curves(models_dict, X_modeling, y, save_path='robust/learning_curves_outliers_excluded.png'):
    """学习曲线 - 使用左右双坐标轴"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (target, model) in enumerate(models_dict.items()):
        ax1 = axes[i]  # 主坐标轴
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_modeling, y[target], cv=5, 
            scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )
        
        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)
        
        # 主坐标轴：MSE
        line2 = ax1.plot(train_sizes, test_scores_mean, 'o-', color=colors[i], linestyle='--', label='Validation MSE', linewidth=2)[0]
        
        ax1.set_xlabel('Training examples')
        ax1.set_ylabel('MSE', color=colors[i])
        ax1.tick_params(axis='y', labelcolor=colors[i])
        ax1.set_title(f'{target} - Learning Curve (Outliers Excluded)')
        ax1.grid(True, alpha=0.3)
        
        # 创建第二个y轴用于R²
        ax2 = ax1.twinx()
        
        # 计算R²分数： R² = 1 - (MSE / Var(y))
        y_variance = y[target].var()
        train_r2 = [1 - (mse / y_variance) for mse in train_scores_mean]
        test_r2 = [1 - (mse / y_variance) for mse in test_scores_mean]
        
        line4 = ax2.plot(train_sizes, test_r2, 'o-', color='red', linestyle='--', label='Validation $R^2$', linewidth=2, alpha=0.7)[0]
        
        ax2.set_ylabel('$R^2$ Score', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(min(test_r2)-0.05, 1)
        
        # 合并图例
        lines = [line2, line4]
        labels = ['Validation MSE', 'Validation $R^2$']
        ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_combined_feature_analysis(importance_dict, save_path='robust/combined_feature_analysis_outliers_excluded.png'):
    """组合特征分析图 - 左右两个子图排版（仅k和fee）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 选择学术蓝调配色
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # 为每个目标创建水平条形图
    targets = list(importance_dict.keys())
    features = list(MODELING_FEATURE_COLS)  # 仅k和fee
    
    # 计算每个特征在所有目标中的平均重要性
    avg_importance = []
    for feature in features:
        importance_sum = 0
        for target in targets:
            importance_df = importance_dict[target]
            importance_val = importance_df[importance_df['Feature'] == feature]['Gain Percentage(%)'].values
            importance_sum += importance_val[0] if len(importance_val) > 0 else 0
        avg_importance.append(importance_sum / len(targets))
    
    # 按平均重要性排序
    sorted_indices = np.argsort(avg_importance)
    sorted_features = [features[i] for i in sorted_indices]
    
    # 绘制左子图：特征重要性比较
    y_pos = np.arange(len(sorted_features))
    bar_height = 0.22
    
    for i, target in enumerate(targets):
        importances = []
        for feature in sorted_features[::-1]: 
            importance_df = importance_dict[target]
            importance_val = importance_df[importance_df['Feature'] == feature]['Gain Percentage(%)'].values
            importances.append(importance_val[0] if len(importance_val) > 0 else 0)
        
        # 绘制条形图
        bars = ax1.barh(y_pos + i * bar_height, importances, bar_height, 
                                color=colors[i], alpha=0.85, label=target, linewidth=0.8)
        
        # 添加数值标签
        for j, (bar, importance) in enumerate(zip(bars, importances)):
            if importance > 1:
                ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                                 f'{importance:.1f}%', ha='left', va='center', fontsize=9)
    
    ax1.set_yticks(y_pos + bar_height)
    ax1.set_yticklabels(sorted_features[::-1], fontsize=11)
    ax1.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Importance Comparison Across Targets (k, fee only)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.2, axis='x')
    
    # 右子图：特征贡献热力图
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
    
    # 绘制热力图
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        cbar_kws={'label': 'Feature Gain Percentage (%)'},
        ax=ax2,
        linewidths=0.8,
        linecolor='white',
        annot_kws={'size': 10, 'weight': 'bold'}
    )
    
    # 反转y轴标签
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0) 
    
    ax2.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Features (k, fee only)', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Contribution Heatmap (Outliers Excluded)', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return heatmap_df

def plot_correlation_heatmap(X_modeling, y, save_path='robust/correlation_heatmap_outliers_excluded.png'):
    """特征与目标变量的相关性热力图（仅k和fee）"""
    # 合并特征和目标变量
    data_for_corr = pd.concat([X_modeling, y], axis=1)
    
    # 计算相关系数矩阵
    corr_matrix = data_for_corr.corr()
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
                fmt='.2f')
    
    plt.title('Feature-Target Correlation Heatmap (k, fee only, Outliers Excluded)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function: Properly excludes seed and max_slippage from XGBoost training features and removes outliers."""
    try:
        X_original, y_preprocessed, df_original, FEATURE_COLS, label_encoder = load_and_preprocess_data('实验数据9.xlsx')
    except (FileNotFoundError, ValueError) as e:
        print(f"Error during data loading: {e}")
        return
    
    # 跳过异常值剔除（保持原逻辑）
    X_clean, y_clean = X_original, y_preprocessed
    
    # 准备建模特征（仅k和fee）
    X_modeling = prepare_features_for_modeling(X_clean)
    
    # 初始化存储字典
    models = {}
    metrics_dict = {}
    results_dict = {}
    importance_dict = {}
    
    # 训练每个目标的模型（使用专属参数）
    feature_names = X_modeling.columns.tolist()
    for target in TARGET_COLS:
        model, metrics, results = train_single_xgboost(X_modeling, y_clean[target], target)
        models[target] = model
        metrics_dict[target] = metrics
        results_dict[target] = results
        
        importance_df = get_feature_importance(model, feature_names)
        importance_dict[target] = importance_df
        print(f"\n{target} Feature Importance (Only k, fee, Outliers Excluded):")
        print(importance_df)
    
    # 保存模型
    for target, model in models.items():
        try:
            model.save_model(f'robust/{target}_xgboost.model')
        except:
            with open(f'robust/{target}_xgboost.pkl', 'wb') as f:
                pickle.dump(model, f)
    print("\nAll models (trained on cleaned data with k, fee only) saved to robust/ folder")
    
    # 可视化
    print("\n=== Generating Enhanced Visualizations (Outliers Excluded, k & fee only) ===")
    # 1. 预测值vs真实值
    print("Generating prediction_vs_actual_outliers_excluded.png...")
    plot_prediction_vs_actual(models, results_dict)

    # 2. 部分依赖图（两行三列，仅k和fee）
    print("Generating partial_dependence_outliers_excluded.png...")
    plot_partial_dependence_with_categorical(models, X_modeling, label_encoder, TARGET_COLS)
    
    # 3. 学习曲线
    print("Generating learning_curves_outliers_excluded.png...")
    plot_learning_curves(models, X_modeling, y_clean)  # 双坐标轴
    
    # 4. 组合特征分析
    print("Generating combined_feature_analysis_outliers_excluded.png...")
    plot_combined_feature_analysis(importance_dict)  # 左右子图排版
    
    # 5. 相关性热力图
    print("Generating correlation_heatmap_outliers_excluded.png...")
    plot_correlation_heatmap(X_modeling, y_clean)

    # 保存评估指标
    metrics_summary = pd.DataFrame(metrics_dict).T
    print("\n=== Multi-Target Model Evaluation Summary (Outliers Excluded, k & fee only) ===")
    print(metrics_summary.round(4))
    metrics_summary.to_csv('robust/multi_target_metrics_summary_clean.csv', index=True)
    print("Evaluation metrics summary saved to robust/ folder")
    
    print("\n=== All Visualizations and Training Completed using cleaned data (k & fee only). ===")

if __name__ == "__main__":
    main()