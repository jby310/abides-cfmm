import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Define parameter ranges and grid density
PARAM_BOUNDS = {
  'k': (1e9, 1e12),     # Pool Size (k) range
  'fee': (0.0001, 0.1), # Fee (fee) range
  'max_slippage': 0.05  # Fixed slippage value (mid-point)
}

# Grid Density (higher value means finer grid)
GRID_DENSITY = 50
# Coexistence Score Thresholds (T1 and T2, will be adaptively determined later)
THRESHOLDS = None

# --- New Slice Parameters ---
K_SLICE = 1e9       # Fixed k for the k-slice plot
FEE_SLICE = 0.003   # Fixed fee for the fee-slice plot
# ---------------------------

def load_models(model_dir='robust'):
  """Load three trained XGBoost models."""
  models = {}
  target_names = ['depth_mean', 'spread_mean', 'volume_mean']
  
  for target in target_names:
    model_path = os.path.join(model_dir, f'{target}_xgboost.model')
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    model = xgb.Booster()
    model.load_model(model_path)
    models[target] = model
  
  return models

def generate_parameter_grid(bounds, density):
  """Generate a fine parameter grid."""
  # Use log space for k (due to large range)
  k_log = np.linspace(np.log10(bounds['k'][0]), np.log10(bounds['k'][1]), density)
  k_values = 10 ** k_log
  
  # Use linear space for fee
  fee_values = np.linspace(bounds['fee'][0], bounds['fee'][1], density)
  
  # Create grid dataframe
  grid_data = []
  for k in k_values:
    for fee in fee_values:
      grid_data.append({
        'k': k,
        'fee': fee,
        'max_slippage': bounds['max_slippage'],
        'seed_encoded': 0
      })
  
  return pd.DataFrame(grid_data), k_values, fee_values

def calculate_coexistence_score(models, grid_df):
  """Calculate the Coexistence Score."""
  # Prepare XGBoost input format
  dmatrix = xgb.DMatrix(grid_df)
  predictions = {}
  for target, model in models.items():
    predictions[target] = model.predict(dmatrix)
  coexistence_scores = (predictions['depth_mean'] + predictions['volume_mean']) / \
        (10e-8 + np.exp(-predictions['spread_mean']))
  return coexistence_scores, predictions

def calculate_adaptive_thresholds(scores, lower_percentile=45, upper_percentile=55):
  """
  Calculate two adaptive thresholds: T1 (lower_percentile) and T2 (upper_percentile).
  """
  T1 = np.percentile(scores, lower_percentile)
  T2 = np.percentile(scores, upper_percentile)
  return T1, T2

def analyze_critical_points(critical_indices, k_values, fee_values, scores, predictions):
  """Analyze and print information about the Critical/Neutral points."""
  global THRESHOLDS
  T1, T2 = THRESHOLDS
  
  num_critical = len(critical_indices[0])
  print(f"\nFound {num_critical} Neutral Zone Points ({T1:.2f} $\leq$ Score $\leq$ {T2:.2f})")
  print("Sample Neutral Zone Parameters and Predicted Values:")
  
  if num_critical == 0:
    print("No neutral zone points found in the 10% range.")
    return
    
  sample_indices = np.random.choice(num_critical, min(5, num_critical), replace=False)
  
  num_fee = len(fee_values)
  
  for idx_num, sample_idx in enumerate(sample_indices):
    i, j = critical_indices[0][sample_idx], critical_indices[1][sample_idx]
    k = k_values[i]
    fee = fee_values[j]
    
    # Calculate the flat index in the 1D scores/predictions array
    flat_idx = i * num_fee + j
    
    print(f"\nNeutral Point Sample {idx_num + 1}:")
    print(f" k: {k:.2e}, fee: {fee:.4f}")
    print(f" Coexistence Score: {scores[flat_idx]:.4f}")
    print(f" Predicted Depth Change: {predictions['depth_mean'][flat_idx]:.4f}")
    print(f" Predicted Spread Change: {predictions['spread_mean'][flat_idx]:.4f}")
    print(f" Predicted Volume Change: {predictions['volume_mean'][flat_idx]:.4f}")

# --- 通用 3D 绘图函数 (保持不变) ---

def plot_3d_surface(k_values, fee_values, metric_scores, metric_name, color_map='viridis'):
  """
  Draw a 3D Surface Plot for a given metric.
  """
  score_matrix = metric_scores.reshape(len(k_values), len(fee_values))
  X, Y = np.meshgrid(fee_values, np.log10(k_values))
  Z = score_matrix
  
  fig = plt.figure(figsize=(16, 12))
  ax = fig.add_subplot(111, projection='3d')
  
  cmap = plt.cm.get_cmap(color_map)
  norm = plt.Normalize(Z.min(), Z.max())

  surface = ax.plot_surface(
    X, Y, Z, 
    cmap=cmap, 
    norm=norm,
    linewidth=0, 
    antialiased=True, 
    alpha=0.9 
  )

  title = f'3D Surface Map of Predicted {metric_name.replace("_mean", "").replace("_", " ").title()} Change'
  z_label = f'Predicted {metric_name.replace("_mean", "").replace("_", " ").title()} Change'
  
  ax.set_title(title, fontsize=18, pad=20)
  ax.set_xlabel('Fee (fee)', fontsize=14, labelpad=15)
  ax.set_ylabel(r'$\log_{10}(k)$ (Pool Size)', fontsize=14, labelpad=15)
  ax.set_zlabel(z_label, fontsize=14, labelpad=15)

  fig.colorbar(surface, shrink=0.6, aspect=20, label=z_label)

  ax.view_init(elev=30, azim=210) 
  
  plt.tight_layout()
  filename = f'robust/{metric_name}_surface_plot.png'
  plt.savefig(filename, dpi=300, bbox_inches='tight')
  plt.show()
  print(f"\n3D Surface Plot for '{metric_name}' saved to {filename}")


# --- 新增组合 2D 横切面绘图函数 ---

def plot_combined_2d_slices(k_values, fee_values, predictions_dict, k_slice, fee_slice):
  """
  Draw a combined (2x3) subplot figure for 2D slices of depth_mean, spread_mean, and volume_mean.
  
  Rows: Fixed k slice (top), Fixed fee slice (bottom).
  Columns: depth_mean, spread_mean, volume_mean.
  """
  
  metrics = ['depth_mean', 'spread_mean', 'volume_mean']
  
  # 1. 准备索引
  # 找到 k_slice 的索引
  k_log_values = np.log10(k_values)
  target_k_log = np.log10(k_slice)
  k_idx = np.argmin(np.abs(k_log_values - target_k_log))
  k_actual = k_values[k_idx]
  
  # 找到 fee_slice 的索引
  fee_idx = np.argmin(np.abs(fee_values - fee_slice))
  fee_actual = fee_values[fee_idx]
  
  # 2. 创建 (2x3) 子图
  fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey='row')
  plt.suptitle(
      f'2D Slices of Predicted Market Impact Changes\n'
      f'Top Row: Fixed k $\\approx$ {k_actual:.2e} | Bottom Row: Fixed Fee $\\approx$ {fee_actual:.4f}', 
      fontsize=16, y=1.02
  )

  # 颜色和标记配置
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
  markers = ['o', 's', '^']

  for col, metric_name in enumerate(metrics):
      metric_scores = predictions_dict[metric_name]
      score_matrix = metric_scores.reshape(len(k_values), len(fee_values))
      metric_title = metric_name.replace("_mean", "").replace("_", " ").title()
      
      # --- Row 1: Fixed k, Z vs Fee (axes[0, col]) ---
      ax1 = axes[0, col]
      k_slice_data = score_matrix[k_idx, :] # Row slice
      
      ax1.plot(fee_values, k_slice_data, marker=markers[col], linestyle='-', color=colors[col], markersize=3)
      ax1.axhline(0, color='r', linestyle='--', linewidth=1.5) # 0 横线
      
      # 标题只在第一行显示，Y轴标签只在第一列显示
      ax1.set_title(f'{metric_title}', fontsize=14)
      if col == 0:
          ax1.set_ylabel(f'Predicted Change (Fixed k)', fontsize=12)
      
      # X轴标签只在第二行显示
      ax1.tick_params(axis='x', labelbottom=False)
      ax1.grid(True, linestyle=':', alpha=0.7)


      # --- Row 2: Fixed Fee, Z vs log10(k) (axes[1, col]) ---
      ax2 = axes[1, col]
      fee_slice_data = score_matrix[:, fee_idx] # Column slice
      
      ax2.plot(k_log_values, fee_slice_data, marker=markers[col], linestyle='-', color=colors[col], markersize=3)
      ax2.axhline(0, color='r', linestyle='--', linewidth=1.5) # 0 横线

      ax2.set_xlabel(r'$\log_{10}(k)$ (Pool Size)', fontsize=12)
      if col == 0:
          ax2.set_ylabel(f'Predicted Change (Fixed Fee)', fontsize=12)
      
      # 调整 X 轴刻度，只显示主要刻度
      ax2.set_xticks(np.log10(PARAM_BOUNDS['k']))
      ax2.grid(True, linestyle=':', alpha=0.7)

  plt.tight_layout(rect=[0, 0, 1, 0.98]) # 调整布局以适应 suptitle
  filename = f'robust/combined_2d_slices.png'
  plt.savefig(filename, dpi=300, bbox_inches='tight')
  plt.show()
  print(f"\nCombined (2x3) 2D Slice Plot saved to {filename}")


# --- Execution Block Modification ---
if __name__ == "__main__":
  os.makedirs('robust', exist_ok=True) 

  print("Loading models...")
  try:
    models = load_models()
  except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure you have the 'robust' directory with the required XGBoost model files.")
    exit()
    
  print(f"Generating parameter grid (Density: {GRID_DENSITY}x{GRID_DENSITY})...")
  grid_df, k_values, fee_values = generate_parameter_grid(PARAM_BOUNDS, GRID_DENSITY)
  
  print("Calculating Coexistence Score and individual predictions...")
  scores, predictions = calculate_coexistence_score(models, grid_df)
  
  # 将 Coexistence Score 加入 predictions 字典，但只用于 3D 图绘制，不用于 2x3 组合图
  predictions['Coexistence_Score'] = scores
  
  print("Calculating adaptive thresholds (T1 and T2)...")
  THRESHOLDS = calculate_adaptive_thresholds(scores, lower_percentile=45, upper_percentile=55)
  T1, T2 = THRESHOLDS
  print(f"Adaptive Threshold T1 (45th percentile): {T1:.2f}")
  print(f"Adaptive Threshold T2 (55th percentile): {T2:.2f}")

  # List of all metrics to process for 3D plots
  metrics_to_plot_3d = {
      'Coexistence_Score': {'scores': scores, 'cmap': 'coolwarm'},
      'depth_mean': {'scores': predictions['depth_mean'], 'cmap': 'viridis'},
      'spread_mean': {'scores': predictions['spread_mean'], 'cmap': 'seismic'},
      'volume_mean': {'scores': predictions['volume_mean'], 'cmap': 'magma'}
  }

  for metric_name, data in metrics_to_plot_3d.items():
      # 1. Draw 3D Plot (保持不变)
      print(f"\nDrawing 3D Surface Plot for '{metric_name}'...")
      plot_3d_surface(k_values, fee_values, data['scores'], metric_name, color_map=data['cmap'])
      print("-" * 50)
      
  # 2. Draw Combined (2x3) 2D Slices (替换了之前的 plot_2d_slices 调用)
  print("\nDrawing Combined (2x3) 2D Slice Plots for depth, spread, and volume mean...")
  # 传入只包含 depth_mean, spread_mean, volume_mean 的子集
  slices_predictions = {k: v for k, v in predictions.items() if k != 'Coexistence_Score'}
  plot_combined_2d_slices(k_values, fee_values, slices_predictions, K_SLICE, FEE_SLICE)
      
  # 分析中立区点 (Coexistence Score)
  score_matrix = scores.reshape(len(k_values), len(fee_values))
  neutral_zone_mask = (score_matrix >= T1) & (score_matrix <= T2)
  critical_indices = np.where(neutral_zone_mask)
  analyze_critical_points(critical_indices, k_values, fee_values, scores, predictions)