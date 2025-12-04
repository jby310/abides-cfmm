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
  'k': (1e9, 1e12),    # Pool Size (k) range
  'fee': (0.0001, 0.1),  # Fee (fee) range
  'max_slippage': 0.05   # Fixed slippage value (mid-point)
}

# Grid Density (higher value means finer grid)
GRID_DENSITY = 50
# Coexistence Score Thresholds (T1 and T2, will be adaptively determined later)
THRESHOLDS = None 

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
            (1 + np.abs(predictions['spread_mean']))
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

# --- 通用 3D 绘图函数 ---

def plot_3d_surface(k_values, fee_values, metric_scores, metric_name, color_map='viridis'):
  """
  Draw a 3D Surface Plot for a given metric.
  X-axis: Fee (fee), Y-axis: log10(k) (Pool Size), Z-axis: Metric Score.
  """
  # 1. Prepare data for 3D plot
  # Reshape the 1D scores array back into the 2D grid matrix
  score_matrix = metric_scores.reshape(len(k_values), len(fee_values))
  
  # Create the meshgrid for X (fee) and Y (log10(k))
  X, Y = np.meshgrid(fee_values, np.log10(k_values))
  Z = score_matrix
  
  # 2. Setup the 3D plot
  fig = plt.figure(figsize=(16, 12))
  ax = fig.add_subplot(111, projection='3d')
  

  # 3. Define the Colormap and Normalization
  cmap = plt.cm.get_cmap(color_map)
  norm = plt.Normalize(Z.min(), Z.max())

  # 4. Draw the 3D Surface Plot
  surface = ax.plot_surface(
    X, Y, Z, 
    cmap=cmap, 
    norm=norm,
    linewidth=0, 
    antialiased=True, 
    alpha=0.9 
  )

  # 5. Add Labels and Title
  title = f'3D Surface Map of Predicted {metric_name.replace("_mean", "").title()} Change'
  z_label = f'Predicted {metric_name.replace("_mean", "").title()} Change'
  
  ax.set_title(title, fontsize=18, pad=20)
  ax.set_xlabel('Fee (fee)', fontsize=14, labelpad=15)
  ax.set_ylabel(r'$\log_{10}(k)$ (Pool Size)', fontsize=14, labelpad=15)
  ax.set_zlabel(z_label, fontsize=14, labelpad=15)

  # Add a color bar
  fig.colorbar(surface, shrink=0.6, aspect=20, label=z_label)

  # Optional: Adjust the viewing angle for better perspective
  # Adjusted view_init slightly for a different angle compared to Coexistence Score
  ax.view_init(elev=30, azim=210) 
  
  plt.tight_layout()
  filename = f'robust/{metric_name}_surface_plot.png'
  plt.savefig(filename, dpi=300, bbox_inches='tight')
  plt.show()
  print(f"\n3D Surface Plot for '{metric_name}' saved to {filename}")


# --- Execution Block Modification ---
if __name__ == "__main__":
  # Make sure the 'robust' directory exists for saving files
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
  
  print("Calculating adaptive thresholds (T1 and T2)...")
  THRESHOLDS = calculate_adaptive_thresholds(scores, lower_percentile=45, upper_percentile=55)
  T1, T2 = THRESHOLDS
  print(f"Adaptive Threshold T1 (45th percentile): {T1:.2f}")
  print(f"Adaptive Threshold T2 (55th percentile): {T2:.2f}")

  # 1. 绘制 Coexistence Score 3D 图 (原有的)
  print("\nDrawing Coexistence Score 3D Surface Plot...")
  # 使用 coolwarm 区分正负变化
  plot_3d_surface(k_values, fee_values, scores, 'Coexistence_Score', color_map='coolwarm')
  
  # 2. 绘制 'depth_mean' 3D 图
  print("\nDrawing 'depth_mean' 3D Surface Plot...")
  # 深度越大越好，使用 sequential colormap (e.g., Viridis)
  plot_3d_surface(k_values, fee_values, predictions['depth_mean'], 'depth_mean', color_map='viridis')

  # 3. 绘制 'spread_mean' 3D 图
  print("\nDrawing 'spread_mean' 3D Surface Plot...")
  # Spread 是差值，越接近 0 越好。使用 diverging colormap (e.g., seismic)
  plot_3d_surface(k_values, fee_values, predictions['spread_mean'], 'spread_mean', color_map='seismic')

  # 4. 绘制 'volume_mean' 3D 图
  print("\nDrawing 'volume_mean' 3D Surface Plot...")
  # Volume 越大越好，使用 sequential colormap (e.g., Magma)
  plot_3d_surface(k_values, fee_values, predictions['volume_mean'], 'volume_mean', color_map='magma')
  
  # 分析中立区点 (Coexistence Score)
  score_matrix = scores.reshape(len(k_values), len(fee_values))
  neutral_zone_mask = (score_matrix >= T1) & (score_matrix <= T2)
  critical_indices = np.where(neutral_zone_mask)
  analyze_critical_points(critical_indices, k_values, fee_values, scores, predictions)