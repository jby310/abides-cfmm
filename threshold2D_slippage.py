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
    'max_slippage': np.linspace(0.01, 0.2, 100)   # Slippage range
}

GRID_DENSITY = 100  # Grid density for parameter space

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

def generate_parameter_grid(bounds):
    """Generate parameter grid with only max_slippage"""
    # Get slippage values from bounds definition
    slippage_values = bounds['max_slippage']
    
    # Create grid dataframe
    grid_data = []
    for slippage in slippage_values:
        grid_data.append({
            'max_slippage': slippage
        })
    
    return pd.DataFrame(grid_data), slippage_values

def calculate_coexistence_score(models, grid_df):
    """Calculate the Coexistence Score: ΔDepth + ΔVolume - ΔSpread"""
    # Prepare XGBoost input format
    dmatrix = xgb.DMatrix(grid_df)
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(dmatrix)
        
    # Calculate coexistence score
    coexistence_scores = predictions['depth_mean'] + predictions['volume_mean'] - predictions['spread_mean']
    return coexistence_scores, predictions

# --- 2D 绘图函数 ---
def plot_2d_slippage(slippage_values, metric_scores, metric_name, color='blue'):
    """Draw 2D plot of metric vs max_slippage"""
    plt.figure(figsize=(12, 6))
    plt.plot(slippage_values, metric_scores, marker='o', linestyle='-', color=color, markersize=4, alpha=0.7)
    
    # Mark the 0 line
    plt.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Zero Change Line')
    
    metric_title = metric_name.replace("_mean", "").replace("_", " ").title()
    plt.title(f'Predicted {metric_title} vs. $\\alpha$ (Max Slippage) at $k = 10^{{12}}$ and fee = 0.003', fontsize=16)
    plt.xlabel('$\\alpha$ (Max Slippage)', fontsize=14)
    plt.ylabel(f'Predicted {metric_title} Change (%)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Create filename and save
    filename = f'robust/{metric_name}_vs_slippage.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f"2D Plot for '{metric_name}' saved to {filename}")


# --- Execution Block ---
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
        
    print(f"Generating parameter grid...")
    grid_df, slippage_values = generate_parameter_grid(PARAM_BOUNDS)
    
    print("Calculating Coexistence Score and individual predictions...")
    scores, predictions = calculate_coexistence_score(models, grid_df)
    predictions['Coexistence_Score'] = scores  # Add score to predictions

    # Define metrics to plot with custom colors
    metrics_to_plot = {
        'Liquidity_Score': {'scores': scores, 'color': 'purple'},
        'depth_mean': {'scores': predictions['depth_mean'], 'color': 'blue'},
        'spread_mean': {'scores': predictions['spread_mean'], 'color': 'green'},
        'volume_mean': {'scores': predictions['volume_mean'], 'color': 'orange'}
    }

    for metric_name, data in metrics_to_plot.items():
        print(f"\nDrawing 2D Plot for '{metric_name}'...")
        plot_2d_slippage(slippage_values, data['scores'], metric_name, data['color'])
        print("-" * 50)