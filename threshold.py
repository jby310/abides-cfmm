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
    'k': (1e9, 1e12),        # Pool Size (k) range
    'fee': (0.0001, 0.1),    # Fee (fee) range
    'max_slippage': 0.05     # Fixed slippage value (mid-point)
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
    Calculate two adaptive thresholds:
    T1 (lower_percentile): 45% of scores are below T1 (Low Coexistence).
    T2 (upper_percentile): 55% of scores are below T2, meaning 45% are above T2 (High Coexistence).
    10% of scores are between T1 and T2 (Critical/Neutral Zone).
    """
    T1 = np.percentile(scores, lower_percentile)
    T2 = np.percentile(scores, upper_percentile)
    return T1, T2

def plot_coexistence_heatmap(k_values, fee_values, scores, thresholds, predictions):
    """Draw the Coexistence Score heatmap, mask the low-score area, and highlight the neutral zone."""
    T1, T2 = thresholds
    score_matrix = scores.reshape(len(k_values), len(fee_values))
    
    # Use a slightly larger figure to accommodate labels and colorbar better
    plt.figure(figsize=(14, 10)) 
    
    # 1. Define the Colormap (Diverging, centered at 0)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Pre-calculate tick labels
    x_tick_labels = np.round(fee_values, 4)
    y_tick_labels = np.round(np.log10(k_values), 1)
    
    # 2. Draw the main heatmap (Coexistence Score)
    ax = sns.heatmap(
        score_matrix,
        cmap=cmap,
        center=0,
        annot=False,
        fmt=".2f",
        xticklabels=x_tick_labels,
        yticklabels=y_tick_labels,
        cbar_kws={'label': 'Coexistence Score'}
    )
    
    # 3. Mask the Low Coexistence Area (Score < T1, bottom 45%)
    low_coexistence_mask = score_matrix < T1
    low_mask_color_rgba = (0.9, 0.9, 0.9, 0.7)
    low_mask_color = ListedColormap([low_mask_color_rgba]) 
    
    sns.heatmap(
        np.where(low_coexistence_mask, 1, np.nan),
        cmap=low_mask_color,
        cbar=False,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )

    # 4. Highlight the Critical/Neutral Zone (T1 <= Score <= T2, middle 10%)
    neutral_zone_mask = (score_matrix >= T1) & (score_matrix <= T2)
    neutral_mask_color_rgba = (0.9, 0.7, 0.2, 0.3)
    neutral_mask_color = ListedColormap([neutral_mask_color_rgba]) 
    
    sns.heatmap(
        np.where(neutral_zone_mask, 1, np.nan),
        cmap=neutral_mask_color,
        cbar=False,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )
    
    # --- FIX 1: Restore Ticks/Labels (Crucial fix) ---
    tick_locations = np.arange(len(x_tick_labels)) + 0.5 
    ax.set_xticks(tick_locations)
    ax.set_yticks(tick_locations)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_tick_labels, rotation=0)

    # Set labels and title
    ax.set_title('Coexistence Score Heatmap Across Parameter Space', fontsize=15, pad=20)
    ax.set_xlabel('Fee (fee)', fontsize=12)
    ax.set_ylabel(r'$\log_{10}(k)$ (Pool Size)', fontsize=12)
    
    # --- FIX 2: Add Custom Legend (Positioned to avoid Cbar) ---
    
    # Create custom patches for the legend
    low_patch = mpatches.Patch(
        color=low_mask_color_rgba, 
        label=f'Low Coexistence Zone (Score < {T1:.2f})'
    )
    neutral_patch = mpatches.Patch(
        color=neutral_mask_color_rgba, 
        label=f'Neutral Zone ({T1:.2f} $\leq$ Score $\leq$ {T2:.2f})'
    )
    # Use a color from the high end of the main colormap
    high_patch = mpatches.Patch(
        color=cmap(0.9), 
        label=f'High Coexistence Zone (Score > {T2:.2f})'
    )
    
    # Add legend to the plot, LOCATED INSIDE the plot area at 'upper left'
    # This placement avoids the default right-side Colorbar.
    plt.legend(
        handles=[high_patch, neutral_patch, low_patch], 
        loc='upper left', # Fixed position inside the plot area
        title='Score Zones',
        frameon=True, # Optional: Keep frame for visibility
        fontsize=10 
    )
    
    # Use standard tight_layout to adjust margins
    plt.tight_layout() 
    plt.savefig('robust/coexistence_heatmap_three_zones_v5_left_legend.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    critical_indices = np.where(neutral_zone_mask)
    return critical_indices

# ... (analyze_critical_points and main execution block remain the same)

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
        print(f"  k: {k:.2e}, fee: {fee:.4f}")
        print(f"  Coexistence Score: {scores[flat_idx]:.4f}")
        print(f"  Predicted Depth Change: {predictions['depth_mean'][flat_idx]:.4f}")
        print(f"  Predicted Spread Change: {predictions['spread_mean'][flat_idx]:.4f}")
        print(f"  Predicted Volume Change: {predictions['volume_mean'][flat_idx]:.4f}")

if __name__ == "__main__":
    # Load models
    print("Loading models...")
    try:
        models = load_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have the 'robust' directory with the required XGBoost model files.")
        exit()
        
    # Generate parameter grid
    print(f"Generating parameter grid (Density: {GRID_DENSITY}x{GRID_DENSITY})...")
    grid_df, k_values, fee_values = generate_parameter_grid(PARAM_BOUNDS, GRID_DENSITY)
    
    # Calculate Coexistence Score
    print("Calculating Coexistence Score...")
    scores, predictions = calculate_coexistence_score(models, grid_df)
    
    # Calculate adaptive thresholds (T1=45th percentile, T2=55th percentile)
    print("Calculating adaptive thresholds (T1 and T2)...")
    THRESHOLDS = calculate_adaptive_thresholds(scores, lower_percentile=45, upper_percentile=55)
    T1, T2 = THRESHOLDS
    print(f"Adaptive Threshold T1 (45th percentile): {T1:.2f}")
    print(f"Adaptive Threshold T2 (55th percentile): {T2:.2f}")
    
    # Draw heatmap and highlight the three zones
    print("Drawing Coexistence Score Heatmap...")
    critical_indices = plot_coexistence_heatmap(
        k_values, fee_values, scores, THRESHOLDS, predictions
    )
    
    # Analyze critical points (middle 10% zone)
    analyze_critical_points(critical_indices, k_values, fee_values, scores, predictions)
    
    print("\nHeatmap saved to robust/coexistence_heatmap_three_zones_v5_left_legend.png")