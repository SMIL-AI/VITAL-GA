# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
import pandas as pd
import numpy as np

# read csv file
df_y = pd.read_csv('/home/disk/hao/poi_v2_healthcare/weekly/accessibility/acess_cal_v2/covid/difference/covid_accessibility_differences_2020_2022.csv')

# Filter out CBGs where accessibility_2020 is zero (to avoid division by zero)
print(f"Original data shape: {df_y.shape}")
print(f"Records with accessibility_2020 = 0: {(df_y['accessibility_2020'] == 0).sum()}")

df_y = df_y[df_y['accessibility_2020'] != 0]
# Calculate percentage reduction: (2020 - 2022) / 2020
#df_y['accessibility_percentage_reduction'] = (df_y['accessibility_2020'] - df_y['accessibility_2022']) / df_y['accessibility_2020']
df_y['accessibility_percentage_reduction'] = (df_y['accessibility_2022'] - df_y['accessibility_2020']) / df_y['accessibility_2020']
epsilon = 1e-8

# Define sign-preserving log transformation to handle negative values
def signed_log_transform(x, epsilon=1e-8):
    """Apply sign-preserving log transformation that handles negative values"""
    return np.where(x == 0, 0, np.log(1 + np.abs(x) + epsilon) * np.sign(x))

# Apply sign-preserving log transform to accessibility_percentage_reduction
df_y['accessibility_percentage_reduction_log'] = signed_log_transform(df_y['accessibility_percentage_reduction'], epsilon)

# print the first 5 rows
print("\nFirst 5 rows:")
print(df_y.head())


# print the first 5 rows
print("\nFirst 5 rows:")
print(df_y.head())

# print the last 5 rows
print("\nLast 5 rows:")
print(df_y.tail())

# print the column names
print("\nColumn names:")
print(df_y.columns)

# Show statistics of the new target variable
print("\n'accessibility_percentage_reduction' statistics:")
print(df_y.groupby('category')['accessibility_percentage_reduction'].describe())
print("\n'accessibility_percentage_reduction_log' statistics:")
print(df_y.groupby('category')['accessibility_percentage_reduction_log'].describe())

# Load metropolitan area county data and create metro_area feature
import pandas as pd

def add_metro_area_feature(df_x):
    """
    Add metro_area feature to df_x based on CBG_ID and metropolitan counties
    """
    # Load metropolitan counties data
    metro_counties = pd.read_csv('/home/disk/hao/poi_v2_healthcare/weekly/data/georgia_14_metropolitan_counties.csv', encoding='ISO-8859-1')
    
    # Get the list of metropolitan county FIPS codes (first 5 digits of CBG_ID)
    metro_fips = set(metro_counties['County FIPS'].astype(str))
    
    # Ensure CBG_ID is string and extract first 5 digits (county FIPS)
    df_x['CBG_ID'] = df_x['CBG_ID'].astype(str)
    df_x['county_fips'] = df_x['CBG_ID'].str[:5]
    
    # Create metro_area feature: 1 if county is metropolitan, 0 otherwise
    df_x['metro_area'] = df_x['county_fips'].apply(lambda x: 1 if x in metro_fips else 0)
    
    # Drop the temporary county_fips column
    df_x.drop(columns=['county_fips'], inplace=True)
    
    print(f"Added metro_area feature:")
    print(f"Metro areas (1): {df_x['metro_area'].sum()}")
    print(f"Non-metro areas (0): {(df_x['metro_area'] == 0).sum()}")
    
    return df_x
import pandas as pd

# read csv file
df_x = pd.read_csv('/home/disk/hao/poi_v2_healthcare/weekly/data/19_23_demo_0725_new/2020_demo_june_data.csv')

# rename_dict for column names
rename_dict = {
    "CBG_ID": "CBG_ID",
    "Sex By Age: Male": "Male",
    "Sex By Age: Female": "Fem", 
    "Total Population": "Pop_Tot",
    "Race: White alone": "White",
    "Means Of Transportation To Work: Total": "Transport_Tot",
    "Travel Time To Work: Total": "Com_Time_Tot",
    "Household Income: Total": "HH_Inc_Tot",  
    "Poverty Status of Families: Total": "Fam_Pov_Tot",
    "Poverty Status of Families: Below poverty level": "Fam_Pov_B",
    "Poverty Status of Families: At or above poverty level": "Fam_Pov_A",
    "Employment Status (16+): Total": "Emp_Tot",
    "Employment Status (16+): In labor force": "LF",
    "Employment Status (16+): Not in labor force": "NLF",
    "Work Status (16–64): Total": "WS_Tot",
    "Race: Non-White": "Non_White",
    "Transport: Active": "Transport_Act",
    "Transport: Private Motorized": "Transport_Pri",
    "Transport: Public Transit": "Transport_Pub",
    "Transport: Shared": "Transport_Sha",
    "Transport: Work From Home": "Transport_WFH",
    "Commute: Short": "Com_S",
    "Commute: Medium": "Com_M",
    "Commute: Long": "Com_L",
    "Departure_Time: Morning": "Dept_M",
    "Departure_Time: Afternoon": "Dept_A",
    "Departure_Time: Night": "Dept_N",
    "Marital_Status: Now married": "Married",
    "Marital_Status: Other": "Marital_O",
    "Education: Before high school": "Edu_Pre_HS",
    "Education: During high school": "Edu_HS",
    "Education: Before bachelor": "Edu_Pre_BS",
    "Education: Bachelor and above": "Edu_BS_Plus",
    "Income: Low": "Inc_L",
    "Income: Medium": "Inc_M",
    "Income: High": "Inc_H",
    "Work_Status: Overworked": "OW",
    "Population: Youth (0-14)": "Youth",
    "Population: Working Age (15-64)": "Work_Age",
    "Population: Senior (65+)": "Senior",
    "No Health Insurance": "No_HI",
    "MaleRatio": "Male_Pt",  
    "FemaleRatio": "Fem_Pt",
    "WhiteRatio": "White_Pt",
    "NonWhiteRatio": "Non_White_Pt",  
    "ActiveTransRatio": "Act_Trans_Pt",  
    "PrivateMotorRatio": "Pri_Veh_Pt",
    "PublicTransitRatio": "Pub_Transit_Pt",  
    "SharedTransRatio": "Sha_Trans_Pt",  
    "WorkFromHomeRatio": "WFH_Pt",
    "CommuteShortRatio": "Com_S_Pt",
    "CommuteMediumRatio": "Com_M_Pt",
    "CommuteLongRatio": "Com_L_Pt",
    "DepMorningRatio": "Dept_M_Pt",
    "DepAfternoonRatio": "Dept_A_Pt",
    "DepNightRatio": "Dept_N_Pt",
    "MarriedRatio": "Married_Pt",
    "OtherMaritalRatio": "Marital_O_Pt",
    "Edu_Before_HS_Ratio": "Edu_Pre_HS_Pt",
    "Edu_HS_Ratio": "Edu_HS_Pt",
    "Edu_Below_BA_Ratio": "Edu_Pre_BS_Pt",
    "Edu_Above_BA_Ratio": "Edu_BS_Plus_Pt",
    "IncomeLowRatio": "Inc_L_Pt",
    "IncomeMidRatio": "Inc_M_Pt",
    "IncomeHighRatio": "Inc_H_Pt",
    "YouthRatio": "Youth_Pt",
    "WorkingAgeRatio": "Work_Age_Pt",
    "SeniorRatio": "Senior_Pt",
    "InLaborForceRatio": "LF_Pt",
    "NotInLaborForceRatio": "NLF_Pt",  
    "PovertyBelowRatio": "Fam_Pov_B_Pt",  
    "PovertyAboveRatio": "Fam_Pov_A_Pt",
    "OverworkedRatio": "OW_Pt",
    "NoHealthInsRatio": "No_HI_Pt",
}

# Apply column renaming - only rename columns that exist in the dataframe
existing_cols_to_rename = {old_name: new_name for old_name, new_name in rename_dict.items() if old_name in df_x.columns}
df_x.rename(columns=existing_cols_to_rename, inplace=True)

print("Columns before renaming:", df_x.columns.tolist())
print(f"Renamed {len(existing_cols_to_rename)} columns")

# print the first 5 rows
# print(df_x.head())

# # print the last 5 rows
# print(df_x.tail())
print(df_x.columns.tolist())
df_x.drop(columns=['HealthAccessibility', 'FoodAccessibility', 'EduAccessibility', 'EntertainmentAccessibility', 'Accessibility'], inplace=True, errors='ignore')
print(df_x.columns.tolist())
# Add metro_area feature to df_x
df_x = add_metro_area_feature(df_x)
print("Updated df_x columns:", df_x.columns.tolist())
# Data preprocessing and merging
def prepare_analysis_data(df_y, df_x):
    """s
    Merge and prepare data for analysis
    """
    # Ensure GEOID formats match
    df_y['GEOID'] = df_y['GEOID'].astype(str)
    df_x['CBG_ID'] = df_x['CBG_ID'].astype(str)
    
    # Merge the datasets
    merged_df = df_y.merge(df_x, left_on='GEOID', right_on='CBG_ID', how='inner')
    
    print(f"Successfully merged {len(merged_df)} records")
    print(f"Categories: {merged_df['category'].value_counts()}")
    
    # Identify demographic columns (exclude ID columns)
    demographic_cols = [col for col in df_x.columns if col != 'CBG_ID']
    
    # Remove demographic columns with all zeros or very low variance
    variance_threshold = 0.01
    valid_demo_cols = []
    
    for col in demographic_cols:
        col_var = merged_df[col].var()
        if col_var > variance_threshold and merged_df[col].sum() > 0:
            valid_demo_cols.append(col)
    
    print(f"Using {len(valid_demo_cols)} demographic variables out of {len(demographic_cols)}")
    
    return merged_df, valid_demo_cols
#df_y.drop(columns=['metro_area'], inplace=True)
# Prepare data
merged_df, demographic_cols = prepare_analysis_data(df_y, df_x)

# Display basic statistics
print("\nAccessibility percentage reduction statistics by category:")
print(merged_df.groupby('category')['accessibility_percentage_reduction'].describe())
print("\nAccessibility percentage reduction log transformed statistics by category:")
print(merged_df.groupby('category')['accessibility_percentage_reduction_log'].describe())


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr

def normalize_shap_preserve_ranking(shap_values):
    """
    Apply feature-wise min-max normalization to SHAP values for better visualization 
    when features have different scales.
    
    Each feature is independently scaled to [-1, 1] range, ensuring all features 
    get equal visual weight in SHAP plots regardless of their original scale.
    
    Parameters:
    -----------
    shap_values : numpy.ndarray
        SHAP values array with shape (n_samples, n_features)
    
    Returns:
    --------
    numpy.ndarray : Normalized SHAP values, each feature scaled to [-1, 1]
    """
    normalized = np.zeros_like(shap_values)
    
    for i in range(shap_values.shape[1]):
        feature_shap = shap_values[:, i]
        feat_min = np.min(feature_shap)
        feat_max = np.max(feature_shap)
        
        if feat_max != feat_min:
            # Scale to [-1, 1]: 2 * (x - min) / (max - min) - 1
            normalized[:, i] = 2 * (feature_shap - feat_min) / (feat_max - feat_min) - 1
        else:
            # Handle constant features (no variance)
            normalized[:, i] = 0
    
    return normalized

def analyze_single_category(merged_df, category_name, demographic_cols, 
                          save_plots=True, show_plots=True, plot_format='png'):
    """
    Analyze accessibility differences for a single healthcare category
    
    Parameters:
    -----------
    save_plots : bool, default True
        Whether to save plots to files
    show_plots : bool, default True  
        Whether to display plots interactively
    plot_format : str, default 'png'
        Format for saved plots ('png', 'pdf', 'svg', etc.)
    """
    import os
    
    # Create output directory for plots
    output_dir = 'shap_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS FOR: {category_name}")
    print(f"{'='*50}")
    
    # Filter data for this category
    category_data = merged_df[merged_df['category'] == category_name].copy()
    print(demographic_cols)
    # Prepare features and target - include accessibility_2020 as an additional feature
    feature_cols = demographic_cols + ['accessibility_2020']
    print(feature_cols)
    X = category_data[feature_cols]
    y = - category_data['accessibility_percentage_reduction_log']
    
    # Check for and handle NaN values in target variable
    nan_mask = ~(np.isnan(y) | np.isinf(y))
    if nan_mask.sum() == 0:
        print(f"ERROR: All target values are NaN/infinite for category {category_name}")
        return None, None, None
    
    nan_count = (~nan_mask).sum()
    if nan_count > 0:
        print(f"Warning: Removing {nan_count} rows with NaN/infinite target values out of {len(y)} total rows")
        X = X[nan_mask]
        y = y[nan_mask]
        category_data = category_data[nan_mask]
    
    # Handle missing values and standardize
    X_clean = X.fillna(0)  # Fill NaN with 0 for demographic data
    
    # Remove features with zero variance
    X_clean = X_clean.loc[:, X_clean.var() > 0]
    
    print(f"Data shape: {X_clean.shape}")
    print(f"Target variable range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42
    )
    
    # Dictionary to store results
    results = {}
    
    # CatBoost with hyperparameter tuning
    print("\nPerforming CatBoost hyperparameter tuning...")
    
    #Define parameter grid for CatBoost
    param_grid = {
        'iterations': [200,400,600,800],
        'learning_rate': [0.01, 0.05,0.1],
        'depth': [8,10,12],
        'l2_leaf_reg': [1,3],
        'random_strength': [1]
    }
    # param_grid = {
    #     'iterations': [200],
    #     'learning_rate': [0.01],
    #     'depth': [8],
    #     'l2_leaf_reg': [1],
    #     'random_strength': [1]
    # }
    
    # Base CatBoost model
    catboost_base = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )
    
    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        catboost_base,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_catboost = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred_cb = best_catboost.predict(X_test)
    
    # Calculate regular metrics
    r2 = r2_score(y_test, y_pred_cb)
    mse = mean_squared_error(y_test, y_pred_cb)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_cb)
    
    # Calculate normalized metrics
    y_range = y_test.max() - y_test.min()
    y_mean = y_test.mean()
    
    nrmse = rmse / y_range  # Normalized by range
    nmse = mse / (y_range ** 2)  # Normalized by range squared
    nmae = mae / y_range  # Normalized by range
    
    # Alternative normalizations (by mean)
    nrmse_mean = rmse / y_mean
    nmse_mean = mse / (y_mean ** 2)
    nmae_mean = mae / y_mean
    
    results['CatBoost'] = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'nrmse_range': nrmse,
        'nmse_range': nmse,
        'nmae_range': nmae,
        'nrmse_mean': nrmse_mean,
        'nmse_mean': nmse_mean,
        'nmae_mean': nmae_mean
    }
    
    # Print results
    print("\nModel Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.6f}")
    print(f"\nNormalized Metrics (by range):")
    print(f"  NRMSE: {nrmse:.4f}")
    print(f"  NMSE: {nmse:.6f}")
    print(f"  NMAE: {nmae:.4f}")
    print(f"\nNormalized Metrics (by mean):")
    print(f"  NRMSE: {nrmse_mean:.4f}")
    print(f"  NMSE: {nmse_mean:.6f}")
    print(f"  NMAE: {nmae_mean:.4f}")
    
    # SHAP analysis with CatBoost
    explainer = shap.TreeExplainer(best_catboost)
    shap_values = explainer.shap_values(X_test)
    
    print(f"\nSHAP values shape: {shap_values.shape}")
    
    # Calculate Spearman correlation between features and their SHAP values
    correlations = {}
    for i, feature in enumerate(X_test.columns):
        # Calculate Spearman correlation between feature values and SHAP values
        corr, p_value = spearmanr(X_test.iloc[:, i], shap_values[:, i])
        correlations[feature] = abs(corr)  # Use absolute correlation
    
    # Define correlation threshold (you can adjust this value)
    correlation_threshold = 0.7  # Features with correlation >= 0.5 will be selected
    
    # Sort features by correlation and filter by threshold
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    threshold_features = [feat[0] for feat in sorted_features if feat[1] >= correlation_threshold]
    threshold_indices = [X_test.columns.get_loc(feat) for feat in threshold_features]
    
    # If no features meet the threshold, fall back to top 5
    if len(threshold_features) == 0:
        print(f"Warning: No features meet correlation threshold of {correlation_threshold}. Using top 5 features instead.")
        threshold_features = [feat[0] for feat in sorted_features[:5]]
        threshold_indices = [X_test.columns.get_loc(feat) for feat in threshold_features]
    
    print("\nSpearman Correlation between Features and SHAP values:")
    for feat, corr in sorted_features:
        print(f"  {feat}: {corr:.4f}")
    
    print(f"\nFeatures meeting correlation threshold (>= {correlation_threshold}): {threshold_features}")
    print(f"Number of features selected: {len(threshold_features)}")
    
    # Create filtered datasets for threshold-selected features
    X_test_threshold = X_test[threshold_features]
    shap_values_threshold = shap_values[:, threshold_indices]
    
    original_importance = np.abs(shap_values_threshold).mean(axis=0)
    importance_order = np.argsort(original_importance)[::-1]  # Descending order

    # Sort features by original importance (both data and feature names)
    shap_values_sorted = shap_values_threshold[:, importance_order]
    X_test_sorted = X_test_threshold.iloc[:, importance_order]

    # Apply feature-wise min-max normalization to handle large scale differences between features
    # Each feature will be independently scaled to [-1, 1] for better visualization
    normalized_shap_sorted = normalize_shap_preserve_ranking(shap_values_sorted)

    # Plot 1: Scale-Fixed SHAP Plot (sort=False, already sorted)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        normalized_shap_sorted, 
        X_test_sorted,
        max_display=15,
        show=False,
        sort=False,  # Don't sort again! Already sorted by original importance
        alpha=0.7
    )
    plt.title(f'Scale-Fixed SHAP Plot - {category_name}')
    # Save the figure
    if save_plots:
        plt.savefig(f'{output_dir}/shap_scale_fixed_{category_name.replace(" ", "_").replace("/", "_")}.{plot_format}', 
                    dpi=400, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # SHAP Summary Plot - only for threshold-selected features
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_threshold, X_test_threshold, show=False)
    plt.title(f'SHAP Summary Plot - Features with Correlation >= {correlation_threshold} for {category_name}')
    plt.tight_layout()
    # Save the figure
    if save_plots:
        plt.savefig(f'{output_dir}/shap_summary_{category_name.replace(" ", "_").replace("/", "_")}.{plot_format}', 
                    dpi=400, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # SHAP Bar Plot - only for threshold-selected features
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_threshold, X_test_threshold, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance (Correlation >= {correlation_threshold}) for {category_name}')
    plt.tight_layout()
    # Save the figure
    if save_plots:
        plt.savefig(f'{output_dir}/shap_bar_{category_name.replace(" ", "_").replace("/", "_")}.{plot_format}', 
                    dpi=400, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return results, correlations, X_clean.columns

# Run analysis for each category
categories = merged_df['category'].unique()
all_results = {}
all_feature_importance = {}

for category in categories:
    results, feature_imp, feature_names = analyze_single_category(merged_df, category, demographic_cols)
    all_results[category] = results
    all_feature_importance[category] = feature_imp
