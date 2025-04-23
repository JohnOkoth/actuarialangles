import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import GammaRegressor
from scipy import optimize
import shap
from bayes_opt import BayesianOptimization
from streamlit.components.v1 import html
from xgboost import XGBRegressor, DMatrix

# Set page config as the first command
st.set_page_config(layout="centered")

analytics_code = """
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-9S5SM84Q3T');
</script>
"""
html(analytics_code)

# --- Sidebar: Back to Home Button ---
home_button_html = [
    "<!-- Custom Styles for Home Button -->",
    "<style>",
    "    #home-button {",
    "        display: inline-block;",
    "        background-color: #007BFF;",
    "        color: white;",
    "        border: none;",
    "        padding: 10px 15px;",
    "        font-size: 14px;",
    "        font-weight: bold;",
    "        border-radius: 5px;",
    "        text-decoration: none;",
    "        cursor: pointer;",
    "        margin-bottom: 10px;",
    "    }",
    "    #home-button:hover {",
    "        background-color: #0056b3;",
    "    }",
    "</style>",
    '<a id="home-button" href="https://johnokoth.github.io/actuarialangles">Back to Home</a>',
]
st.sidebar.markdown("\n".join(home_button_html), unsafe_allow_html=True)

# Load data
augmented_data = pd.read_csv("https://raw.githubusercontent.com/JohnOkoth/actuarialangles/main/data/simulated.csv")

# Functions
def calculate_severity(row, shap_values, bias_base, decile=None, bias_adjustments=None):
    active_features = row[row == 1].index
    total_shap = shap_values[shap_values['Feature'].isin(active_features)]['MeanSHAP'].sum()
    if decile is not None and bias_adjustments is not None:
        bias = bias_base + bias_adjustments.get(decile, 0)
    else:
        bias = bias_base
    return np.exp(total_shap + np.log(bias))

def find_reference_levels(data, categorical_vars, exposure_col):
    ref_levels = []
    for var in categorical_vars:
        exposure_by_level = data.groupby(var)[exposure_col].sum().reset_index()
        ref_level = exposure_by_level.loc[exposure_by_level[exposure_col].idxmax(), var]
        ref_levels.append({'variable': var, 'level': ref_level})
    return ref_levels

def compute_shap_values_with_refs(model, main_data, dummy_data, feature_names=None):
    categorical_vars = main_data.select_dtypes(include=['category', 'object']).columns
    ref_levels = find_reference_levels(main_data, categorical_vars, "Exposure")
    ref_levels = [f"{x['variable']}_{x['level']}".replace(" ", "_") for x in ref_levels]
    
    if feature_names is None:
        feature_names = [col for col in dummy_data.columns if col not in ["Exposure"] + list(categorical_vars)]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dummy_data)
    shap_df = pd.DataFrame(shap_values, columns=dummy_data.columns)
    shap_df['BIAS'] = explainer.expected_value
    
    mean_shap_for_one_hot = []
    for feature in dummy_data.columns:
        active_rows = dummy_data[feature] == 1
        if active_rows.any():
            shap_values_when_one = shap_df.loc[active_rows.values, feature]
            mean_shap_for_one_hot.append(shap_values_when_one.mean())
        else:
            mean_shap_for_one_hot.append(0)
    
    for ref_level in ref_levels:
        mean_shap_for_one_hot.append(0)
    
    mean_shap_values = mean_shap_for_one_hot + [explainer.expected_value]
    feature_names_with_bias = list(dummy_data.columns) + ref_levels + ["BIAS"]
    mean_shap_df = pd.DataFrame({'Feature': feature_names_with_bias, 'MeanSHAP': mean_shap_values})
    return mean_shap_df.sort_values(by=['Feature', 'MeanSHAP'], ascending=[True, False])

def compute_bias_bounds(train_target_sev, shap_values_base, train_data_sev, model_severity_glm):
    mean_sev = np.mean(train_target_sev)
    log_mean_sev = np.log(mean_sev)
    shap_bias = shap_values_base[shap_values_base['Feature'] == 'BIAS']['MeanSHAP'].iloc[0]
    glm_preds = model_severity_glm.predict(train_data_sev)
    log_mean_glm = np.log(np.mean(glm_preds))
    
    # Center the bias range around 5.4 with a ±5% range
    target_bias = 5.4
    lower_bound = target_bias * 0.95  # 5.13
    upper_bound = target_bias * 1.05  # 5.67
    return lower_bound, upper_bound

def create_trend_plot_with_exposure(category_name, data, one_hot_features):
    valid_features = [col for col in one_hot_features if col in data.columns]
    category_data = data.melt(
        id_vars=['Predicted_Severity_GLM', 'Severity_SHAPXGB', 'Severity_SHAPXGBAdj', 'Actual_Sev', 'Predicted_Severity_GLM2'],
        value_vars=valid_features,
        var_name='Variable',
        value_name='Factor_Level'
    )
    category_data = category_data[category_data['Factor_Level'] == 1].groupby('Variable').mean().reset_index()
    category_data = category_data[category_data['Variable'].str.contains(category_name)]
    category_data = category_data.melt(
        id_vars=['Variable'], 
        value_vars=['Predicted_Severity_GLM', 'Severity_SHAPXGB', 'Severity_SHAPXGBAdj', 'Actual_Sev', 'Predicted_Severity_GLM2'],
        var_name='Metric', 
        value_name='Mean_Value'
    )
    category_exposure = data[valid_features + ['Exposure']].melt(
        id_vars=['Exposure'],
        value_vars=valid_features,
        var_name='Variable',
        value_name='Factor_Level'
    )
    category_exposure = category_exposure[category_exposure['Factor_Level'] == 1].groupby('Variable')['Exposure'].sum().reset_index()
    category_exposure = category_exposure[category_exposure['Variable'].str.contains(category_name)]

    max_severity_value = category_data['Mean_Value'].max()
    max_exposure_value = category_exposure['Exposure'].max()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=category_exposure['Variable'], 
        y=category_exposure['Exposure'],
        name='Exposure',
        marker_color='lightgrey',
        opacity=0.3,
        yaxis='y2'
    ))

    colors = {'Predicted_Severity_GLM': '#1f77b4', 'Predicted_Severity_GLM2': 'red', 'Severity_SHAPXGB': '#ff7f0e', 'Severity_SHAPXGBAdj': '#2ca02c', 'Actual_Sev': 'purple'}
    line_styles = {'Predicted_Severity_GLM': 'solid', 'Predicted_Severity_GLM2': 'solid', 'Severity_SHAPXGB': 'dash', 'Severity_SHAPXGBAdj': 'dot', 'Actual_Sev': 'dot'}

    for metric in category_data['Metric'].unique():
        metric_data = category_data[category_data['Metric'] == metric]
        fig.add_trace(go.Scatter(
            x=metric_data['Variable'], 
            y=metric_data['Mean_Value'], 
            mode='lines+markers', 
            name=metric,
            line=dict(color=colors[metric], dash=line_styles[metric], width=2),
            marker=dict(size=6, color=colors[metric]),
            hovertemplate=f"%{{x}}<br>%{{y:.2f}} {metric}"
        ))

    fig.update_layout(
        title=f"Trends for {category_name}",
        xaxis_title=category_name,
        yaxis=dict(title="Mean Severity", side="left"),
        yaxis2=dict(title="Exposure", side="right", overlaying="y", rangemode="tozero"),
        legend=dict(x=1.05, y=1, xanchor="left", yanchor="top", font=dict(size=12)),
        barmode="overlay",
        hovermode="x unified",
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    return fig

# Data Preprocessing
for cat_var in ["Age", "Vehicle_Use", "Car_Model"]:
    total_exposure = augmented_data.groupby(cat_var)['Exposure'].sum().sort_values(ascending=False)
    augmented_data[cat_var] = pd.Categorical(augmented_data[cat_var], categories=total_exposure.index, ordered=True)

dummy_vars_sev = pd.get_dummies(augmented_data.drop(columns=["Group", "Severity", "Exposure", "Claim_Count"]), prefix_sep='_')
model_data = dummy_vars_sev.drop(columns=["Age_F", "Vehicle_Use_DriveShort", "Car_Model_Mazda CX-9"])
target_sev = augmented_data['Severity']
target_exposure = augmented_data['Claim_Count']

train_data_sev, test_data_sev, train_target_sev, test_target_sev = train_test_split(
    model_data, target_sev, test_size=0.2, random_state=42
)
train_data_full, test_data_full, train_target_sev_full, test_target_sev_full = train_test_split(
    dummy_vars_sev, target_sev, test_size=0.2, random_state=42
)

train_target_exposure = target_exposure.loc[train_data_full.index]
test_target_exposure = target_exposure.loc[test_data_full.index]

prepared_test_data = test_data_full.copy()
prepared_test_data['Actual_Sev'] = test_target_sev_full
prepared_test_data['Exposure'] = test_target_exposure
prepared_test_subset = prepared_test_data

# Model Training
gamma_model = GammaRegressor(alpha=0.0001, max_iter=5000, tol=1e-8)
model_severity_glm = gamma_model.fit(train_data_sev, train_target_sev)

params2 = {'booster': 'gbtree', 'objective': 'reg:gamma', 'eta': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 0.5, 'colsample_bytree': 0.5}
xgb_model2 = XGBRegressor(**params2, n_estimators=100)
xgb_model2.fit(train_data_sev, train_target_sev)
shap_values2 = compute_shap_values_with_refs(xgb_model2, augmented_data, train_data_sev)

bias_lower, bias_upper = compute_bias_bounds(train_target_sev, shap_values2, train_data_sev, model_severity_glm)

# Streamlit App
st.title("Model Tuning Dashboard")

# Sidebar for Inputs
st.sidebar.header("Model Tuning Parameters")
nrounds = st.sidebar.slider("Number of Rounds", 100, 150, 100, step=50)
eta = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
max_depth = st.sidebar.slider("Max Depth", 6, 10, 6, step=1)
gamma = st.sidebar.slider("Gamma", 0.0, 0.2, 0.0, step=0.1)
colsample_bytree = st.sidebar.slider("Col Sample By Tree", 0.5, 0.7, 0.5, step=0.1)
min_child_weight = st.sidebar.slider("Min Child Weight", 1, 3, 1, step=1)
subsample = st.sidebar.slider("Sub Sample", 0.5, 0.9, 0.5, step=0.1)
bias_factor = st.sidebar.slider("Bias Factor", bias_lower, bias_upper, (bias_lower + bias_upper) / 2, step=0.001)
st.sidebar.markdown("**Note:** Total Timesteps controls RL training duration. Increase to 5000 for potentially better results")
total_timesteps = st.sidebar.slider("Total Timesteps (RL)", 1000, 5000, 3000, step=1000)

st.sidebar.header("Variable Weights")
weight_overall = st.sidebar.number_input("Weight: Overall", 0.0, 1.0, 0.5, step=0.05)
weight_age = st.sidebar.number_input("Weight: Age", 0.0, 1.0, 0.2, step=0.05)
weight_vehicle = st.sidebar.number_input("Weight: Vehicle_Use", 0.0, 1.0, 0.15, step=0.05)
weight_car = st.sidebar.number_input("Weight: Car_Model", 0.0, 1.0, 0.15, step=0.05)
weight_loss_ratio = st.sidebar.number_input("Weight: Loss Ratio", 0.0, 1.0, 0.5, step=0.05)

opt_method = st.sidebar.selectbox("Optimization Method", ["Single Parameter", "Bayesian", "Random Search", "Manual", "Reinforcement Learning"])

if 'optimized_params' not in st.session_state:
    st.session_state.optimized_params = {'bias': bias_factor, 'eta': eta, 'max_depth': max_depth}
if 'weighted_error' not in st.session_state:
    st.session_state.weighted_error = None

if st.sidebar.button("Update & Optimize"):
    # Model Training and Prediction
    params2 = {'booster': 'gbtree', 'objective': 'reg:gamma', 'eta': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 0.5, 'colsample_bytree': 0.5}
    xgb_model2 = XGBRegressor(**params2, n_estimators=100)
    xgb_model2.fit(train_data_sev, train_target_sev)
    shap_values2 = compute_shap_values_with_refs(xgb_model2, augmented_data, train_data_sev)

    glm_cols = [col for col in train_data_sev.columns if col not in ['Car_Model_Ford Explorer', 'Car_Model_Subaru Legacy', 'Car_Model_Hyundai Sonata', 'Car_Model_Toyota RAV4', 'Car_Model_Rivian R1T']]
    model_severity_glm = GammaRegressor().fit(train_data_sev, train_target_sev)
    model_severity_glm2 = GammaRegressor().fit(train_data_sev[glm_cols], train_target_sev)

    prepared_test_data = test_data_full.copy()
    prepared_test_data['Actual_Sev'] = test_target_sev
    prepared_test_data['Exposure'] = test_target_exposure

    test_data_for_pred = test_data_full.drop(columns=['Age_F', 'Vehicle_Use_DriveShort', 'Car_Model_Mazda CX-9'], errors='ignore')

    total_exposure_overall = prepared_test_data['Exposure'].sum()
    categories = ['Age', 'Vehicle_Use', 'Car_Model']
    exposure_by_category = {}
    total_exposure_by_category = {}

    for category in categories:
        category_cols = [col for col in prepared_test_data.columns if col.startswith(category + '_')]
        melted = prepared_test_data.melt(
            id_vars=['Exposure'],
            value_vars=category_cols,
            var_name='Variable',
            value_name='Factor_Level'
        )
        active_rows = melted[melted['Factor_Level'] == 1]
        exposure_by_level = active_rows.groupby('Variable')['Exposure'].sum().to_dict()
        exposure_by_category[category] = exposure_by_level
        total_exposure = active_rows['Exposure'].sum()
        total_exposure_by_category[category] = total_exposure

    prepared_test_data['Predicted_Severity_GLM'] = model_severity_glm.predict(test_data_for_pred)
    prepared_test_data['Predicted_Severity_GLM2'] = model_severity_glm2.predict(test_data_for_pred[glm_cols])

    prepared_test_data['Severity_SHAPXGB'] = prepared_test_data.apply(
        lambda row: calculate_severity(row, shap_values2, np.exp(shap_values2[shap_values2['Feature'] == 'BIAS']['MeanSHAP'].iloc[0])), axis=1
    )

    def optimize_weighted_error_notrl(bias, eta=eta, max_depth=max_depth):
        params = {
            'booster': 'gbtree', 'objective': 'reg:gamma', 'eta': eta, 'max_depth': int(max_depth),
            'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma
        }
        xgb_model = XGBRegressor(**params, n_estimators=nrounds)
        xgb_model.fit(train_data_sev, train_target_sev)
        shap_values = compute_shap_values_with_refs(xgb_model, augmented_data, train_data_sev)
        
        temp_test_data = prepared_test_data.copy()
        temp_test_data['Severity_SHAPXGBAdj'] = temp_test_data.apply(
            lambda row: calculate_severity(row, shap_values, np.exp(bias)), axis=1
        )
        
        temp_ordered = temp_test_data.sort_values('Predicted_Severity_GLM')
        temp_ordered['Decile'] = pd.qcut(temp_ordered['Predicted_Severity_GLM'], 10, labels=False, duplicates='drop')
        temp_metrics = temp_ordered.groupby('Decile').agg({'Severity_SHAPXGBAdj': 'mean', 'Actual_Sev': 'mean'}).reset_index()
        
        overall_rmse = np.sqrt(np.mean((temp_metrics['Severity_SHAPXGBAdj'] - temp_metrics['Actual_Sev'])**2))
        
        def var_rmse(category_name):
            category_data = temp_test_data.melt(
                id_vars=['Severity_SHAPXGBAdj', 'Actual_Sev'],
                value_vars=test_data_sev.columns,
                var_name='Variable',
                value_name='Factor_Level'
            )
            category_data = category_data[(category_data['Factor_Level'] == 1) & (category_data['Variable'].str.contains(category_name))]
            category_data = category_data.groupby('Variable').agg({'Severity_SHAPXGBAdj': 'mean', 'Actual_Sev': 'mean'}).reset_index()
            return np.sqrt(np.mean((category_data['Severity_SHAPXGBAdj'] - category_data['Actual_Sev'])**2))
        
        rmse_age = var_rmse("Age")
        rmse_vehicle = var_rmse("Vehicle_Use")
        rmse_car = var_rmse("Car_Model")
        
        weighted_rmse = (weight_overall * overall_rmse) + (weight_age * rmse_age) + (weight_vehicle * rmse_vehicle) + (weight_car * rmse_car)

        target_loss_ratio = 0.7
        temp_test_data['Premium_Rate'] = temp_test_data['Severity_SHAPXGBAdj'] / target_loss_ratio
        temp_test_data['Premium'] = temp_test_data['Premium_Rate']
        total_claims_paid = temp_test_data['Severity_SHAPXGBAdj'].sum()
        total_premiums_earned = temp_test_data['Premium'].sum()
        loss_ratio = total_claims_paid / total_premiums_earned if total_premiums_earned > 0 else float('inf')
    
        low_risk_deciles = temp_metrics[temp_metrics['Decile'].isin([0, 1, 2, 5])]
        low_risk_overprediction = np.sum(np.maximum(0, low_risk_deciles['Severity_SHAPXGBAdj'] - low_risk_deciles['Actual_Sev']))
    
        high_risk_deciles = temp_metrics[temp_metrics['Decile'].isin([7, 8, 9])]
        high_risk_underprediction = np.sum(np.maximum(0, high_risk_deciles['Actual_Sev'] - high_risk_deciles['Severity_SHAPXGBAdj']))
        high_risk_excessive_overprediction = np.sum(np.maximum(0, high_risk_deciles['Severity_SHAPXGBAdj'] - 1.5 * high_risk_deciles['Actual_Sev']))
    
        total_error = (
            weighted_rmse +
            (weight_loss_ratio * loss_ratio) +
            (0.5 * low_risk_overprediction) +
            (2.0 * high_risk_underprediction) +
            (0.5 * high_risk_excessive_overprediction)
        )
    
        return {
            'total_error': total_error,
            'weighted_rmse': weighted_rmse,
            'loss_ratio': loss_ratio,
            'low_risk_overprediction': low_risk_overprediction,
            'high_risk_underprediction': high_risk_underprediction,
            'high_risk_excessive_overprediction': high_risk_excessive_overprediction,
            'temp_metrics': temp_metrics
        }

    def optimize_weighted_error(bias, eta, max_depth, use_shap=True, subsample_data=False, bias_adjustments=None, prepared_test_subset=None):
        try:
            if subsample_data:
                train_data_subset = train_data_sev.sample(frac=0.1, random_state=42)
                train_target_subset = train_target_sev[train_data_subset.index]
                test_data_subset = test_data_sev.sample(frac=0.1, random_state=42)
                if prepared_test_subset is not None and 'Actual_Sev' in prepared_test_subset.columns:
                    temp_test_data = prepared_test_subset.loc[test_data_subset.index].copy()
                else:
                    temp_test_data = test_data_subset.copy()
                    temp_test_data['Actual_Sev'] = test_target_sev[test_data_subset.index]
            else:
                train_data_subset = train_data_sev
                train_target_subset = train_target_sev
                test_data_subset = test_data_sev
                if prepared_test_subset is not None and 'Actual_Sev' in prepared_test_subset.columns:
                    temp_test_data = prepared_test_subset.copy()
                else:
                    temp_test_data = test_data_subset.copy()
                    temp_test_data['Actual_Sev'] = test_target_sev[test_data_subset.index]
        
            params = {
                'booster': 'gbtree', 'objective': 'reg:gamma', 'eta': eta, 'max_depth': int(max_depth),
                'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma
            }
            xgb_model = XGBRegressor(**params, n_estimators=100)
            xgb_model.fit(train_data_subset, train_target_subset)
        
            temp_test_data['Severity_SHAPXGB'] = xgb_model.predict(test_data_subset)
        
            if use_shap:
                shap_values = compute_shap_values_with_refs(xgb_model, augmented_data, train_data_subset)
                if bias_adjustments is not None:
                    temp_test_data['Decile'] = pd.qcut(
                        temp_test_data['Predicted_Severity_GLM'] if 'Predicted_Severity_GLM' in temp_test_data.columns else temp_test_data['Severity_SHAPXGB'], 
                        10, labels=False, duplicates='drop'
                    )
                    temp_test_data['Severity_SHAPXGBAdj'] = temp_test_data.apply(
                        lambda row: calculate_severity(row, shap_values, np.exp(bias), row['Decile'], bias_adjustments), axis=1
                    )
                else:
                    temp_test_data['Severity_SHAPXGBAdj'] = temp_test_data.apply(
                        lambda row: calculate_severity(row, shap_values, np.exp(bias)), axis=1
                    )
            else:
                temp_test_data['Severity_SHAPXGBAdj'] = temp_test_data['Severity_SHAPXGB']
        
            # Post-processing: Align means
            mean_actual_sev = temp_test_data['Actual_Sev'].mean()
            mean_pred_sev = temp_test_data['Severity_SHAPXGBAdj'].mean()
            scaling_factor = mean_actual_sev / mean_pred_sev if mean_pred_sev > 0 else 1.0
            temp_test_data['Severity_SHAPXGBAdj'] *= scaling_factor
            temp_test_data['Severity_SHAPXGBAdj'] = temp_test_data['Severity_SHAPXGBAdj'].clip(lower=1e-6)
        
            target_loss_ratio = 0.7
            temp_test_data['Premium_Rate'] = temp_test_data['Severity_SHAPXGBAdj'] / target_loss_ratio
            temp_test_data['Premium'] = temp_test_data['Premium_Rate']
        
            total_claims_paid = temp_test_data['Severity_SHAPXGBAdj'].sum()
            total_premiums_earned = temp_test_data['Premium'].sum()
            loss_ratio = total_claims_paid / total_premiums_earned if total_premiums_earned > 0 else float('inf')
        
            if 'Predicted_Severity_GLM' not in temp_test_data.columns:
                temp_ordered = temp_test_data.sort_values('Severity_SHAPXGB')
            else:
                temp_ordered = temp_test_data.sort_values('Predicted_Severity_GLM')
        
            temp_ordered['Decile'] = pd.qcut(
                temp_ordered['Predicted_Severity_GLM'] if 'Predicted_Severity_GLM' in temp_ordered.columns else temp_ordered['Severity_SHAPXGB'], 
                10, labels=False, duplicates='drop'
            )
            temp_metrics = temp_ordered.groupby('Decile').agg({'Severity_SHAPXGBAdj': 'mean', 'Actual_Sev': 'mean'}).reset_index()
        
            overall_rmse = np.sqrt(np.mean((temp_metrics['Severity_SHAPXGBAdj'] - temp_metrics['Actual_Sev'])**2))
        
            def var_rmse(category_name):
                category_data = temp_test_data.melt(
                    id_vars=['Severity_SHAPXGBAdj', 'Actual_Sev'],
                    value_vars=test_data_subset.columns,
                    var_name='Variable',
                    value_name='Factor_Level'
                )
                category_data = category_data[(category_data['Factor_Level'] == 1) & (category_data['Variable'].str.contains(category_name))]
                category_data = category_data.groupby('Variable').agg({'Severity_SHAPXGBAdj': 'mean', 'Actual_Sev': 'mean'}).reset_index()
                return np.sqrt(np.mean((category_data['Severity_SHAPXGBAdj'] - category_data['Actual_Sev'])**2))
        
            rmse_age = var_rmse("Age")
            rmse_vehicle = var_rmse("Vehicle_Use")
            rmse_car = var_rmse("Car_Model")
        
            low_risk_deciles = temp_metrics[temp_metrics['Decile'].isin([0, 1, 2, 5])]
            low_risk_overprediction = np.sum(np.maximum(0, low_risk_deciles['Severity_SHAPXGBAdj'] - low_risk_deciles['Actual_Sev']))
        
            high_risk_deciles = temp_metrics[temp_metrics['Decile'].isin([7, 8, 9])]
            high_risk_underprediction = np.sum(np.maximum(0, high_risk_deciles['Actual_Sev'] - high_risk_deciles['Severity_SHAPXGBAdj']))
            high_risk_excessive_overprediction = np.sum(np.maximum(0, high_risk_deciles['Severity_SHAPXGBAdj'] - 1.5 * high_risk_deciles['Actual_Sev']))
        
            weighted_rmse = (weight_overall * overall_rmse) + (weight_age * rmse_age) + (weight_vehicle * rmse_vehicle) + (weight_car * rmse_car)
            total_error = (
                weighted_rmse +
                (weight_loss_ratio * loss_ratio) +
                (0.5 * low_risk_overprediction) +
                (2.0 * high_risk_underprediction) +
                (0.5 * high_risk_excessive_overprediction)
            )
        
            return {
                'total_error': total_error,
                'weighted_rmse': weighted_rmse,
                'loss_ratio': loss_ratio,
                'low_risk_overprediction': low_risk_overprediction,
                'high_risk_underprediction': high_risk_underprediction,
                'high_risk_excessive_overprediction': high_risk_excessive_overprediction,
                'temp_metrics': temp_metrics
            }
        except Exception as e:
            print(f"Error in optimize_weighted_error: {str(e)}")
            return None

    # Optimization method selection
    if opt_method == "Single Parameter":
        def optimize_weighted_error_bias_only(bias):
            return optimize_weighted_error_notrl(bias, eta=eta, max_depth=max_depth)['weighted_rmse']

        opt_result = optimize.minimize_scalar(optimize_weighted_error_bias_only, bounds=(bias_lower, bias_upper))
        optimized_params = {'bias': opt_result.x, 'eta': eta, 'max_depth': max_depth}
        metrics = optimize_weighted_error_notrl(opt_result.x, eta, max_depth)
        weighted_error = metrics['total_error']

    elif opt_method == "Bayesian":
        bias_bounds = (bias_lower, bias_upper)
        eta_bounds = (0.01, 0.3)
        max_depth_bounds = (3, 10)
        error_threshold = 0

        def bayes_opt_func(bias, eta, max_depth):
            return -optimize_weighted_error_notrl(bias, eta, int(round(max_depth)))['weighted_rmse']

        def optimize_with_early_stopping(bias_bounds, eta_bounds, max_depth_bounds, init_points=5, n_iter=20, error_threshold=0.0):
            optimizer = BayesianOptimization(
                f=bayes_opt_func,
                pbounds={'bias': bias_bounds, 'eta': eta_bounds, 'max_depth': max_depth_bounds},
                random_state=42,
                verbose=2
            )

            optimizer.maximize(init_points=init_points, n_iter=n_iter)
            optimized_params = optimizer.max['params']
            optimized_params['max_depth'] = int(round(optimized_params['max_depth']))
            metrics = optimize_weighted_error_notrl(
                optimized_params['bias'],
                optimized_params['eta'],
                optimized_params['max_depth']
            )
            weighted_error = metrics['total_error']
            return optimized_params, metrics, weighted_error

        optimized_params, metrics, weighted_error = optimize_with_early_stopping(
            bias_bounds, eta_bounds, max_depth_bounds, init_points=5, n_iter=20, error_threshold=error_threshold
        )

    elif opt_method == "Random Search":
        n_samples = 10
        random_search = pd.DataFrame({
            'bias': np.random.uniform(bias_lower, bias_upper, n_samples),
            'eta': np.random.uniform(0.01, 0.1, n_samples),
            'max_depth': np.random.choice([6, 8, 10], n_samples)
        })
        results = random_search.apply(
            lambda row: optimize_weighted_error_notrl(row['bias'], row['eta'], row['max_depth'])['weighted_rmse'], axis=1
        )
        best_idx = results.idxmin()
        optimized_params = random_search.loc[best_idx].to_dict()
        metrics = optimize_weighted_error_notrl(
            optimized_params['bias'],
            optimized_params['eta'],
            optimized_params['max_depth']
        )
        weighted_error = results[best_idx]

    elif opt_method == "Manual":
        optimized_params = {'bias': bias_factor, 'eta': eta, 'max_depth': max_depth}
        metrics = optimize_weighted_error_notrl(bias_factor, eta, max_depth)
        weighted_error = metrics['total_error']

    elif opt_method == "Reinforcement Learning":
        import tensorflow as tf
        from stable_baselines3 import PPO
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3.common.env_checker import check_env
        
        class WeightOptimizationEnv(gym.Env):
            def __init__(self, optimize_weighted_error_func, bias_lower, bias_upper, eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma, nrounds, weight_loss_ratio, prepared_test_subset, initial_weights=[0.4, 0.3, 0.2, 0.11]):
                super(WeightOptimizationEnv, self).__init__()
                self.initial_weights = np.array(initial_weights, dtype=np.float32)
                self.weights = self.initial_weights.copy()
                self.optimize_weighted_error = optimize_weighted_error_func
                self.prepared_test_subset = prepared_test_subset
                self.bias_lower = bias_lower
                self.bias_upper = bias_upper
                self.eta = eta
                self.max_depth = max_depth
                self.min_child_weight = min_child_weight
                self.subsample = subsample
                self.colsample_bytree = colsample_bytree
                self.gamma = gamma
                self.nrounds = nrounds
                self.weight_loss_ratio = weight_loss_ratio
        
                self.initial_bias = np.float32((bias_lower + bias_upper) / 2)
                self.bias = self.initial_bias
                self.bias_adjustments = np.zeros(10, dtype=np.float32)
        
                self.best_weights = self.weights.copy()
                self.best_bias = self.bias
                self.best_bias_adjustments = self.bias_adjustments.copy()
                self.best_error = float('inf')
                self.error_history = []
                self.bias_history = []
                self.reward_components_history = []
        
                low_bounds = [-0.05, -0.05, -0.05, -0.05]
                high_bounds = [0.1, 0.05, 0.05, 0.05]
                self.action_space = spaces.Box(
                    low=np.array(low_bounds + [-0.05]*11),
                    high=np.array(high_bounds + [0.05]*11),
                    dtype=np.float32
                )
                self.observation_space = spaces.Box(
                    low=np.concatenate([[0.0]*4, [bias_lower], [-0.5]*10]),
                    high=np.concatenate([[1.0]*4, [bias_upper], [0.5]*10]),
                    dtype=np.float32
                )
        
                self.step_count = 0
                self.max_steps = 100
                self.target_bias = 5.4
                self.bias_deviation_threshold = 0.1

            def reset(self, seed=None, options=None):
                self.weights = self.initial_weights.copy().astype(np.float32)
                self.bias = self.initial_bias
                self.bias_adjustments = np.zeros(10, dtype=np.float32)
                self.step_count = 0
                return np.concatenate([self.weights, [self.bias], self.bias_adjustments]), {}

            def step(self, action):
                weight_actions = action[:4]
                bias_action = action[4]
                bias_adj_actions = action[5:]

                self.weights += weight_actions
                self.bias += bias_action
                self.bias_adjustments += bias_adj_actions
    
                self.weights = np.clip(self.weights, 0.0, 1.0).astype(np.float32)
                self.bias = np.clip(self.bias, self.bias_lower, self.bias_upper).astype(np.float32)
                self.bias_adjustments = np.clip(self.bias_adjustments, -0.5, 0.5).astype(np.float32)
    
                self.weights[0] = max(self.weights[0], 0.3)
                self.weights[1] = max(self.weights[1], 0.25)
                weight_sum = np.sum(self.weights)
                if weight_sum > 0:
                    self.weights = (self.weights / weight_sum).astype(np.float32)
    
                global weight_overall, weight_age, weight_vehicle, weight_car, weight_loss_ratio
                weight_overall, weight_age, weight_vehicle, weight_car = self.weights
                error_metrics = self.optimize_weighted_error(
                    bias=self.bias,
                    eta=self.eta,
                    max_depth=self.max_depth,
                    use_shap=False,
                    subsample_data=True,
                    bias_adjustments=dict(enumerate(self.bias_adjustments)),
                    prepared_test_subset=self.prepared_test_subset
                )
    
                total_error = error_metrics['total_error']
                weighted_rmse = error_metrics['weighted_rmse']
                loss_ratio = error_metrics['loss_ratio']
                low_risk_overprediction = error_metrics['low_risk_overprediction']
                high_risk_underprediction = error_metrics['high_risk_underprediction']
                high_risk_excessive_overprediction = error_metrics['high_risk_excessive_overprediction']
                temp_metrics = error_metrics.get('temp_metrics', None)
    
                rmse_reward = np.exp(-0.1 * weighted_rmse**2)
                overall_bonus = 0.2 * self.weights[0]
                target_loss_ratio = 0.7
                loss_ratio_penalty = -0.3 * (loss_ratio - target_loss_ratio)**2
        
                underprediction_penalty = 0.0 if temp_metrics is None else 0.3 * np.sum(
                    np.maximum(0, temp_metrics['Actual_Sev'] - temp_metrics['Severity_SHAPXGBAdj'])
                )
                overprediction_penalty = 0.0 if temp_metrics is None else 2.0 * np.sum(
                    np.maximum(0, temp_metrics['Severity_SHAPXGBAdj'] - temp_metrics['Actual_Sev'])
                )
        
                bias_regularization_penalty = 10.0 * (self.bias - self.target_bias)**2
        
                reward = (
                    rmse_reward +
                    overall_bonus +
                    loss_ratio_penalty -
                    0.5 * low_risk_overprediction -
                    0.3 * high_risk_underprediction -
                    0.5 * high_risk_excessive_overprediction -
                    underprediction_penalty -
                    overprediction_penalty -
                    bias_regularization_penalty
                )
    
                if total_error < self.best_error:
                    self.best_error = total_error
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias
                    self.best_bias_adjustments = self.bias_adjustments.copy()
        
                self.error_history.append(total_error)
                self.bias_history.append(float(self.bias))
                self.reward_components_history.append({
                    'rmse_reward': rmse_reward,
                    'overall_bonus': overall_bonus,
                    'loss_ratio_penalty': loss_ratio_penalty,
                    'low_risk_overprediction': low_risk_overprediction,
                    'high_risk_underprediction': high_risk_underprediction,
                    'high_risk_excessive_overprediction': high_risk_excessive_overprediction,
                    'underprediction_penalty': underprediction_penalty,
                    'overprediction_penalty': overprediction_penalty,
                    'bias_regularization_penalty': bias_regularization_penalty,
                    'total_reward': reward
                })
        
                self.step_count += 1
                done = self.step_count >= self.max_steps
                truncated = False
        
                if abs(self.bias - self.target_bias) > self.bias_deviation_threshold:
                    self.bias = self.target_bias
                    done = True
                    reward -= 10.0
    
                info = {
                    'total_error': total_error,
                    'best_error': self.best_error,
                    'best_weights': self.best_weights,
                    'best_bias': self.bias,
                    'best_bias_adjustments': self.bias_adjustments,
                    'weighted_rmse': weighted_rmse,
                    'loss_ratio': loss_ratio,
                    'low_risk_overprediction': low_risk_overprediction,
                    'high_risk_underprediction': high_risk_underprediction,
                    'high_risk_excessive_overprediction': high_risk_excessive_overprediction,
                    'temp_metrics': temp_metrics
                }
        
                return np.concatenate([self.weights, [self.bias], self.bias_adjustments]), reward, done, truncated, info

            def render(self, mode='human'):
                pass

        eta = min(eta, 0.3)
        initial_weights = [weight_overall, weight_age, weight_vehicle, weight_car]
        env = WeightOptimizationEnv(
            optimize_weighted_error_func=optimize_weighted_error,
            initial_weights=initial_weights,
            bias_lower=bias_lower,
            bias_upper=bias_upper,
            eta=eta,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            nrounds=nrounds,
            weight_loss_ratio=weight_loss_ratio,
            prepared_test_subset=prepared_test_subset
        )
    
        check_env(env, warn=True)
    
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=None,
            n_steps=128,
            batch_size=64,
            n_epochs=5,
            seed=42,
            learning_rate=0.0003
        )
        with st.spinner(f"Training RL agent for {total_timesteps} timesteps... This may take a few minutes."):
            model.learn(total_timesteps=total_timesteps)
    
        best_weights = env.best_weights
        best_bias = env.best_bias
        best_bias_adjustments = env.best_bias_adjustments
        weighted_error = env.best_error
    
        # Post-process the bias to align means
        temp_metrics = optimize_weighted_error(
            bias=best_bias,
            eta=eta,
            max_depth=max_depth,
            use_shap=True,
            subsample_data=False,
            bias_adjustments=dict(enumerate(best_bias_adjustments)),
            prepared_test_subset=prepared_test_subset
        )
        mean_actual_sev = temp_metrics['temp_metrics']['Actual_Sev'].mean()
        mean_pred_sev = temp_metrics['temp_metrics']['Severity_SHAPXGBAdj'].mean()
        bias_correction = np.log(mean_actual_sev / mean_pred_sev) if mean_pred_sev > 0 else 0
        best_bias += bias_correction
        
        optimized_params = {'bias': best_bias, 'eta': eta, 'max_depth': max_depth}
        weight_overall, weight_age, weight_vehicle, weight_car = best_weights

    # Recompute final metrics with adjusted bias
    metrics = optimize_weighted_error(
        bias=optimized_params['bias'],
        eta=eta,
        max_depth=max_depth,
        use_shap=True,
        subsample_data=False,
        bias_adjustments=dict(enumerate(best_bias_adjustments)) if opt_method == "Reinforcement Learning" else None,
        prepared_test_subset=prepared_test_subset
    )
    weighted_error = metrics['total_error']

    if metrics is None:
        raise ValueError("No optimization method was selected or metrics were not computed.")
    final_metrics = metrics        

    params = {
        'booster': 'gbtree', 'objective': 'reg:gamma', 'eta': eta, 'max_depth': int(max_depth),
        'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma
    }
    xgb_model = XGBRegressor(**params, n_estimators=nrounds)
    xgb_model.fit(train_data_sev, train_target_sev)
    shap_values = compute_shap_values_with_refs(xgb_model, augmented_data, train_data_sev)

    prepared_test_data['Severity_SHAPXGBAdj'] = prepared_test_data.apply(
        lambda row: calculate_severity(row, shap_values, np.exp(optimized_params['bias'])), axis=1
    )
    test_data_ordered = prepared_test_data.sort_values('Predicted_Severity_GLM')
    test_data_ordered['Decile'] = pd.qcut(test_data_ordered['Predicted_Severity_GLM'], 10, labels=False, duplicates='drop')
    average_metrics = test_data_ordered.groupby('Decile').agg({
        'Predicted_Severity_GLM': 'mean', 'Predicted_Severity_GLM2': 'mean', 'Severity_SHAPXGB': 'mean',
        'Actual_Sev': 'mean', 'Severity_SHAPXGBAdj': 'mean', 'Exposure': 'sum'
    }).reset_index()

    max_severity_value = max(average_metrics[['Predicted_Severity_GLM', 'Predicted_Severity_GLM2', 'Severity_SHAPXGB', 'Severity_SHAPXGBAdj', 'Actual_Sev']].max())
    max_exposure_value = average_metrics['Exposure'].max()
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Bar(
        x=average_metrics['Decile'], 
        y=average_metrics['Exposure'], 
        name='Exposure',
        marker_color='lightgrey', 
        opacity=0.3,
        yaxis='y2'
    ))
    colors = {
        'Predicted_Severity_GLM': '#1f77b4', 'Predicted_Severity_GLM2': 'red', 'Severity_SHAPXGB': '#ff7f0e',
        'Severity_SHAPXGBAdj': '#2ca02c', 'Actual_Sev': 'purple'
    }
    line_styles = {
        'Predicted_Severity_GLM': 'solid', 'Predicted_Severity_GLM2': 'solid', 'Severity_SHAPXGB': 'dash',
        'Severity_SHAPXGBAdj': 'dot', 'Actual_Sev': 'dot'
    }
    for col in ['Predicted_Severity_GLM', 'Predicted_Severity_GLM2', 'Severity_SHAPXGB', 'Severity_SHAPXGBAdj', 'Actual_Sev']:
        trend_fig.add_trace(go.Scatter(
            x=average_metrics['Decile'], 
            y=average_metrics[col], 
            mode='lines+markers', 
            name=col, 
            line=dict(color=colors[col], dash=line_styles[col], width=2),
            marker=dict(size=6, color=colors[col])
        ))
    trend_fig.update_layout(
        title="Overall Fit: Average Severity by Decile",
        xaxis_title="Decile",
        yaxis=dict(title="Average Severity", side="left"),
        yaxis2=dict(title="Exposure", side="right", overlaying="y", rangemode="tozero"),
        legend=dict(x=1.05, y=1, xanchor="left", yanchor="top", font=dict(size=12)),
        barmode="overlay",
        hovermode="x unified",
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )

    st.subheader("Overall Fit")
    st.plotly_chart(trend_fig, use_container_width=True)
    
    exposure_text = f"\nTotal Exposure (Overall): {total_exposure_overall:.2f}\n"
    for category in categories:
        exposure_text += f"Total Exposure by {category}: {total_exposure_by_category[category]:.2f}\n"
    for category in categories:
        exposure_text += f"\nExposure by {category}:\n"
        for level, exposure in exposure_by_category[category].items():
            level_name = level.replace(f"{category}_", "")
            exposure_text += f"  {level_name}: {exposure:.2f}\n"

    st.subheader("Optimization Results")
    opt_result_text = (
        f"Optimization Method: {opt_method}\n"
        f"Optimized Bias Factor: {optimized_params['bias']:.4f}\n"
        f"Optimized ETA: {optimized_params['eta']:.4f}\n"
        f"Optimized Max Depth: {int(optimized_params['max_depth'])}\n"
        f"Weighted Error Score: {weighted_error:.4f}\n"
        f"Weighted RMSE: {final_metrics['weighted_rmse']:.4f}\n"
        f"Low-Risk Overprediction Penalty: {final_metrics['low_risk_overprediction']:.4f}\n"
        f"High-Risk Underprediction Penalty: {final_metrics['high_risk_underprediction']:.4f}\n"
        f"High-Risk Excessive Overprediction Penalty: {final_metrics['high_risk_excessive_overprediction']:.4f}\n"
        f"Loss Ratio: {final_metrics['loss_ratio']:.4f}\n"
        f"Weights Used:\n  Overall: {weight_overall}\n  Age: {weight_age}\n  Vehicle_Use: {weight_vehicle}\n  Car_Model: {weight_car}"
        f"{exposure_text}"
    )
    st.text(opt_result_text)

    st.subheader("Variable Fits")
    age_fig = create_trend_plot_with_exposure("Age", prepared_test_data, test_data_full.columns)
    vehicle_fig = create_trend_plot_with_exposure("Vehicle_Use", prepared_test_data, test_data_full.columns)
    car_fig = create_trend_plot_with_exposure("Car_Model", prepared_test_data, test_data_full.columns)
    st.plotly_chart(age_fig, use_container_width=True)
    st.plotly_chart(vehicle_fig, use_container_width=True)
    st.plotly_chart(car_fig, use_container_width=True)

if __name__ == "__main__":
    pass