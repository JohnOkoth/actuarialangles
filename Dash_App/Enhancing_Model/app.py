import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, DMatrix
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import GammaRegressor
from scipy import optimize
import bayes_opt
import random
import shap
from statsmodels.api import GLM, families
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import log

# Set page config as the FIRST Streamlit command
st.set_page_config(
    page_title="Auto Insurance Predictive Model Tuning Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Google Analytics tracking code and custom "Back to Home" button with styles
st.markdown("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-9S5SM84Q3T', { 'anonymize_ip': true });
</script>

# Privacy notice
#st.markdown("**Privacy Notice**: This app uses Google Analytics to track user access for improving user experience. No personal data is collected.")

# Load data with error handling
try:
    augmented_data = pd.read_csv("https://raw.githubusercontent.com/JohnOkoth/actuarialangles/main/data/simulated.csv")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Functions
def calculate_severity(row, shap_values, bias_severity):
    active_features = row[row == 1].index
    total_shap = shap_values[shap_values['Feature'].isin(active_features)]['MeanSHAP'].sum()
    return np.exp(total_shap + np.log(bias_severity))

def find_reference_levels(data, categorical_vars, exposure_col):
    ref_levels = []
    for var in categorical_vars:
        exposure_by_level = data.groupby(var, observed=False)[exposure_col].sum().reset_index()
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
    
    #central_bias = (log_mean_sev + shap_bias + log_mean_glm) / 3
    central_bias = (log_mean_sev + shap_bias) / 2
    lower_bound = min(np.log(train_target_sev))
    upper_bound = central_bias * 1.0
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
        y=category_exposure['Exposure'] / max_exposure_value * max_severity_value,
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
main_data = dummy_vars_sev.drop(columns=[col for col in dummy_vars_sev.columns if any(cat in col for cat in ["Age_", "Vehicle_Use_", "Car_Model_"])])
model_data = dummy_vars_sev.drop(columns=[col for col in dummy_vars_sev.columns if col in ["Age_F", "Vehicle_Use_DriveShort", "Car_Model_Mazda CX-9"]])
target_sev = augmented_data['Severity']
target_exposure = augmented_data['Exposure']

train_data_sev, test_data_sev, train_target_sev, test_target_sev = train_test_split(model_data, target_sev, test_size=0.2, random_state=42)
train_target_exposure = target_exposure.loc[train_data_sev.index]
test_target_exposure = target_exposure.loc[test_data_sev.index]

dtrain_sev = DMatrix(train_data_sev, label=train_target_sev)
dtest_sev = DMatrix(test_data_sev, label=test_target_sev)

print(type(train_target_sev))
print(type(train_data_sev))

gamma_model = GammaRegressor(alpha=0.0001, max_iter=5000, tol=1e-8)
model_severity_glm = gamma_model.fit(train_data_sev, train_target_sev)


params2 = {'booster': 'gbtree', 'objective': 'reg:gamma', 'eta': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 0.5, 'colsample_bytree': 0.5}
xgb_model2 = XGBRegressor(**params2, n_estimators=100)
xgb_model2.fit(train_data_sev, train_target_sev)
shap_values2 = compute_shap_values_with_refs(xgb_model2, augmented_data, train_data_sev)

bias_lower, bias_upper = compute_bias_bounds(train_target_sev, shap_values2, train_data_sev, model_severity_glm)



# Streamlit App
st.title("Auto Insurance Predictive Model Tuning Dashboard")

# Sidebar: Add "Back to Home" button at the top
home_button_html = [
    "<!-- Custom Styles for Home Button -->",
    "<style>",
    "    /* Home Button Styles */",
    "    #home-button {",
    "        display: inline-block;",
    "        background-color: #007BFF; /* Blue background */",
    "        color: white;",
    "        border: none;",
    "        padding: 10px 15px;",
    "        font-size: 14px;",
    "        font-weight: bold;",
    "        border-radius: 5px;",
    "        text-decoration: none;",
    "        cursor: pointer;",
    "        margin-bottom: 10px; /* Add spacing below the button */",
    "        z-index: 1000; /* Ensure it stays on top of other sidebar elements */",
    "    }",
    "",
    "    #home-button:hover {",
    "        background-color: #0056b3; /* Darker blue on hover */",
    "    }",
    "</style>",
    "",
    "<!-- Home Button -->",
    '<a id="home-button" href="https://johnokoth.github.io/actuarialangles">Back to Home</a>',
]
st.sidebar.markdown("\n".join(home_button_html), unsafe_allow_html=True)
<!-- Home Button -->
<a id="home-button" href="https://johnokoth.github.io/actuarialangles">Back to Home</a>
""", unsafe_allow_html=True)

# Sidebar for Inputs
st.sidebar.header("Model Tuning Parameters")
nrounds = st.sidebar.slider("Number of Rounds", 100, 150, 100, step=50)
eta = st.sidebar.slider("Learning Rate", 0.01, 0.1, 0.1, step=0.01)
max_depth = st.sidebar.slider("Max Depth", 6, 10, 6, step=1)
gamma = st.sidebar.slider("Gamma", 0.0, 0.2, 0.0, step=0.1)
colsample_bytree = st.sidebar.slider("Col Sample By Tree", 0.5, 0.7, 0.5, step=0.1)
min_child_weight = st.sidebar.slider("Min Child Weight", 1, 3, 1, step=1)
subsample = st.sidebar.slider("Sub Sample", 0.5, 0.9, 0.5, step=0.1)
bias_factor = st.sidebar.slider("Bias Factor", bias_lower, bias_upper, (bias_lower + bias_upper) / 2, step=0.001)

st.sidebar.header("Variable Weights")
weight_overall = st.sidebar.number_input("Weight: Overall", 0.0, 1.0, 0.5, step=0.05)
weight_age = st.sidebar.number_input("Weight: Age", 0.0, 1.0, 0.2, step=0.05)
weight_vehicle = st.sidebar.number_input("Weight: Vehicle_Use", 0.0, 1.0, 0.15, step=0.05)
weight_car = st.sidebar.number_input("Weight: Car_Model", 0.0, 1.0, 0.15, step=0.05)

opt_method = st.sidebar.selectbox("Optimization Method", ["Single Parameter", "Bayesian", "Random Search", "Manual"])

# At the top of your script, initialize session state
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

    prepared_test_data = test_data_sev.copy()
    prepared_test_data['Predicted_Severity_GLM'] = model_severity_glm.predict(test_data_sev)
    prepared_test_data['Predicted_Severity_GLM2'] = model_severity_glm2.predict(test_data_sev[glm_cols])
    prepared_test_data['Actual_Sev'] = test_target_sev
    prepared_test_data['Exposure'] = test_target_exposure
    prepared_test_data['Severity_SHAPXGB'] = prepared_test_data.apply(lambda row: calculate_severity(row, shap_values2, np.exp(shap_values2[shap_values2['Feature'] == 'BIAS']['MeanSHAP'].iloc[0])), axis=1)

    def optimize_weighted_error(bias, eta=eta, max_depth=max_depth):
        params = {'booster': 'gbtree', 'objective': 'reg:gamma', 'eta': eta, 'max_depth': int(max_depth),
                  'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma}
        xgb_model = XGBRegressor(**params, n_estimators=nrounds)
        xgb_model.fit(train_data_sev, train_target_sev)
        shap_values = compute_shap_values_with_refs(xgb_model, augmented_data, train_data_sev)
        
        temp_test_data = prepared_test_data.copy()
        temp_test_data['Severity_SHAPXGBAdj'] = temp_test_data.apply(lambda row: calculate_severity(row, shap_values, np.exp(bias)), axis=1)
        
        temp_ordered = temp_test_data.sort_values('Predicted_Severity_GLM')
        temp_ordered['Decile'] = pd.qcut(temp_ordered['Predicted_Severity_GLM'], 10, labels=False, duplicates='drop')
        temp_metrics = temp_ordered.groupby('Decile').agg({'Severity_SHAPXGBAdj': 'mean', 'Actual_Sev': 'mean'}).reset_index()
        
        overall_rmse = np.sqrt(np.mean((temp_metrics['Severity_SHAPXGBAdj'] - temp_metrics['Actual_Sev'])**2))
        
        def var_rmse(category_name):
            category_data = temp_test_data.melt(id_vars=['Severity_SHAPXGBAdj', 'Actual_Sev'],
                                                value_vars=test_data_sev.columns,
                                                var_name='Variable',
                                                value_name='Factor_Level')
            category_data = category_data[(category_data['Factor_Level'] == 1) & (category_data['Variable'].str.contains(category_name))]
            category_data = category_data.groupby('Variable').agg({'Severity_SHAPXGBAdj': 'mean', 'Actual_Sev': 'mean'}).reset_index()
            return np.sqrt(np.mean((category_data['Severity_SHAPXGBAdj'] - category_data['Actual_Sev'])**2))
        
        rmse_age = var_rmse("Age")
        rmse_vehicle = var_rmse("Vehicle_Use")
        rmse_car = var_rmse("Car_Model")
        
        return (weight_overall * overall_rmse) + (weight_age * rmse_age) + (weight_vehicle * rmse_vehicle) + (weight_car * rmse_car)

    # Define a wrapper function to fix eta and max_depth
    def optimize_weighted_error_bias_only(bias):
        return optimize_weighted_error(bias, eta=eta, max_depth=max_depth)

    # Optimization method selection
    if opt_method == "Single Parameter":
        opt_result = optimize.minimize_scalar(optimize_weighted_error_bias_only, bounds=(bias_lower, bias_upper))
        optimized_params = {'bias': opt_result.x, 'eta': eta, 'max_depth': max_depth}
        weighted_error = opt_result.fun
    elif opt_method == "Bayesian":
        def bayes_opt_func(bias, eta, max_depth):
            return optimize_weighted_error(bias, eta, int(round(max_depth)))
        #optimizer = bayes_opt.BayesianOptimization(f=bayes_opt_func, 
                                                   #pbounds={'bias': (bias_lower, bias_upper), 'eta': (0.01, 0.1), 'max_depth': (6, 10)}, 
                                                   #random_state=42,
                                                   #verbose=2
        #)
        # Define the optimization function with early stopping
        def optimize_with_early_stopping(bias_lower, bias_upper, init_points=5, n_iter=20, error_threshold=0.0):

            # Wrap the objective function to return negative error for maximization
            def wrapped_bayes_opt_func(bias, eta, max_depth):
                return -bayes_opt_func(bias, eta, max_depth)

            optimizer = BayesianOptimization(
                f=wrapped_bayes_opt_func,
                pbounds={'bias': (bias_lower, bias_upper), 'eta': (0.01, 0.1), 'max_depth': (6, 10)},
                random_state=42,
                verbose=2  # Print progress
             )

        # Warm start with a good initial guess
            optimizer.probe(
                params={'bias': 5.5, 'eta': 0.05, 'max_depth': 8},
                lazy=True,
                )
       
        # Track the best error value
            best_error = np.inf
            best_params = None
        
        # Run optimization with early stopping
            for _ in range(init_points + n_iter):
                optimizer.maximize(init_points=0, n_iter=1)  # Perform one iteration at a time
                current_error = -optimizer.max['target']  # Convert back to positive error

            # Update the best error
                if current_error < best_error:
                    best_error = current_error
                    best_params = optimizer.max['params']

            # Check if the error is below the threshold
                if best_error <= error_threshold:
                    print(f"Early stopping: Error reached {best_error} (<= {error_threshold})")
                    break
        # Extract results
            optimized_params = best_params
            optimized_params['max_depth'] = int(round(optimized_params['max_depth']))
            weighted_error = best_error # Convert back to positive error

            return optimized_params, weighted_error

        # Run the optimization
        optimized_params, weighted_error = optimize_with_early_stopping(
            bias_lower=bias_lower,
            bias_upper=bias_upper,
            init_points=5,
            n_iter=20,
            error_threshold=0.0  # Stop if error reaches zero
        )

        print("Optimized Parameters:", optimized_params)
        print("Weighted Error:", weighted_error)

        

        #optimizer.maximize(init_points=10, n_iter=20)
        #optimized_params = optimizer.max['params']
        #optimized_params['max_depth'] = int(round(optimized_params['max_depth']))
        #weighted_error = optimizer.max['target']
    elif opt_method == "Random Search":
        n_samples = 10
        random_search = pd.DataFrame({
            'bias': np.random.uniform(bias_lower, bias_upper, n_samples),
            'eta': np.random.uniform(0.01, 0.1, n_samples),
            'max_depth': np.random.choice([6, 8, 10], n_samples)
        })
        results = random_search.apply(lambda row: optimize_weighted_error(row['bias'], row['eta'], row['max_depth']), axis=1)
        best_idx = results.idxmin()
        optimized_params = random_search.loc[best_idx].to_dict()
        weighted_error = results[best_idx]


    elif opt_method == "Manual":
        optimized_params = {'bias': bias_factor, 'eta': eta, 'max_depth': max_depth}
        weighted_error = optimize_weighted_error(bias_factor, eta, max_depth)

    # Final model with optimized parameters
    params = {'booster': 'gbtree', 'objective': 'reg:gamma', 'eta': eta, 'max_depth': int(max_depth),
              'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma}
    xgb_model = XGBRegressor(**params, n_estimators=nrounds)
    xgb_model.fit(train_data_sev, train_target_sev)
    shap_values = compute_shap_values_with_refs(xgb_model, augmented_data, train_data_sev)

    prepared_test_data['Severity_SHAPXGBAdj'] = prepared_test_data.apply(lambda row: calculate_severity(row, shap_values, np.exp(optimized_params['bias'])), axis=1)
    test_data_ordered = prepared_test_data.sort_values('Predicted_Severity_GLM')
    test_data_ordered['Decile'] = pd.qcut(test_data_ordered['Predicted_Severity_GLM'], 10, labels=False, duplicates='drop')
    average_metrics = test_data_ordered.groupby('Decile').agg({
        'Predicted_Severity_GLM': 'mean', 'Predicted_Severity_GLM2': 'mean', 'Severity_SHAPXGB': 'mean',
        'Actual_Sev': 'mean', 'Severity_SHAPXGBAdj': 'mean', 'Exposure': 'sum'
    }).reset_index()

    # Trend Plot
    max_severity_value = max(average_metrics[['Predicted_Severity_GLM', 'Predicted_Severity_GLM2', 'Severity_SHAPXGB', 'Severity_SHAPXGBAdj', 'Actual_Sev']].max())
    max_exposure_value = average_metrics['Exposure'].max()
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Bar(
        x=average_metrics['Decile'], 
        y=average_metrics['Exposure'] / max_exposure_value * max_severity_value, 
        name='Exposure',
        marker_color='lightgrey', 
        opacity=0.3,
        yaxis='y2'
    ))
    colors = {'Predicted_Severity_GLM': '#1f77b4', 'Predicted_Severity_GLM2': 'red', 'Severity_SHAPXGB': '#ff7f0e', 'Severity_SHAPXGBAdj': '#2ca02c', 'Actual_Sev': 'purple'}
    line_styles = {'Predicted_Severity_GLM': 'solid', 'Predicted_Severity_GLM2': 'solid', 'Severity_SHAPXGB': 'dash', 'Severity_SHAPXGBAdj': 'dot', 'Actual_Sev': 'dot'}
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

    # Display Results
    st.subheader("Overall Fit")
    st.plotly_chart(trend_fig, use_container_width=True)

    st.subheader("Optimization Results")
    opt_result_text = (f"Optimization Method: {opt_method}\n"
                       f"Optimized Bias Factor: {optimized_params['bias']:.4f}\n"
                       f"Optimized ETA: {optimized_params['eta']:.4f}\n"
                       f"Optimized Max Depth: {int(optimized_params['max_depth'])}\n"
                       f"Weighted Error Score: {weighted_error:.4f}\n"
                       f"Weights Used:\n  Overall: {weight_overall}\n  Age: {weight_age}\n  Vehicle_Use: {weight_vehicle}\n  Car_Model: {weight_car}")
    st.text(opt_result_text)
    


    st.subheader("Variable Fits")
    age_fig = create_trend_plot_with_exposure("Age", prepared_test_data, test_data_sev.columns)
    vehicle_fig = create_trend_plot_with_exposure("Vehicle_Use", prepared_test_data, test_data_sev.columns)
    car_fig = create_trend_plot_with_exposure("Car_Model", prepared_test_data, test_data_sev.columns)
    st.plotly_chart(age_fig, use_container_width=True)
    st.plotly_chart(vehicle_fig, use_container_width=True)
    st.plotly_chart(car_fig, use_container_width=True)

if __name__ == "__main__":
    pass  # No need for st.run() as Streamlit handles execution when run via `streamlit run script_name.py`