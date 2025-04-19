import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Set Page Configuration ---
st.set_page_config(
    page_title="Ontario Auto Trends 🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- Load the data ---
data_path = r"C:\Users\grace\OneDrive\Documents\data\ontario_combined_data_with_coverage.xlsx"
data = pd.read_excel(data_path)

# --- Data processing ---
if 'Accident Semester' in data.columns:
    data['Accident Semester'] = data['Accident Semester'].astype(str)
    data['Year'] = data['Accident Semester'].str.split(".").str[0].astype(int)
    data['Semester'] = data['Accident Semester'].str.split(".").str[1].astype(int)

if 'Maturity (Months)' not in data.columns:
    st.error("The required column 'Maturity (Months)' is missing from the dataset.")
    st.stop()

data['Maturity (Months)'] = data['Maturity (Months)'].astype(int)

# --- Sidebar Filters ---
st.sidebar.title("Filter Options")

# Multiselect with all coverages selected by default
all_coverages = sorted(data['Coverage'].unique())

selected_coverages = st.sidebar.multiselect(
    "Select Coverages",
    options=all_coverages,
    default=all_coverages
)

# Show number of selected coverages
st.sidebar.success(f"✅ {len(selected_coverages)} coverage(s) selected.")

# Warning if no coverage selected
if not selected_coverages:
    st.sidebar.warning("⚠️ Please select at least one coverage to display the charts.")
    st.stop()

# Time period selection
time_period = st.sidebar.radio("Select Time Period", ['Annual', 'Half Year'])

# Include exposure option
include_exposure = st.sidebar.checkbox("Include Exposure Bar Chart")

# Trend types selection
trend_options = ['Frequency', 'Severity', 'Loss Cost']
selected_trends = st.sidebar.multiselect(
    "Select Trends to Display",
    options=trend_options,
    default=trend_options
)

# --- Slider for time selection ---
if time_period == "Annual":
    min_year, max_year = data["Year"].min(), data["Year"].max()
    range_year = st.sidebar.slider("Select Range (Annual)", min_year, max_year, (min_year, max_year))
    filtered_data = data[(data["Year"] >= range_year[0]) & (data["Year"] <= range_year[1])]
    x_col = "Year"
else:
    min_maturity, max_maturity = data["Maturity (Months)"].min(), data["Maturity (Months)"].max()
    maturity_range = st.sidebar.slider("Select Maturity (Months)", min_maturity, max_maturity, (min_maturity, max_maturity), step=6)
    filtered_data = data[(data["Maturity (Months)"] >= maturity_range[0]) & (data["Maturity (Months)"] <= maturity_range[1])]
    x_col = "Accident Semester"

# --- Main Content ---
st.title("Ontario Auto Coverage Trend Analysis")

for coverage in selected_coverages:
    coverage_data = filtered_data[filtered_data['Coverage'] == coverage]

    if coverage_data.empty:
        st.warning(f"No data available for {coverage} in the selected range.")
        continue

    grouped_data = coverage_data.groupby(x_col).agg({
        'Ultimate Loss Cost': 'mean',
        'Ultimate Severity': 'mean',
        'Ultimate Frequency per 1000': 'mean',
        'Earned Car Years': 'sum'
    }).reset_index()

    with st.expander(f"{coverage} Trends", expanded=False):
        tabs = st.tabs(selected_trends)

        for trend_name, tab in zip(selected_trends, tabs):
            with tab:
                fig = go.Figure()

                if trend_name == 'Frequency':
                    fig.add_trace(go.Scatter(
                        x=grouped_data[x_col], y=grouped_data['Ultimate Frequency per 1000'],
                        mode='lines+markers', name='Frequency', line=dict(color='green')
                    ))
                    yaxis_title = "Frequency"

                elif trend_name == 'Severity':
                    fig.add_trace(go.Scatter(
                        x=grouped_data[x_col], y=grouped_data['Ultimate Severity'],
                        mode='lines+markers', name='Severity', line=dict(color='red')
                    ))
                    yaxis_title = "Severity"

                elif trend_name == 'Loss Cost':
                    fig.add_trace(go.Scatter(
                        x=grouped_data[x_col], y=grouped_data['Ultimate Loss Cost'],
                        mode='lines+markers', name='Loss Cost', line=dict(color='blue')
                    ))
                    yaxis_title = "Loss Cost"

                # Add Exposure if selected
                if include_exposure:
                    fig.add_trace(go.Bar(
                        x=grouped_data[x_col], y=grouped_data['Earned Car Years'] / 1e6,
                        name='Exposure (Millions)', yaxis='y2', marker=dict(color='orange', opacity=0.5)
                    ))

                fig.update_layout(
                    title=f"{coverage} - {trend_name} Trend",
                    xaxis_title=x_col,
                    yaxis_title=yaxis_title,
                    yaxis2=dict(title="Exposure (Millions)", overlaying='y', side='right'),
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
