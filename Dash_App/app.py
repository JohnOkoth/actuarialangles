import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

# --- Set Page Configuration ---
st.set_page_config(
    page_title="Insurance Metrics Dashboard 🚗🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Google Analytics Injection ---
components.html("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-9S5SM84Q3T');
</script>
""", height=0)

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

# --- Load Data with Caching ---
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

#property_file_path = r"C:\Users\grace\OneDrive\Documents\data\Property_Metrics.xlsx"
#auto_file_path = r"C:\Users\grace\OneDrive\Documents\data\Auto_Metrics.xlsx"

property_file_path = "https://raw.githubusercontent.com/JohnOkoth/actuarialangles/main/data/Property_Metrics.xlsx"
auto_file_path = "https://raw.githubusercontent.com/JohnOkoth/actuarialangles/main/data/Auto_Metrics.xlsx"


data_property = load_data(property_file_path)
data_auto = load_data(auto_file_path)

# --- Title ---
st.title("Insurance Metrics Dashboard")

# --- Sidebar - Filter Section ---
data_type = st.radio("Select Data Type:", ["Property", "Auto"])

data = data_property if data_type == "Property" else data_auto

metrics = sorted(data["Metric"].dropna().unique().tolist())
companies = sorted(data["Company"].dropna().unique().tolist())
time_periods = sorted(data["Time"].dropna().unique().tolist(), key=lambda x: (
    ["Q1", "Q2", "Q3", "Q4", "H1", "H2"].index(x.split()[0]),
    int(x.split()[1])
))

# --- Default Selections ---
default_metrics = [metrics[0]] if metrics else []
default_insurers = [ins for ins in ["Intact", "Definity"] if ins in companies]
default_time = [tp for tp in ["Q1 2023", "Q1 2024", "Q2 2023", "Q2 2024", "Q3 2023", "Q3 2024"] if tp in time_periods]

# --- User Selections ---
selected_metrics = st.sidebar.multiselect("Select Metrics:", metrics, default=default_metrics)
selected_insurers = st.sidebar.multiselect("Select Insurers:", companies, default=default_insurers)
selected_time_period = st.sidebar.multiselect("Select Time Period:", time_periods, default=default_time)
selected_tab = st.sidebar.selectbox("Select View:", ["Bar Chart", "Trend Charts by Metric"])

# --- Sidebar Info ---
st.sidebar.success(f"✅ {len(selected_metrics)} metric(s) selected.")
st.sidebar.success(f"✅ {len(selected_insurers)} insurer(s) selected.")
st.sidebar.success(f"✅ {len(selected_time_period)} time period(s) selected.")

# --- Main Content Logic ---
if not selected_metrics or not selected_insurers or not selected_time_period:
    st.warning("⚠️ Please select at least one Metric, Insurer, and Time Period.")
    st.stop()

if selected_tab == "Bar Chart":
    for metric in selected_metrics:
        filtered_data = data[
            (data["Metric"] == metric) &
            (data["Company"].isin(selected_insurers)) &
            (data["Time"].isin(selected_time_period))
        ]
        if filtered_data.empty:
            st.warning(f"No data available for {metric} under current selections.")
            continue
        fig = px.bar(
            filtered_data,
            x="Company",
            y="Value",
            color="Time",
            barmode="group",
            title=f"Bar Chart for {metric}",
            category_orders={"Time": time_periods}
        )
        st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Trend Charts by Metric":
    filtered_data = data[
        (data["Metric"].isin(selected_metrics)) &
        (data["Company"].isin(selected_insurers)) &
        (data["Time"].isin(selected_time_period))
    ]
    for metric in selected_metrics:
        metric_data = filtered_data[filtered_data["Metric"] == metric].copy()
        if metric_data.empty:
            st.warning(f"No data available for {metric} under current selections.")
            continue
        metric_data["Time"] = pd.Categorical(metric_data["Time"], categories=time_periods, ordered=True)
        metric_data = metric_data.sort_values("Time")
        fig = px.line(
            metric_data,
            x="Time",
            y="Value",
            color="Company",
            title=f"Trend Chart for {metric}",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
