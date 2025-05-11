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

# --- Time Period Selection ---
st.sidebar.subheader("Time Period Selection")
time_view = st.sidebar.selectbox("Select Time View:", ["Quarterly", "Half-Yearly", "Full-Year"])

# Extract unique years from Time column
years = sorted(set([t.split()[-1] for t in data["Time"].dropna().unique()]))
selected_years = st.sidebar.multiselect("Select Years:", years, default=years[:3])

# Define available periods based on time view
period_mapping = {
    "Quarterly": ["Q1", "Q2", "Q3", "Q4"],
    "Half-Yearly": ["H1", "H2"],
    "Full-Year": ["FY"]
}
available_periods = period_mapping[time_view]
selected_periods = st.sidebar.multiselect(f"Select {time_view} Periods:", available_periods, default=available_periods)

# Generate selected time periods
selected_time_periods = []
for year in selected_years:
    for period in selected_periods:
        if time_view == "Full-Year":
            selected_time_periods.append(f"{year}")
        else:
            selected_time_periods.append(f"{period} {year}")

# Filter valid time periods that exist in data
selected_time_periods = [tp for tp in selected_time_periods if tp in data["Time"].unique()]

# --- Sort Time Periods ---
period_priority = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "H1": 4, "H2": 5, "FY": 6}

def sort_key(label):
    parts = label.split()
    if len(parts) == 2:
        prefix, year = parts
        year = int(year)
    else:
        prefix = "FY"
        year = int(parts[0])
    return (year, period_priority[prefix])

# --- Default Selections ---
default_metrics = [metrics[2]] if metrics else []
default_insurers = [ins for ins in ["Intact", "Definity"] if ins in companies]

# --- User Selections ---
selected_metrics = st.sidebar.multiselect("Select Metrics:", metrics, default=default_metrics)
selected_insurers = st.sidebar.multiselect("Select Insurers:", companies, default=default_insurers)
selected_tab = st.sidebar.selectbox("Select View:", ["Bar Chart", "Trend Charts by Metric"])

# --- Sidebar Info ---
st.sidebar.success(f"✅ {len(selected_metrics)} metric(s) selected.")
st.sidebar.success(f"✅ {len(selected_insurers)} insurer(s) selected.")
st.sidebar.success(f"✅ {len(selected_time_periods)} time period(s) selected.")

# --- Main Content Logic ---
if not selected_metrics or not selected_insurers or not selected_time_periods:
    st.warning("⚠️ Please select at least one Metric, Insurer, Time View, Year, and Period.")
    st.stop()

# --- Visualization ---
if selected_tab == "Bar Chart":
    for metric in selected_metrics:
        for year in selected_years:
            # Filter time periods for the specific year
            year_time_periods = [tp for tp in selected_time_periods if year in tp]
            if not year_time_periods:
                continue
            filtered_data = data[
                (data["Metric"] == metric) &
                (data["Company"].isin(selected_insurers)) &
                (data["Time"].isin(year_time_periods))
            ]
            if filtered_data.empty:
                st.warning(f"No data available for {metric} in {year} under current selections.")
                continue
            fig = px.bar(
                filtered_data,
                x="Company",
                y="Value",
                color="Time",
                barmode="group",
                title=f"{metric} - {time_view} Comparison for {year}",
                category_orders={"Time": sorted(year_time_periods, key=sort_key)},
                text_auto=".2f"
            )
            fig.update_layout(
                xaxis_title="Insurer",
                yaxis_title=metric,
                legend_title="Time Period"
            )
            st.plotly_chart(fig, use_container_width=True)

elif selected_tab == "Trend Charts by Metric":
    for metric in selected_metrics:
        for year in selected_years:
            # Filter time periods for the specific year
            year_time_periods = [tp for tp in selected_time_periods if year in tp]
            if not year_time_periods:
                continue
            filtered_data = data[
                (data["Metric"] == metric) &
                (data["Company"].isin(selected_insurers)) &
                (data["Time"].isin(year_time_periods))
            ]
            if filtered_data.empty:
                st.warning(f"No data available for {metric} in {year} under current selections.")
                continue
            filtered_data["Time"] = pd.Categorical(filtered_data["Time"], categories=sorted(year_time_periods, key=sort_key), ordered=True)
            filtered_data = filtered_data.sort_values("Time")
            fig = px.line(
                filtered_data,
                x="Time",
                y="Value",
                color="Company",
                title=f"{metric} - {time_view} Trend for {year}",
                markers=True,
                text="Value"
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title=metric,
                legend_title="Insurer"
            )
            st.plotly_chart(fig, use_container_width=True)