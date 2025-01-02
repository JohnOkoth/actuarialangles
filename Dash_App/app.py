import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Data Preparation
data_auto = pd.DataFrame({
    "Metric": [
        "Combined Ratio", "Claims Ratio", "Core Claim Ratio", "CAT Loss Ratio", "Expense Ratio", 
        "PYD Ratio", "Gross Written Premium", "Underwriting Income", "ROE"
    ] * 4,
    "Value": [
        97.1, 78.4, 71.1, 0, 26.0, -7.5, 1169, 41, 13.5,  # Intact Q1 2023
        100.9, 72.4, 72.9, 0.1, 28.5, -0.6, 357.8, -3.2, 9.3,  # Definity Q1 2023
        98.6, 75.4, 72.2, 0, 26.4, -3.2, 1300, 20, 14.0,  # Intact Q1 2024
        97.1, 70.8, 71.5, 0.1, 26.3, -0.8, 413.5, 10.9, 12.7   # Definity Q1 2024
    ],
    "Company": ["Intact"] * 9 + ["Definity"] * 9 + ["Intact"] * 9 + ["Definity"] * 9,
    "Time": ["Q1 2023"] * 18 + ["Q1 2024"] * 18
})

data_property = pd.DataFrame({
    "Metric": [
        "Combined Ratio", "Claims Ratio", "Core Claim Ratio", "CAT Loss Ratio", "Expense Ratio", 
        "PYD Ratio", "Gross Written Premium", "Underwriting Income", "ROE"
    ] * 4,
    "Value": [
        84.5, 45.0, 50.2, 3.4, 34.3, 1.8, 760, 136, 13.5,  # Intact Q1 2023
        91.1, 54.2, 50.7, 3.8, 36.9, -0.3, 225, 21.5, 9.3,  # Definity Q1 2023
        82.5, 51.0, 46.4, 0, 36.1, -4.6, 828, 166, 14.0,  # Intact Q1 2024
        91.0, 55.2, 51.3, 5.9, 35.8, -2.0, 236, 23.5, 12.7   # Definity Q1 2024
    ],
    "Company": ["Intact"] * 9 + ["Definity"] * 9 + ["Intact"] * 9 + ["Definity"] * 9,
    "Time": ["Q1 2023"] * 18 + ["Q1 2024"] * 18
})

# Dash Layout
app.layout = html.Div([
    html.H1("Insurance Metrics Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id="dataset",
            options=[
                {"label": "Auto", "value": "Auto"},
                {"label": "Property", "value": "Property"}
            ],
            value="Auto"
        ),
        html.Label("Select Metrics:"),
        dcc.Checklist(
            id="metrics",
            options=[{"label": metric, "value": metric} for metric in data_auto["Metric"].unique()],
            value=["Combined Ratio", "Claims Ratio", "Core Claim Ratio"]
        ),
        html.Label("Select Insurers:"),
        dcc.Checklist(
            id="insurers",
            options=[
                {"label": "Intact", "value": "Intact"},
                {"label": "Definity", "value": "Definity"}
            ],
            value=["Intact", "Definity"]
        ),
        html.Label("Select Years:"),
        dcc.Checklist(
            id="years",
            options=[
                {"label": "Q1 2023", "value": "Q1 2023"},
                {"label": "Q1 2024", "value": "Q1 2024"}
            ],
            value=["Q1 2023", "Q1 2024"]
        )
    ], style={"padding": "10px", "border": "1px solid #ddd", "marginBottom": "20px"}),

    dcc.Tabs([
        dcc.Tab(label="Bar Chart", children=[
            dcc.Graph(id="bar-chart")
        ]),
        dcc.Tab(label="Trend Chart", children=[
            html.Div([
                dcc.Graph(id="trend-chart-intact", style={"display": "inline-block", "width": "49%"}),
                dcc.Graph(id="trend-chart-definity", style={"display": "inline-block", "width": "49%"})
            ])
        ])
    ])
])

# Callbacks for Interactivity
@app.callback(
    Output("bar-chart", "figure"),
    Input("dataset", "value"),
    Input("metrics", "value"),
    Input("insurers", "value"),
    Input("years", "value")
)
def update_bar_chart(dataset, selected_metrics, selected_insurers, selected_years):
    data = data_auto if dataset == "Auto" else data_property
    filtered_data = data[
        (data["Metric"].isin(selected_metrics)) &
        (data["Company"].isin(selected_insurers)) &
        (data["Time"].isin(selected_years))
    ]
    fig = px.bar(
        filtered_data,
        x="Value",
        y="Metric",
        color="Time",
        barmode="group",
        facet_col="Company",
        title=f"Bar Chart of Selected Metrics ({dataset} Dataset)"
    )
    return fig

@app.callback(
    [Output("trend-chart-intact", "figure"),
     Output("trend-chart-definity", "figure")],
    Input("dataset", "value"),
    Input("metrics", "value"),
    Input("years", "value")
)
def update_trend_charts(dataset, selected_metrics, selected_years):
    data = data_auto if dataset == "Auto" else data_property

    # Filter data for Intact
    data_intact = data[
        (data["Company"] == "Intact") &
        (data["Metric"].isin(selected_metrics)) &
        (data["Time"].isin(selected_years))
    ]
    fig_intact = px.line(
        data_intact,
        x="Time",
        y="Value",
        color="Metric",
        markers=True,
        title=f"Trend Chart - Intact ({dataset} Dataset)"
    )

    # Filter data for Definity
    data_definity = data[
        (data["Company"] == "Definity") &
        (data["Metric"].isin(selected_metrics)) &
        (data["Time"].isin(selected_years))
    ]
    fig_definity = px.line(
        data_definity,
        x="Time",
        y="Value",
        color="Metric",
        markers=True,
        title=f"Trend Chart - Definity ({dataset} Dataset)"
    )

    return fig_intact, fig_definity

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
