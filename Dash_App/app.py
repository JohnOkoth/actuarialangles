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

# Dash Layout
app.layout = html.Div([
    html.H1("Insurance Metrics Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id="dataset",
            options=[
                {"label": "Auto", "value": "Auto"}
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
            dcc.Graph(id="trend-chart")
        ])
    ])
])

# Callbacks for Interactivity
@app.callback(
    Output("bar-chart", "figure"),
    Input("metrics", "value"),
    Input("insurers", "value"),
    Input("years", "value")
)
def update_bar_chart(selected_metrics, selected_insurers, selected_years):
    filtered_data = data_auto[
        (data_auto["Metric"].isin(selected_metrics)) &
        (data_auto["Company"].isin(selected_insurers)) &
        (data_auto["Time"].isin(selected_years))
    ]
    fig = px.bar(
        filtered_data,
        x="Value",
        y="Metric",
        color="Time",
        barmode="group",
        facet_col="Company",
        title="Bar Chart of Selected Metrics"
    )
    return fig


@app.callback(
    Output("trend-chart", "figure"),
    Input("metrics", "value"),
    Input("insurers", "value")
)
def update_trend_chart(selected_metrics, selected_insurers):
    filtered_data = data_auto[
        (data_auto["Metric"].isin(selected_metrics)) &
        (data_auto["Company"].isin(selected_insurers))
    ]
    fig = px.line(
        filtered_data,
        x="Time",
        y="Value",
        color="Company",
        line_group="Metric",
        markers=True,
        title="Trend Chart of Selected Metrics"
    )
    return fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
