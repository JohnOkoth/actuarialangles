import dash 
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# File paths
#property_file_path = r"C:\\Users\\grace\\OneDrive\\Documents\\data\\Property_Metrics.xlsx"
#auto_file_path = r"C:\\Users\\grace\\OneDrive\\Documents\\data\\Auto_Metrics.xlsx"

property_file_path = "https://raw.githubusercontent.com/JohnOkoth/actuarialangles/main/data/Property_Metrics.xlsx"
auto_file_path = "https://raw.githubusercontent.com/JohnOkoth/actuarialangles/main/data/Auto_Metrics.xlsx"


# Load Data
data_property = pd.read_excel(property_file_path)
data_auto = pd.read_excel(auto_file_path)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Inject Google Analytics script
# Inject Google Analytics script and Home Button
app.index_string = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>Insurance Metrics Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-9S5SM84Q3T"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-9S5SM84Q3T');
        </script>
        <style>
            /* Home Button Styles */
            #home-button {
                position: absolute;
                top: 10px;
                left: 10px;
                background-color: #007BFF; /* Blue background */
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                text-decoration: none;
                cursor: pointer;
                z-index: 1000; /* Ensure it stays on top */
            }

            #home-button:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }
        </style>
    </head>
    <body>
        <!-- Home Button -->
        <a id="home-button" href="https://johnokoth.github.io/actuarialangles">Back to Home</a>

        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# Layout
app.layout = html.Div([
    html.H1("Insurance Metrics Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Data Type:"),
        dcc.RadioItems(
            id="data-type",
            options=[
                {"label": "Property", "value": "Property"},
                {"label": "Auto", "value": "Auto"}
            ],
            value="Auto",  # Default selection
            inline=True
        ),
        html.Label("Select Metrics:"),
        dcc.Checklist(
            id="metrics",
            options=[],
            value=[],
            inline=True
        ),
        html.Label("Select Insurers:"),
        dcc.Checklist(
            id="insurers",
            options=[],
            value=[],
            inline=True
        ),
        html.Label("Select Time Period:"),
        dcc.Checklist(
            id="time-period",
            options=[],
            value=[]
        )
    ], style={"padding": "10px", "border": "1px solid #ddd", "marginBottom": "20px"}),

    dcc.Tabs(
        id="tabs",
        value="bar",
        children=[
            dcc.Tab(label="Bar Chart", value="bar", style={"textAlign": "center"}, selected_style={"backgroundColor": "#007BFF", "color": "white", "fontWeight": "bold"}),
            dcc.Tab(label="Trend Charts by Metric", value="trend", style={"textAlign": "center"}, selected_style={"backgroundColor": "#007BFF", "color": "white", "fontWeight": "bold"})
        ]
    ),
    html.Div(id="tabs-content")
])

# Callbacks to update filters and content
@app.callback(
    [
        Output("metrics", "options"),
        Output("metrics", "value"),
        Output("insurers", "options"),
        Output("insurers", "value"),
        Output("time-period", "options"),
        Output("time-period", "value")
    ],
    [Input("data-type", "value")]
)
def update_filters(data_type):
    # Select data based on type
    if data_type == "Property":
        data = data_property
    else:
        data = data_auto

    # Update filter options
    metrics = [{"label": metric, "value": metric} for metric in data["Metric"].unique()]
    insurers = [{"label": insurer, "value": insurer} for insurer in data["Company"].unique()]
    time_periods = [{"label": time, "value": time} for time in sorted(data["Time"].unique())]

    # Default selections
    default_metrics = [metrics[0]["value"]] if metrics else []
    default_insurers = default_insurers = ["Intact", "Definity"]
    default_time =     default_time = ["Q1 2023", "Q1 2024", "Q2 2023", "Q2 2024", "Q3 2023", "Q3 2024"]

    return metrics, default_metrics, insurers, default_insurers, time_periods, default_time


@app.callback(
    Output("tabs-content", "children"),
    [
        Input("tabs", "value"),
        Input("metrics", "value"),
        Input("insurers", "value"),
        Input("time-period", "value"),
        Input("data-type", "value")
    ]
)
def render_tab_content(selected_tab, selected_metrics, selected_insurers, selected_time_period, data_type):
    # Define custom sorting order for time periods
    time_order = ["Q1 2023", "Q2 2023", "Q3 2023", "H1 2023", 
                  "Q1 2024", "Q2 2024", "Q3 2024", "H1 2024"]
    
    # Select data based on type
    data = data_property if data_type == "Property" else data_auto

    if selected_tab == "bar":
        bar_charts = []
        for metric in selected_metrics:
            filtered_data = data[
                (data["Metric"] == metric) &
                (data["Company"].isin(selected_insurers)) &
                (data["Time"].isin(selected_time_period))
            ]
            fig = px.bar(
                filtered_data,
                x="Company",
                y="Value",
                color="Time",
                barmode="group",
                title=f"Bar Chart for {metric}",
                category_orders={"Time": time_order}
            )
            bar_charts.append(dcc.Graph(figure=fig))
        return html.Div(bar_charts)

    elif selected_tab == "trend":
        trend_charts = []
        filtered_data = data[
            (data["Metric"].isin(selected_metrics)) &
            (data["Company"].isin(selected_insurers)) &
            (data["Time"].isin(selected_time_period))
        ]
        for metric in selected_metrics:
            metric_data = filtered_data[filtered_data["Metric"] == metric]
            # Sort time periods in the order of the custom sorting key
            metric_data["Time"] = pd.Categorical(metric_data["Time"], categories=time_order, ordered=True)
            metric_data = metric_data.sort_values(by="Time")
            fig = px.line(
                metric_data,
                x="Time",
                y="Value",
                color="Company",
                title=f"Trend Chart for {metric}",
                markers=True
            )
            trend_charts.append(dcc.Graph(figure=fig, style={"marginBottom": "20px"}))
        return html.Div(trend_charts)


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
