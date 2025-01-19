from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go

# Load the data (replace with your actual file path)
#data_path = "https://raw.githubusercontent.com/JohnOkoth/actuarialangles/main/data/ontario_combined_data_with_coverage.xlsx"

data_path = r"C:\\Users\\grace\\OneDrive\\Documents\\data\\ontario_combined_data_with_coverage.xlsx"

data = pd.read_excel(data_path)

# Ensure the data contains the expected columns
print(data.columns)  # Check that 'Coverage' and other required columns exist

# Check if 'Accident Semester' exists and process year and semester columns
if 'Accident Semester' in data.columns:
    data['Accident Semester'] = data['Accident Semester'].astype(str)  # Convert to string
    data['Year'] = data['Accident Semester'].str.split(".").str[0].astype(int)
    data['Semester'] = data['Accident Semester'].str.split(".").str[1].astype(int)
if 'Maturity (Months)' in data.columns:
    data['Maturity (Months)'] = data['Maturity (Months)'].astype(int)
else:
    raise KeyError("The required columns are missing from the dataset.")

# Initialize the Dash app
app = Dash(__name__)
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


app.layout = html.Div([
    html.H1("Ontario Auto Coverage Trend Analysis", style={'textAlign': 'center', 'width': '80%'}),

    html.Div([
        # Sidebar for selection pane
        html.Div([
            html.Label("Select Coverages:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Checklist(
                id='coverage-checklist',
                options=[{'label': cov, 'value': cov} for cov in data['Coverage'].unique()],
                value=['BI Coverage', 'Collision', 'Comprehensive'],  # Default values
                inline=False
            ),

            html.Label("Select Time Period:", style={'fontWeight': 'bold', 'marginTop': '20px', 'marginBottom': '10px'}),
            dcc.RadioItems(
                id='time-period-radio',
                options=[{'label': 'Annual', 'value': 'Annual'},
                         {'label': 'Half Year', 'value': 'Half Year'}],
                value='Annual'
            ),

            html.Label("Toggle Exposure Bar Chart:", style={'fontWeight': 'bold', 'marginTop': '20px', 'marginBottom': '10px'}),
            dcc.Checklist(
                id='exposure-toggle',
                options=[{'label': 'Include Exposure', 'value': 'Include'}],
                value=[]
            ),

            html.Div([
                html.Div([
                    html.Label("Select Range (Annual):", style={'fontWeight': 'bold', 'marginBottom': '20px'}),
                    dcc.RangeSlider(
                        id='range-slider',
                        min=data['Year'].min(),
                        max=data['Year'].max(),
                        step=1,
                        value=[data['Year'].min(), data['Year'].max()],
                        marks={i: {'label': str(i), 'style': {'transform': 'rotate(-90deg)', 'color': '#000'}} for i in range(data['Year'].min(), data['Year'].max() + 1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], id='range-slider-container', style={'display': 'block', 'marginTop': '20px'}),

                html.Div([
                    html.Label("Select Maturity (Months) Range (Half Year):", style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(
                        id='maturity-slider',
                        min=data['Maturity (Months)'].min(),
                        max=data['Maturity (Months)'].max(),
                        step=6,
                        value=[data['Maturity (Months)'].min(), data['Maturity (Months)'].max()],
                        marks={i: {'label': str(i), 'style': {'transform': 'rotate(-90deg)', 'color': '#000'}} for i in range(data['Maturity (Months)'].min(), data['Maturity (Months)'].max() + 6, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], id='maturity-slider-container', style={'display': 'none', 'marginTop': '20px'})
            ], style={'marginTop': '100px'}),

        ], style={'flex': '1', 'padding': '20px'}),

        # Charts container
        html.Div(id='charts-container', style={'flex': '3', 'padding': '20px'}),

    ], style={'display': 'flex'}),
])

@app.callback(
    [Output('charts-container', 'children'),
     Output('range-slider-container', 'style'),
     Output('maturity-slider-container', 'style')],
    [Input('coverage-checklist', 'value'),
     Input('time-period-radio', 'value'),
     Input('exposure-toggle', 'value'),
     Input('range-slider', 'value'),
     Input('maturity-slider', 'value')]
)
def update_charts(selected_coverages, time_period, exposure_toggle, range_value, maturity_value):
    if not selected_coverages:
        return html.Div("Select at least one coverage to display charts."), {'display': 'none'}, {'display': 'none'}

    charts = []

    if time_period == 'Annual':
        slider_style = {'display': 'block'}
        maturity_style = {'display': 'none'}
        filtered_data = data[(data['Year'] >= range_value[0]) & (data['Year'] <= range_value[1])]
        x_col = 'Year'
    else:
        slider_style = {'display': 'none'}
        maturity_style = {'display': 'block'}
        filtered_data = data[(data['Maturity (Months)'] >= maturity_value[0]) & (data['Maturity (Months)'] <= maturity_value[1])]
        x_col = 'Accident Semester'

    for coverage in selected_coverages:
        coverage_data = filtered_data[filtered_data['Coverage'] == coverage]

        grouped_data = coverage_data.groupby(x_col).agg({
            'Ultimate Loss Cost': 'mean',
            'Ultimate Severity': 'mean',
            'Ultimate Frequency per 1000': 'mean',
            'Earned Car Years': 'sum'
        }).reset_index()

        # Create Loss Cost Chart
        fig_loss_cost = go.Figure()
        fig_loss_cost.add_trace(go.Scatter(
            x=grouped_data[x_col],
            y=grouped_data['Ultimate Loss Cost'],
            mode='lines+markers',
            name='Loss Cost Trend',
            line=dict(color='blue')
        ))

        if 'Include' in exposure_toggle:
            fig_loss_cost.add_trace(go.Bar(
                x=grouped_data[x_col],
                y=grouped_data['Earned Car Years'] / 1e6,
                name='Exposure (Millions)',
                marker=dict(color='orange', opacity=0.5),
                yaxis='y2'
            ))

        fig_loss_cost.update_layout(
            title=f"{coverage} - Loss Cost",
            xaxis_title=x_col,
            yaxis_title="Loss Cost",
            yaxis2=dict(
                title="Exposure (Millions)",
                overlaying='y',
                side='right'
            ),
            template="plotly_white"
        )

        # Create Severity Chart
        fig_severity = go.Figure()
        fig_severity.add_trace(go.Scatter(
            x=grouped_data[x_col],
            y=grouped_data['Ultimate Severity'],
            mode='lines+markers',
            name='Severity Trend',
            line=dict(color='red')
        ))

        if 'Include' in exposure_toggle:
            fig_severity.add_trace(go.Bar(
                x=grouped_data[x_col],
                y=grouped_data['Earned Car Years'] / 1e6,
                name='Exposure (Millions)',
                marker=dict(color='orange', opacity=0.5),
                yaxis='y2'
            ))

        fig_severity.update_layout(
            title=f"{coverage} - Severity",
            xaxis_title=x_col,
            yaxis_title="Severity",
            yaxis2=dict(
                title="Exposure (Millions)",
                overlaying='y',
                side='right'
            ),
            template="plotly_white"
        )

        # Create Frequency Chart
        fig_frequency = go.Figure()
        fig_frequency.add_trace(go.Scatter(
            x=grouped_data[x_col],
            y=grouped_data['Ultimate Frequency per 1000'],
            mode='lines+markers',
            name='Frequency Trend',
            line=dict(color='green')
        ))

        if 'Include' in exposure_toggle:
            fig_frequency.add_trace(go.Bar(
                x=grouped_data[x_col],
                y=grouped_data['Earned Car Years'] / 1e6,
                name='Exposure (Millions)',
                marker=dict(color='orange', opacity=0.5),
                yaxis='y2'
            ))

        fig_frequency.update_layout(
            title=f"{coverage} - Frequency",
            xaxis_title=x_col,
            yaxis_title="Frequency",
            yaxis2=dict(
                title="Exposure (Millions)",
                overlaying='y',
                side='right'
            ),
            template="plotly_white"
        )

        charts.append(html.Div([
            html.H3(f"Coverage: {coverage}"),
            html.Div([
                dcc.Graph(figure=fig_loss_cost),
                dcc.Graph(figure=fig_severity),
                dcc.Graph(figure=fig_frequency)
            ], style={'display': 'flex', 'justify-content': 'space-between'})
        ]))

    return charts, slider_style, maturity_style

if __name__ == '__main__':
    app.run_server(debug=True)
