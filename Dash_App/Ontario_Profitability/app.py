import geopandas as gpd
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pickle
import os
ontario_file_path = "https://raw.githubusercontent.com/JohnOkoth/actuarialangles/main/data/summarized3.csv"
shapefile_path = "https://raw.githubusercontent.com/JohnOkoth/actuarialangles/Dash_App/Ontario_Profitability/ontario_regions.shp"
processed_gdf_path = "https://raw.githubusercontent.com/JohnOkoth/actuarialangles/Dash_App/Ontario_Profitability/processed_ontario_regions.pkl"



# Pre-process shapefile and save to disk for faster loading
shapefile_path = "ontario_regions.shp"
processed_gdf_path = "processed_ontario_regions.pkl"

if os.path.exists(processed_gdf_path):
    with open(processed_gdf_path, 'rb') as f:
        gdf = pickle.load(f)
else:
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != "EPSG:3347":
        gdf = gdf.to_crs(epsg=3347)
    gdf["centroid"] = gdf.geometry.centroid
    gdf = gdf.to_crs(epsg=4326)
    gdf["lat"] = gdf["centroid"].y
    gdf["long"] = gdf["centroid"].x
    gdf.drop(columns=["centroid"], inplace=True)
    gdf["geometry"] = gdf.simplify(tolerance=0.005, preserve_topology=True)
    with open(processed_gdf_path, 'wb') as f:
        pickle.dump(gdf, f)

# Load summarized data
df = pd.read_csv(ontario_file_path)

# Pre-compute unique values for dropdowns
coverage_options = [{'label': cov, 'value': cov} for cov in df["MinorCoverageType"].unique()]
accident_year_options = [{'label': str(year), 'value': year} for year in sorted(df["AccidentYear"].unique())]
loss_ratio_levels = [{'label': lvl, 'value': lvl} for lvl in df["LossRatio_Level"].unique()]
claim_frequency_levels = [{'label': lvl, 'value': lvl} for lvl in df["ClaimFrequency_Level"].unique()]
combined_ratio_levels = [{'label': lvl, 'value': lvl} for lvl in df["CombinedRatio_Level"].unique()]
exposure_levels = [{'label': lvl, 'value': lvl} for lvl in df["Exposure_Level"].unique()]

# Pre-compute region mapping and cache it
region_map_path = "region_to_cdnames.pkl"
if os.path.exists(region_map_path):
    with open(region_map_path, 'rb') as f:
        region_to_cdnames = pickle.load(f)
else:
    region_to_cdnames = {region: set(gdf[gdf["CDNAME"].str.contains(region, case=False, na=False)]["CDNAME"].tolist()) 
                         for region in df["Region_1"].unique()}
    with open(region_map_path, 'wb') as f:
        pickle.dump(region_to_cdnames, f)

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Inject Google Analytics script and Home Button
app.index_string = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>Ontario Auto Profitability Dashboard</title>
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
    html.H1("Ontario Region Auto Profitability Map", style={'textAlign': 'center'}),

    html.Div(id='triggered-regions-text', 
             style={'textAlign': 'center', 'color': 'blue', 'fontSize': '18px', 'marginBottom': '10px'}),  

    html.Div([
        html.Div([
            html.Label("Coverage", style={'fontSize': '16px', 'fontWeight': 'bold', 'display': 'block'}),
            dcc.Dropdown(id='coverage-dropdown', options=coverage_options, placeholder="Select Coverage",
                         style={'width': '100%'})
        ], style={'width': '16%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'}),

        html.Div([
            html.Label("Accident Year", style={'fontSize': '16px', 'fontWeight': 'bold', 'display': 'block'}),
            dcc.Dropdown(id='accident-year-dropdown', options=accident_year_options, placeholder="Select Accident Year",
                         style={'width': '100%'})
        ], style={'width': '16%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'}),

        html.Div([
            html.Label("Loss Ratio", style={'fontSize': '16px', 'fontWeight': 'bold', 'display': 'block'}),
            dcc.Dropdown(id='loss-ratio-level-dropdown', options=loss_ratio_levels, placeholder="Select Loss Ratio Level",
                         style={'width': '100%'})
        ], style={'width': '16%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'})
    ], style={'width': '100%', 'textAlign': 'center', 'margin-bottom': '20px', 'display': 'flex', 'justifyContent': 'center'}),

    html.Div([
        html.Div([
            html.Label("Frequency", style={'fontSize': '16px', 'fontWeight': 'bold', 'display': 'block'}),
            dcc.Dropdown(id='claim-frequency-dropdown', options=claim_frequency_levels, placeholder="Select Claim Frequency Level",
                         style={'width': '100%'})
        ], style={'width': '16%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'}),

        html.Div([
            html.Label("Combined Ratio", style={'fontSize': '16px', 'fontWeight': 'bold', 'display': 'block'}),
            dcc.Dropdown(id='combined-ratio-dropdown', options=combined_ratio_levels, placeholder="Select Combined Ratio Level",
                         style={'width': '100%'})
        ], style={'width': '16%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'}),

        html.Div([
            html.Label("Exposure", style={'fontSize': '16px', 'fontWeight': 'bold', 'display': 'block'}),
            dcc.Dropdown(id='exposure-level-dropdown', options=exposure_levels, placeholder="Select Exposure Level",
                         style={'width': '100%'})
        ], style={'width': '16%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'})
    ], style={'width': '100%', 'textAlign': 'center', 'display': 'flex', 'justifyContent': 'center'}),

    html.Div([
        html.Label("Select a Region to Zoom", style={'fontSize': '16px', 'fontWeight': 'bold', 'display': 'block'}),
        dcc.Dropdown(id='selected-region-dropdown', options=[], placeholder="Select a Region",
                     style={'width': '30%', 'margin': '10px auto'}),
        html.Button("Reset Zoom", id='reset-zoom-button', n_clicks=0, 
                    style={'margin': '10px', 'fontSize': '16px', 'padding': '5px 10px'})
    ], style={'textAlign': 'center', 'margin-bottom': '20px', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}),

    dcc.Graph(id='map-output', style={'height': '800px'}),
    html.Div(id='no-region-message', 
             style={'textAlign': 'center', 'color': 'red', 'fontSize': '18px', 'marginTop': '10px'}),
    html.H3("Triggered Regions", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'black'})
])

# Callback with reset zoom feature
@app.callback(
    [
        Output('map-output', 'figure'),
        Output('triggered-regions-text', 'children'),
        Output('no-region-message', 'children'),
        Output('selected-region-dropdown', 'options'),
        Output('selected-region-dropdown', 'value')
    ],
    [
        Input('coverage-dropdown', 'value'),
        Input('accident-year-dropdown', 'value'),
        Input('loss-ratio-level-dropdown', 'value'),
        Input('claim-frequency-dropdown', 'value'),
        Input('combined-ratio-dropdown', 'value'),
        Input('exposure-level-dropdown', 'value'),
        Input('selected-region-dropdown', 'value'),
        Input('reset-zoom-button', 'n_clicks')
    ],
    [State('selected-region-dropdown', 'value')]
)
def update_map(selected_coverage, selected_accident_year, selected_loss_ratio, selected_claim_frequency, 
               selected_combined_ratio, selected_exposure_level, selected_region, n_clicks, current_region):
    # Track if reset was triggered
    ctx = dash.callback_context
    reset_triggered = 'reset-zoom-button.n_clicks' in ctx.triggered_prop_ids and n_clicks > 0

    # If reset is triggered, override selected_region to None
    if reset_triggered:
        selected_region = None

    # Default map if no filters are applied
    if not any([selected_coverage, selected_accident_year, selected_loss_ratio, selected_claim_frequency, 
                selected_combined_ratio, selected_exposure_level]):
        return (
            px.scatter_mapbox(lat=[43.7], lon=[-79.4], zoom=4, mapbox_style="open-street-map"), 
            "No filters applied - showing default map (0 regions)", 
            "", 
            [], 
            None
        )

    # Vectorized filtering
    mask = pd.Series(True, index=df.index)
    if selected_coverage:
        mask &= df["MinorCoverageType"] == selected_coverage
    if selected_accident_year:
        mask &= df["AccidentYear"] == selected_accident_year
    if selected_loss_ratio:
        mask &= df["LossRatio_Level"] == selected_loss_ratio
    if selected_claim_frequency:
        mask &= df["ClaimFrequency_Level"] == selected_claim_frequency
    if selected_combined_ratio:
        mask &= df["CombinedRatio_Level"] == selected_combined_ratio
    if selected_exposure_level:
        mask &= df["Exposure_Level"] == selected_exposure_level
    
    filtered_df = df[mask]

    if filtered_df.empty:
        return (
            px.scatter_mapbox(lat=[43.7], lon=[-79.4], zoom=4, mapbox_style="open-street-map"), 
            "No Filtered Region In The Criteria (0 regions)", 
            "No Region Filtered", 
            [], 
            None
        )

    # Optimized region matching
    matching_cdnames = set().union(*[region_to_cdnames.get(region, set()) for region in filtered_df["Region_1"].unique()])
    filtered_gdf = gdf[gdf["CDNAME"].isin(matching_cdnames)].copy()

    if filtered_gdf.empty:
        return (
            px.scatter_mapbox(lat=[43.7], lon=[-79.4], zoom=4, mapbox_style="open-street-map"), 
            "No Filtered Region In The Criteria (0 regions)", 
            "No Region Filtered", 
            [], 
            None
        )

    filtered_gdf["Region_1"] = filtered_gdf["CDNAME"].map(
        lambda x: next((key for key, values in region_to_cdnames.items() if x in values), "Unknown")
    )

    triggered_regions = filtered_gdf["Region_1"].unique().tolist()
    region_options = [{'label': region, 'value': region} for region in triggered_regions]

    # Zoom logic
    if selected_region and selected_region in triggered_regions and not reset_triggered:
        region_gdf = filtered_gdf[filtered_gdf["Region_1"] == selected_region]
        bounds = region_gdf.total_bounds
        center_lat, center_lon = (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2
        zoom_level = 8
    else:
        # Default zoom when no region is selected or after reset
        bounds = filtered_gdf.total_bounds
        center_lat, center_lon = (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2
        zoom_level = max(6, min(12, 10 - max(bounds[2] - bounds[0], bounds[3] - bounds[1])))

    fig = px.choropleth_mapbox(
        filtered_gdf, geojson=filtered_gdf.__geo_interface__, locations=filtered_gdf.index,
        color="Region_1", mapbox_style="open-street-map", center={"lat": center_lat, "lon": center_lon},
        zoom=zoom_level, opacity=0.5
    )
    fig.update_traces(marker_line_color="black", marker_line_width=1)

    region_count = len(triggered_regions)
    triggered_text_base = ", ".join(triggered_regions[:10]) + (f", and {region_count - 10} more..." if region_count > 10 else "")
    triggered_text = f"Filtered Regions ({region_count}): {triggered_text_base}"

    # Return None for dropdown value if reset was triggered, otherwise keep the selected value
    return fig, triggered_text, "", region_options, None if reset_triggered else selected_region
if __name__ == '__main__':
    app.run_server(debug=True)