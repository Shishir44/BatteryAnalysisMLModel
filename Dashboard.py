# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output
# import numpy as np
# from datetime import datetime
# import dash_bootstrap_components as dbc
# import os

# # Initialize the Dash app with a Bootstrap theme
# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # Get the current directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# try:
#     # Try to read the CSV file
#     file_path = os.path.join(current_dir, 'merged_data.csv')
#     print(f"Attempting to read file from: {file_path}")
#     df = pd.read_csv(file_path)

#     # Calculate cycle-based statistics
#     cycle_stats = df.groupby('cycle').agg({
#         'capacity': 'mean',
#         'temperature': 'mean',
#         'SOH': 'mean',
#         'terminal_voltage': 'mean',
#         'terminal_current': 'mean'
#     }).reset_index()

#     # Create the layout with data visualizations
#     app.layout = dbc.Container([
#         dbc.Row([
#             dbc.Col([
#                 html.H1("Battery Performance Analysis Dashboard",
#                        className="text-center mb-4 mt-4")
#             ])
#         ]),

#         # First row of graphs
#         dbc.Row([
#             # SOH Over Cycles
#             dbc.Col([
#                 dbc.Card([
#                     dbc.CardHeader("State of Health (SOH) Over Cycles"),
#                     dbc.CardBody([
#                         dcc.Graph(
#                             id='soh-cycles-graph',
#                             figure=px.line(
#                                 cycle_stats,
#                                 x='cycle',
#                                 y='SOH',
#                                 title='SOH vs Cycles'
#                             ).update_layout(
#                                 xaxis_title="Cycle Number",
#                                 yaxis_title="SOH (%)",
#                                 hovermode='x unified'
#                             )
#                         )
#                     ])
#                 ], className="mb-4")
#             ], width=6),

#             # Capacity Over Cycles
#             dbc.Col([
#                 dbc.Card([
#                     dbc.CardHeader("Battery Capacity Over Cycles"),
#                     dbc.CardBody([
#                         dcc.Graph(
#                             id='capacity-cycles-graph',
#                             figure=px.line(
#                                 cycle_stats,
#                                 x='cycle',
#                                 y='capacity',
#                                 title='Capacity vs Cycles'
#                             ).update_layout(
#                                 xaxis_title="Cycle Number",
#                                 yaxis_title="Capacity",
#                                 hovermode='x unified'
#                             )
#                         )
#                     ])
#                 ], className="mb-4")
#             ], width=6)
#         ]),

#         # Second row of graphs
#         dbc.Row([
#             # Temperature Over Cycles
#             dbc.Col([
#                 dbc.Card([
#                     dbc.CardHeader("Temperature Over Cycles"),
#                     dbc.CardBody([
#                         dcc.Graph(
#                             id='temperature-cycles-graph',
#                             figure=px.line(
#                                 cycle_stats,
#                                 x='cycle',
#                                 y='temperature',
#                                 title='Temperature vs Cycles'
#                             ).update_layout(
#                                 xaxis_title="Cycle Number",
#                                 yaxis_title="Temperature (°C)",
#                                 hovermode='x unified'
#                             )
#                         )
#                     ])
#                 ], className="mb-4")
#             ], width=6),

#             # Capacity vs SOH Correlation
#             dbc.Col([
#                 dbc.Card([
#                     dbc.CardHeader("Capacity vs State of Health Correlation"),
#                     dbc.CardBody([
#                         dcc.Graph(
#                             id='capacity-soh-correlation',
#                             figure=px.scatter(
#                                 df.sample(1000),  # Sample for better performance
#                                 x='capacity',
#                                 y='SOH',
#                                 title='Capacity vs SOH Correlation'
#                             ).update_layout(
#                                 xaxis_title="Capacity",
#                                 yaxis_title="SOH (%)",
#                                 hovermode='closest'
#                             )
#                         )
#                     ])
#                 ], className="mb-4")
#             ], width=6)
#         ]),

#         # Third row - Statistics Cards
#         dbc.Row([
#             # Basic Statistics Card
#             dbc.Col([
#                 dbc.Card([
#                     dbc.CardHeader("Basic Statistics"),
#                     dbc.CardBody([
#                         html.Div([
#                             html.H6(f"Total Cycles: {df['cycle'].nunique()}"),
#                             html.H6(f"Average SOH: {df['SOH'].mean():.2f}%"),
#                             html.H6(f"Average Capacity: {df['capacity'].mean():.2f}"),
#                             html.H6(f"Average Temperature: {df['temperature'].mean():.2f}°C")
#                         ])
#                     ])
#                 ], className="mb-4")
#             ], width=6),

#             # Correlation Statistics Card
#             dbc.Col([
#                 dbc.Card([
#                     dbc.CardHeader("Parameter Correlations with SOH"),
#                     dbc.CardBody([
#                         html.Div([
#                             html.H6(f"Capacity Correlation: {df['capacity'].corr(df['SOH']):.3f}"),
#                             html.H6(f"Temperature Correlation: {df['temperature'].corr(df['SOH']):.3f}"),
#                             html.H6(f"Terminal Voltage Correlation: {df['terminal_voltage'].corr(df['SOH']):.3f}"),
#                             html.H6(f"Terminal Current Correlation: {df['terminal_current'].corr(df['SOH']):.3f}")
#                         ])
#                     ])
#                 ], className="mb-4")
#             ], width=6)
#         ]),

#         # Add cycle selector for detailed view
#         dbc.Row([
#             dbc.Col([
#                 dbc.Card([
#                     dbc.CardHeader("Cycle Details"),
#                     dbc.CardBody([
#                         dcc.Slider(
#                             id='cycle-slider',
#                             min=df['cycle'].min(),
#                             max=df['cycle'].max(),
#                             step=1,
#                             value=df['cycle'].min(),
#                             marks={i: str(i) for i in range(
#                                 df['cycle'].min(),
#                                 df['cycle'].max()+1,
#                                 max(1, int((df['cycle'].max() - df['cycle'].min())/10))
#                             )}
#                         ),
#                         dcc.Graph(id='cycle-detail-graph')
#                     ])
#                 ])
#             ])
#         ])
#     ], fluid=True)

# except FileNotFoundError:
#     # Create a simple layout with error message if file is not found
#     app.layout = dbc.Container([
#         dbc.Row([
#             dbc.Col([
#                 html.H1("Error Loading Dashboard",
#                        className="text-center text-danger mb-4 mt-4"),
#                 html.Div([
#                     html.P("Could not find the data file 'merged_data.csv'"),
#                     html.P(f"Please ensure the file is located at: {file_path}"),
#                     html.P("Place the CSV file in the same directory as this script and try again.")
#                 ], className="text-center")
#             ])
#         ])
#     ])

# except Exception as e:
#     # Handle any other errors
#     app.layout = dbc.Container([
#         dbc.Row([
#             dbc.Col([
#                 html.H1("Error Loading Dashboard",
#                        className="text-center text-danger mb-4 mt-4"),
#                 html.Div([
#                     html.P(f"An error occurred: {str(e)}"),
#                     html.P("Please check the data file and try again.")
#                 ], className="text-center")
#             ])
#         ])
#     ])

# # Callback for cycle detail graph
# @app.callback(
#     Output('cycle-detail-graph', 'figure'),
#     [Input('cycle-slider', 'value')]
# )
# def update_cycle_detail(selected_cycle):
#     try:
#         cycle_data = df[df['cycle'] == selected_cycle]
        
#         fig = go.Figure()
        
#         # Add voltage trace
#         fig.add_trace(go.Scatter(
#             y=cycle_data['terminal_voltage'],
#             name='Terminal Voltage',
#             line=dict(color='blue')
#         ))
        
#         # Add current trace
#         fig.add_trace(go.Scatter(
#             y=cycle_data['terminal_current'],
#             name='Terminal Current',
#             line=dict(color='red'),
#             yaxis='y2'
#         ))

#         # Update layout
#         fig.update_layout(
#             title=f'Detailed View for Cycle {selected_cycle}',
#             xaxis_title='Measurement Point',
#             yaxis_title='Terminal Voltage',
#             yaxis2=dict(
#                 title='Terminal Current',
#                 overlaying='y',
#                 side='right'
#             ),
#             hovermode='x unified'
#         )
        
#         return fig
#     except Exception as e:
#         # Return an empty figure with error message if something goes wrong
#         return go.Figure().add_annotation(
#             text=f"Error loading cycle detail: {str(e)}",
#             xref="paper", yref="paper",
#             x=0.5, y=0.5, showarrow=False
#         )

# if __name__ == '__main__':
#     print(f"Looking for data file at: {file_path}")
#     app.run_server(debug=True)