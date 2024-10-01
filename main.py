import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# Initial dataset
data = {'X': [1, 2, 2.5, 4, 5, 6, 7, 8, 9, 10],
        'y': [2, 4, 5, 4, 5, 7, 8, 9, 9, 12]}
df = pd.DataFrame(data)

# Create Dash app
app = dash.Dash(__name__)
app.title = "Real-Time Linear Regression Analysis"


# Initial model setup
def train_model(df):
    X = df[['X']]  # Feature
    y = df['y']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X_test, y_test, y_pred


model, mse, r2, X_test, y_test, y_pred = train_model(df)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Real-Time Linear Regression with Visualizations", style={'text-align': 'center'}),
    html.H2(id='performance-metrics', style={'text-align': 'center'}),
    html.Div([
        dcc.Graph(id='scatter-plot', style={'display': 'inline-block', 'width': '48%'}),
        dcc.Graph(id='residual-plot', style={'display': 'inline-block', 'width': '48%'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    html.Div([
        dcc.Graph(id='residual-hist', style={'display': 'inline-block', 'width': '48%'}),
        dcc.Graph(id='actual-vs-predicted', style={'display': 'inline-block', 'width': '48%'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    dcc.Graph(id='learning-curve-plot', style={'display': 'inline-block', 'width': '100%'}),

    dcc.Interval(
        id='interval-component',
        interval=5000,  # Update every 5 seconds
        n_intervals=0  # Number of times interval has triggered
    )
])


# Update the plots and metrics periodically
@app.callback(
    [Output('performance-metrics', 'children'),
     Output('scatter-plot', 'figure'),
     Output('residual-plot', 'figure'),
     Output('residual-hist', 'figure'),
     Output('actual-vs-predicted', 'figure'),
     Output('learning-curve-plot', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    global df
    # Simulate new data being added
    new_data = {'X': [df['X'].max() + 1], 'y': [df['y'].max() + np.random.randn()]}
    new_df = pd.DataFrame(new_data)
    df = pd.concat([df, new_df], ignore_index=True)

    # Retrain the model with new data
    model, mse, r2, X_test, y_test, y_pred = train_model(df)

    # 1. Scatter Plot with Regression Line
    scatter_plot = go.Figure()
    scatter_plot.add_trace(go.Scatter(x=df['X'], y=df['y'], mode='markers', name='Data points'))
    scatter_plot.add_trace(
        go.Scatter(x=df['X'], y=model.predict(df[['X']]), mode='lines', name='Regression line', line=dict(color='red')))
    scatter_plot.update_layout(title='Scatter Plot with Regression Line', xaxis_title='X', yaxis_title='y')

    # 2. Residual Plot
    residuals = y_test - y_pred
    residual_plot = go.Figure()
    residual_plot.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
    residual_plot.add_shape(type='line', x0=y_pred.min(), y0=0, x1=y_pred.max(), y1=0,
                            line=dict(color='red', dash='dash'))
    residual_plot.update_layout(title='Residual Plot', xaxis_title='Predicted Values', yaxis_title='Residuals')

    # 3. Distribution Plot of Residuals
    residual_hist = px.histogram(residuals, nbins=10, title="Distribution of Residuals")
    residual_hist.update_layout(xaxis_title='Residuals', yaxis_title='Frequency')

    # 4. Actual vs Predicted Values Plot
    actual_vs_pred = go.Figure()
    actual_vs_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs Predicted'))
    actual_vs_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                             line=dict(color='red', dash='dash'))
    actual_vs_pred.update_layout(title='Actual vs Predicted Values', xaxis_title='Actual Values',
                                 yaxis_title='Predicted Values')

    # 5. Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(model, df[['X']], df['y'], cv=5,
                                                            scoring='neg_mean_squared_error')
    train_mean = np.mean(-train_scores, axis=1)
    test_mean = np.mean(-test_scores, axis=1)

    learning_curve_plot = go.Figure()
    learning_curve_plot.add_trace(
        go.Scatter(x=train_sizes, y=train_mean, mode='lines', name='Training Error', line=dict(color='blue')))
    learning_curve_plot.add_trace(
        go.Scatter(x=train_sizes, y=test_mean, mode='lines', name='Testing Error', line=dict(color='green')))
    learning_curve_plot.update_layout(title='Learning Curve', xaxis_title='Training Set Size',
                                      yaxis_title='Mean Squared Error')

    # Update performance metrics
    performance_metrics = f"Model Performance: MSE = {mse:.2f}, R² = {r2:.2f}"

    return performance_metrics, scatter_plot, residual_plot, residual_hist, actual_vs_pred, learning_curve_plot


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
