import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import zscore
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, plotting
from pypfopt import expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from datetime import date

# Streamlit App Title
st.write("""# Yaseen's Portfolio Optimizer""")

# Stock Input
stocks_input = st.text_input("Enter the stock ticker symbols and number of shares (e.g. '100 nvda, 200 aapl, 300 etsy'):")

# Date Input
start_date = st.date_input('Start date', value=pd.to_datetime('2020-01-01'), min_value=pd.to_datetime('2000-01-01'), max_value=pd.to_datetime('today'))
end_date = st.date_input('End date', value=pd.to_datetime('today'), min_value=start_date, max_value=pd.to_datetime('today'))

# Optimization objective
objective = st.selectbox('Select the optimization objective', ['Maximize Sharpe Ratio', 'Minimize Volatility'])

# Risk Model Input
risk_model = st.selectbox('Select the risk model', ['Sample Covariance', 'Ledoit-Wolf Shrinkage'])

if stocks_input:
    # Parse the input to get the stocks and shares
    stocks_input = stocks_input.split(',')
    stocks = [stock.split()[1] for stock in stocks_input]
    shares = [int(stock.split()[0]) for stock in stocks_input]

    # Download Stock Data
    data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']

    # Check if the stock data was downloaded successfully for all ticker symbols
    missing_stocks = [stock for stock in stocks if stock not in data.columns]

if missing_stocks:
    st.write(f"Could not download data for: {', '.join(missing_stocks)}")
else:
    # Create a new dataframe that multiplies the daily adjusted close price by the number of shares
    portfolio_values = data.copy()
    for i, stock in enumerate(stocks):
        portfolio_values[stock] *= shares[i]

    # Sum the values for all stocks for each day to get the total portfolio value for each day
    portfolio_values['Total'] = portfolio_values.sum(axis=1)

    # Create a line chart of the total portfolio value
    st.line_chart(portfolio_values['Total'])

    # Calculate expected returns and the annualized sample covariance matrix of asset returns
    returns = data.pct_change()

    # Handle outliers by removing rows where the Z-score is > 3 or < -3
    z_scores = returns.apply(zscore)
    returns = returns[(z_scores < 3).all(axis=1) & (z_scores > -3).all(axis=1)]

    expected_returns = returns.mean() * 252

    # Calculate the covariance matrix based on the selected risk model
    if risk_model == 'Sample Covariance':
        cov_matrix = returns.cov() * 252
    elif risk_model == 'Ledoit-Wolf Shrinkage':
        cs = CovarianceShrinkage(returns)
        cov_matrix = cs.ledoit_wolf()

    # Portfolio Optimization
    ef = EfficientFrontier(expected_returns, cov_matrix)
    if objective == 'Maximize Sharpe Ratio':
        weights = ef.max_sharpe()
    elif objective == 'Minimize Volatility':
        weights = ef.min_volatility()

    cleaned_weights = ef.clean_weights()

    # Performance metrics of the optimal portfolio
    performance = ef.portfolio_performance(verbose=True)

    # Import plotly
    import plotly.graph_objects as go

    # Calculate standard deviations (volatility)
    std_devs = np.sqrt(np.diag(cov_matrix))  # Standard deviations for each stock

    # Create a new Plotly Figure
    fig = go.Figure()

    # Add a trace for each stock
    for i, stock in enumerate(stocks):
        fig.add_trace(
            go.Scatter(
                x=[std_devs[i]],  # Volatility
                y=[expected_returns[i]],  # Return
                mode='markers',
                name=stock,
                hovertemplate=
                '<i>Volatility</i>: %{x}' +
                '<br><b>Return</b>: %{y}<br>' +
                '<b>Stock</b>: %{text}',
                text=[stock],
            )
        )

    # Update layout
    fig.update_layout(
        title='Stock Returns and Volatility',
        xaxis=dict(title='Volatility'),
        yaxis=dict(title='Expected Return'),
        showlegend=True,
    )

    # Output
    st.write(cleaned_weights)
    st.write('Expected annual return: %.2f%%' % (performance[0]*100))
    st.write('Annual volatility: %.2f%%' % (performance[1]*100))
    st.write('Sharpe Ratio: %.2f' % performance[2])
    st.plotly_chart(fig)  # Use st.plotly_chart to display Plotly figures
