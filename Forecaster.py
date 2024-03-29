import requests
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np 
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from github import Github
import io



st.title('Runge Forecasting')
st.write ("")
st.write ("Bitcoin - Daily Forecaster")
st.write ("")
st.write ("Inputs : ")
st.write( " -> BITUSD / Nasdaq / NYSE / Shanghai-Composite Index")
st.write( " -> NSE-India / Korean-SE / FSTE-Index / Hang Seng- Index / Nikki")

backfill = True

if st.button("Forecast"):
  
    ######################################################### IMPORT DATA 
    st.write ("Importing data")  
    
    # Get the current date and time
    current_date = datetime.now()
    # Strip the time information
    current_date_only = current_date.date()
    # Convert the date to a string
    current_date_string = current_date_only.strftime('%Y-%m-%d')
    del(current_date, current_date_only )
    
    
    # Define the ticker symbols for the exchanges and Bitcoin
    ticker_symbols = ['^IXIC', '^NYA', '000001.SS', '^NSEI', '^KS11', '^FTSE', '^HSI', '^N225', 'BTC-USD']
    
    # Fetch daily historical data for the past 5 years
    data = yf.download(ticker_symbols, start='2019-02-16', end=current_date_string)
    
    st.write( " ----> Data successfully imported")
    
    
    ######################################################### DATA PROCESSING 
    # IF FILLING WITH PREVIOUS DATA 
    if backfill is True:
       data = data.fillna(method='ffill')
       data.dropna(inplace=True)
    else:
        data.dropna(inplace=True)
    
    # Flatten the multi-level column index into a single level
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    # Reset index to make Date a column
    data.reset_index(inplace=True)
    # Set Date to index
    data = data.set_index('Date')
    
    # FIND THE NEXT FIVE DAYS
    # Get the last/most upto date sample point 
    last_timestamp = data.index[-1]  # Get the last timestamp
    last_time_stamp2 =  last_timestamp.strftime('%Y-%m-%d')
    
    # Calculate the next five days
    next_five_days = pd.date_range(start=last_timestamp, periods=6)[1:]  # Exclude the first entry (last_timestamp)
    # Extract only the dates from next_five_days
    next_five_days_dates = next_five_days.date
    next_five_days_dates = pd.DataFrame(next_five_days_dates, columns =['Date'])
    next_five_days_dates = next_five_days_dates.set_index('Date')
    del(next_five_days)
    
    st.write( " ----> Data successfully processed")
    
    
    
    ######################################################### MODEL IMPORTING PROCESSING 
    st.write ("Importing models") 
        
    # Load the Multi-output-XGboost Bitcoin daily high model
    Bitcoin_XGBoost_Daily_high = joblib.load('./BCH-XG.pkl')


    # Load the Multi-output-XGboost Bitcoin daily low model
    Bitcoin_XGBoost_Daily_low = joblib.load('./BCL-XG.pkl')
    
    # Load the Multi-output-XGboost Bitcoin daily close model
    Bitcoin_XGBoost_Daily_close = joblib.load('./BCC-XG.pkl')
    
    st.write( " ----> Models successfully processed")
    
    
    
        
    ######################################################### MODEL FORECASTING  
    st.write ("Generating forecasts ") 
    # FORECAST NEXT FIVE DAYS HIGHS
    Y_forecast_High_full = Bitcoin_XGBoost_Daily_high.predict(data)
    Y_forecast_High_today = Y_forecast_High_full[-1]
    
    
    # FORECAST NEXT FIVE DAYS LOWS
    Y_forecast_Low_full = Bitcoin_XGBoost_Daily_low.predict(data)
    Y_forecast_Low_today = Y_forecast_Low_full[-1]
    
    
    # FORECAST NEXT FIVE DAYS CLOSES
    Y_forecast_Close_full = Bitcoin_XGBoost_Daily_close.predict(data)
    Y_forecast_Close_today = Y_forecast_Close_full[-1]
    
    
    
    # MERGE INTO OUTPUT 
    Output = {
        'Bitcoin High': Y_forecast_High_today,
        'Bitcoin Low' : Y_forecast_Low_today,
        'Bitcoin Close' : Y_forecast_Close_today
        }
    Output = pd.DataFrame(Output)
    Output.index = next_five_days_dates.index
    
    
    
    Y_forecast_High_today = {
        'Input Sample Date': [last_time_stamp2],
        't+1': [Y_forecast_High_today[0]],
        't+2': [Y_forecast_High_today[1]],
        't+3': [Y_forecast_High_today[2]],
        't+4': [Y_forecast_High_today[3]],
        't+5': [Y_forecast_High_today[4]]
    }
    # Create a DataFrame from the dictionary
    Y_forecast_High_today = pd.DataFrame(Y_forecast_High_today)
    
    
    Y_forecast_Low_today = {
        'Input Sample Date': [last_time_stamp2],
        't+1': [Y_forecast_Low_today[0]],
        't+2': [Y_forecast_Low_today[1]],
        't+3': [Y_forecast_Low_today[2]],
        't+4': [Y_forecast_Low_today[3]],
        't+5': [Y_forecast_Low_today[4]]
    }
    # Create a DataFrame from the dictionary
    Y_forecast_Low_today = pd.DataFrame(Y_forecast_Low_today)
    
    Y_forecast_Close_today = {
        'Input Sample Date': [last_time_stamp2],
        't+1': [Y_forecast_Close_today[0]],
        't+2': [Y_forecast_Close_today[1]],
        't+3': [Y_forecast_Close_today[2]],
        't+4': [Y_forecast_Close_today[3]],
        't+5': [Y_forecast_Close_today[4]]
    }
    # Create a DataFrame from the dictionary
    Y_forecast_Close_today = pd.DataFrame(Y_forecast_Close_today)
    del(last_time_stamp2)
    
    st.write( " ----> Forecasts successfully generated")
    
    
    
    
    ######################################################### RESULTS EXPORTING 
    
    # Assuming Output is your DataFrame with dates as the index and 'Bitcoin High', 'Bitcoin Low', 'Bitcoin Close' as columns
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size if needed
    for column in Output.columns:
        ax.plot(Output.index, Output[column], label=column)
    
    ax.set_xlabel('Date')  # Set x-axis label
    ax.set_ylabel('Dollars')  # Set y-axis label
    ax.set_title('Bitcoin Forecasts')  # Set plot title
    ax.legend()  # Show legend
    ax.grid(True)  # Show grid
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    
    # Finding the overall minimum and maximum values across all three columns
    y_min = Output[['Bitcoin High', 'Bitcoin Low', 'Bitcoin Close']].min().min()
    y_max = Output[['Bitcoin High', 'Bitcoin Low', 'Bitcoin Close']].max().max()
    
    # Adjusting y-axis limits
    ax.set_ylim(y_min - 100, y_max + 100)
    
    # Displaying the plot using Streamlit
    st.pyplot(fig)
    
    # Displaying the DataFrame
    st.write("Forecasted Values include")
    st.dataframe(Output)
        
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
