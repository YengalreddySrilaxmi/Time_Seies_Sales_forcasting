import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar

# Function to convert date format


# Function to convert date format
def convert_date_format(date_str):
    try:
        date_obj = pd.to_datetime(date_str, format='%d/%m/%y', errors='coerce')
        if pd.notna(date_obj):
            return date_obj.strftime('%d-%m-%Y')
        else:
            return pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce').strftime('%d-%m-%Y')
    except Exception:
        return None


# Function for data preprocessing and feature engineering
def preprocess_and_create_features(data):
    # Convert date columns to consistent formats
    data['order_date'] = data['order_date'].apply(convert_date_format)
    data['order_date'] = pd.to_datetime(data['order_date'], format='%d-%m-%Y', errors='coerce')

    # Extract date-related features
    data['day'] = data['order_date'].dt.day
    data['month'] = data['order_date'].dt.month
    data['year'] = data['order_date'].dt.year
    data['day_of_week'] = data['order_date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Handle outliers in sales data
    percentile_99 = data['sales_$'].quantile(0.99)
    
    # Print the 99th percentile and number of rows after removing outliers
    print(f"99th Percentile of Sales: {percentile_99}")
    
    cleaned_data = data[data['sales_$'] <= percentile_99]
    
    print(f"Number of rows after removing values above the 99th percentile: {cleaned_data.shape[0]}")

    # Group data by relevant columns
    return cleaned_data
def sales(data):
    data=preprocess_and_create_features(data)
    sales_data = data.groupby(['order_date', 'category', 'sub_category', 'state'])[['sales_$']].sum().reset_index()
    # Extracting month, day, and year from the order_date
    sales_data['month'] = sales_data['order_date'].dt.month
    sales_data['day'] = sales_data['order_date'].dt.day
    sales_data['year'] = sales_data['order_date'].dt.year
    sales_data['day_of_week'] = sales_data['order_date'].dt.dayofweek
    sales_data['is_weekend'] = sales_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    # Create lag features
    lags = [1, 7, 15, 30, 90, 180, 270, 365]
    for lag in lags:
        sales_data[f'sales_lag_{lag}'] = sales_data.groupby('category')['sales_$'].shift(lag)

    # Handle missing values in lag features
    sales_data.fillna({'sales_lag_1': 0, 'sales_lag_7': 0, 'sales_lag_30': 0, 'sales_lag_365': 0,'sales_lag_15': 0,'sales_lag_30': 0,'sales_lag_180': 0,'sales_lag_270': 0,'sales_lag_90': 0}, inplace=True)
    # Create LFL growth feature
    sales_data['LFL_growth'] = (sales_data['sales_$'] - sales_data['sales_lag_365']) / sales_data['sales_lag_365'] * 100
    sales_data['LFL_growth'] = sales_data['LFL_growth'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Create moving averages and exponential moving averages
    # List of moving average windows
    windows = [3, 7, 15, 30, 90, 180,270,365]

# Adding moving averages
    for window in windows:
        sales_data[f'moving_avg_{window}'] = sales_data.groupby('category')['sales_$'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())

# Adding 30-day moving average for the same period last year
    sales_data['moving_avg_30_last_year'] = sales_data.groupby('category')['sales_$'].transform(lambda x: x.shift(365).rolling(window=30, min_periods=1).mean())
# Handling NaN values in moving averages columns
    moving_avg_columns = [
    'moving_avg_3', 'moving_avg_7', 'moving_avg_15', 'moving_avg_30',
    'moving_avg_90', 'moving_avg_180', 'moving_avg_365', 'moving_avg_270',
    'moving_avg_30_last_year'
    ]
    sales_data[moving_avg_columns] = sales_data[moving_avg_columns].fillna(0)


    spans = [3, 7, 15, 30, 90, 180, 270, 365]
    for span in spans:
        sales_data[f'ema_{span}'] = sales_data.groupby('category')['sales_$'].transform(
            lambda x: x.shift(1).ewm(span=span, adjust=False).mean())
    
    for span in spans:
        sales_data[f'ema_{span}'].fillna(0, inplace=True)
    # Add holiday and month-related features
    # Adding day of the week indicators
    sales_data['day_of_week'] = sales_data['order_date'].dt.dayofweek
    sales_data = pd.get_dummies(sales_data, columns=['day_of_week'], prefix='day_of_week', drop_first=True)

# Adding month and day indicators
    sales_data['month'] = sales_data['order_date'].dt.month
    sales_data['day'] = sales_data['order_date'].dt.day
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=sales_data['order_date'].min(), end=sales_data['order_date'].max())
    sales_data['is_holiday'] = sales_data['order_date'].isin(holidays).astype(int)
    sales_data['start_of_month'] = sales_data['order_date'].dt.is_month_start.astype(int)
    sales_data['end_of_month'] = sales_data['order_date'].dt.is_month_end.astype(int)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    sales_data['category'] = label_encoder.fit_transform(sales_data['category'])
    sales_data['state'] = label_encoder.fit_transform(sales_data['state'])
    sales_data['sub_category'] = label_encoder.fit_transform(sales_data['sub_category'])

    return sales_data


# Train and evaluate model
def train_model_and_evaluate(data):
    sales_data = sales(data)
    split_date = '2018-09-01'

    train_data = sales_data[sales_data['order_date'] < split_date]
    test_data = sales_data[sales_data['order_date'] >= split_date]

    train = train_data[['day_of_week_2', 'category', 'moving_avg_180', 'ema_3', 'moving_avg_365', 'ema_15',
                        'ema_180', 'moving_avg_90', 'sales_lag_1', 'sales_lag_365', 'moving_avg_15', 'LFL_growth',
                        'sub_category', 'day_of_week_4', 'sales_$']].values
    test = test_data[['day_of_week_2', 'category', 'moving_avg_180', 'ema_3', 'moving_avg_365', 'ema_15',
                      'ema_180', 'moving_avg_90', 'sales_lag_1', 'sales_lag_365', 'moving_avg_15', 'LFL_growth',
                      'sub_category', 'day_of_week_4', 'sales_$']].values

    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.2, max_depth=6)
    model.fit(trainX, trainy)

    return model, sales_data


def predict_and_update(data, category, category_name, days_to_predict):
    model, sales_data = train_model_and_evaluate(data)
    split_date = '2018-09-01'

    train_data = sales_data[sales_data['order_date'] < split_date]
    test_data = sales_data[sales_data['order_date'] >= split_date]
    category_test_df = test_data[test_data['category'] == category][[
        'day_of_week_2', 'category', 'moving_avg_180', 'ema_3', 'moving_avg_365',
        'ema_15', 'ema_180', 'moving_avg_90', 'sales_lag_1', 'sales_lag_365',
        'moving_avg_15', 'LFL_growth', 'sub_category', 'day_of_week_4', 'sales_$'
    ]]

    # Map for current day of the week to column name
    day_of_week_columns = ['day_of_week_2', 'day_of_week_4']
    day_of_week_map = {0: 'day_of_week_2', 3: 'day_of_week_4'}
    inverse_day_of_week_map = {v: k for k, v in day_of_week_map.items()}

    predictions = []

    # Check if the subset is not empty before trying to access the last row
    if category_test_df.empty:
        raise ValueError(f"No data available for category {category_name} in the test set.")

    current_day_of_week_col = category_test_df[day_of_week_columns].iloc[-1].idxmax()
    # Set a default day of the week
    current_day_of_week_num = inverse_day_of_week_map[current_day_of_week_col]

    for day in range(1, days_to_predict + 1):
        current_test_data = category_test_df.iloc[-1, :-1].values
        current_pred = model.predict(current_test_data.reshape(1, -1))[0]

        if current_pred < 0:
            current_pred = 0

        predictions.append(current_pred)

        new_row = category_test_df.iloc[-1].copy()
        new_row['sales_$'] = current_pred

        # Update the day of the week
        current_day_of_week_num = (current_day_of_week_num + 1) % 7
        
        new_row[day_of_week_columns] = 0

        # Use pd.concat instead of append
        if current_day_of_week_num in day_of_week_map:
            new_row[day_of_week_map[current_day_of_week_num]] = 1

        # Append the new row to the DataFrame
        category_test_df = pd.concat([category_test_df, pd.DataFrame([new_row])], ignore_index=True)

        # Recalculate the lag features after appending the new data
        category_test_df['sales_lag_1'] = category_test_df['sales_$'].shift(1)
        category_test_df['sales_lag_365'] = category_test_df['sales_$'].shift(365) if len(category_test_df) > 365 else np.nan

        # Recalculate moving averages based on the newly appended data
        category_test_df['moving_avg_180'] = category_test_df['sales_$'].rolling(window=180).mean() if len(category_test_df) >= 180 else np.nan
        category_test_df['moving_avg_90'] = category_test_df['sales_$'].rolling(window=90).mean() if len(category_test_df) >= 90 else np.nan
        category_test_df['moving_avg_15'] = category_test_df['sales_$'].rolling(window=15).mean() if len(category_test_df) >= 15 else np.nan
        category_test_df['moving_avg_365'] = category_test_df['sales_$'].rolling(window=365).mean() if len(category_test_df) >= 365 else np.nan

        # Recalculate exponential moving averages (EMAs) based on the newly appended data
        category_test_df['ema_3'] = category_test_df['sales_$'].ewm(span=3).mean() if len(category_test_df) >= 3 else np.nan
        category_test_df['ema_15'] = category_test_df['sales_$'].ewm(span=15).mean() if len(category_test_df) >= 15 else np.nan
        category_test_df['ema_180'] = category_test_df['sales_$'].ewm(span=180).mean() if len(category_test_df) >= 180 else np.nan

    return predictions
def visualize_sales(data):
    st.title("Sales Data Analysis")

    categories = data['category'].unique()

    # Plotting monthly sales trends for each category
    for cat in categories:
        category_data = data[data['category'] == cat]
        monthly_sales = category_data.groupby(['year', 'month'])['sales_$'].sum().unstack(level=0)

        st.subheader(f"Monthly Sales Trends for {cat}")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define month names for the x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for year in monthly_sales.columns:
            ax.plot(monthly_sales.index, monthly_sales[year], marker='o', label=f'Year {year}')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('sales_$')
        ax.set_title(f'Monthly Sales Trends for {cat}')
        ax.set_xticks(range(1, 13))  # Set the x-ticks for months
        ax.set_xticklabels(month_names)  # Label months with names
        ax.legend(title='Year')
        ax.grid(True)
        st.pyplot(fig)

    # Plotting daily sales trends for each category
    for cat in categories:
        category_data = data[data['category'] == cat]
        years = category_data['year'].unique()
        
        st.subheader(f"Daily Sales Trends for {cat}")
        fig, ax = plt.subplots(figsize=(10, 6))
        for year in years:
            yearly_data = category_data[category_data['year'] == year]
            daily_sales = yearly_data.groupby('day')['sales_$'].sum()
            ax.plot(daily_sales.index, daily_sales, marker='o', label=f'Year {year}')
        ax.set_xlabel('Date')
        ax.set_ylabel('sales_$')
        ax.set_title(f'Daily Sales Trends for {cat}')
        ax.legend(title='Year')
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
from PIL import Image

# Example content
# Streamlit interface
st.title("Time Series Sales Forecasting")

    # File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview", data.head())
        
    if st.button("Visualize Sales Trends"):
        visualize_sales(preprocess_and_create_features(data))
        
    if st.button("Train Model"):
        model = train_model_and_evaluate(data)
        st.write("Model Trained Successfully!")

    st.subheader("Predict Sales")
        
        # Mapping category names to their respective numerical labels
    category_map = {
            "Furniture": 0,
            "Technology": 1,
            "Office": 2
    }
        
        # Sales data for category selection
    sales_data = data  # Assuming `sales_data` is simply `data` here
        
    category = st.selectbox("Select Category", list(category_map.keys()))
    days_to_predict = st.number_input("Enter Number of Days to Predict", min_value=1, max_value=365, value=30)

    if st.button("Generate Predictions"):
        category_label = category_map[category]  # Get the category label from the selected category
        sales_data = preprocess_and_create_features(data)
        predictions = predict_and_update(sales_data, category_label, category, days_to_predict)
            
        st.write(f"Next {days_to_predict} Predictions:", predictions)

            # Visualize the predictions
        # Visualize the predictions
        plt.plot(predictions)
        plt.title(f"Predicted Sales for Category {category}")
        plt.xlabel("Days")
        plt.ylabel("Sales")
        st.pyplot(plt)

# Main function to control the flow of pages


