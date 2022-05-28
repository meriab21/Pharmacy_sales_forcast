from scripts.data_manipulation import DataManipulator
from scripts.data_cleaner import DataCleaner
import streamlit as st
import mlflow
from pickle import load
import pandas as pd
import numpy as np
import sys
import datetime
import base64

from scripts.results_pickler import ResultPickler
from scripts.data_loader import load_df_from_csv

def read_csv_without_index(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print('File Not Found')

def get_model_columns():
    # Create Result Pickler Object
    results = ResultPickler()
    results.load_data('./models/column_reference.pkl')

    return results.get_data()['model_input_columns']

def import_model(columns=None):
    '''Import from mlflow if possible or get saved model'''
    try:
        if(columns != None):
            print('mlflow')
            model = 'runs:/2d6250149bd746ab84c41372792902b4/model'
            # Load model as a PyFuncModel.
            model = mlflow.pyfunc.load_model(model)
            # data = pd.DataFrame(columns=columns)
            # a_series = pd.Series(a["data"], index=data.columns)
            # data = data.append(a_series, ignore_index=True)
        else:
            with open('./models/01-08-2021-21-23-15-74.17%.pkl', 'rb') as handle:
                model = load(handle)
        
        return model
            # model.predict([a['data']])

    except Exception as e:
        print('Failed to load model', e)

def dayofweek(data):
    values = []
    try:
        for index, row in data.iterrows():
            day = datetime.date(row['Year'],row['Month'],row['Day']).weekday()
            values.append(day + 1)
        data['DayOfWeek'] = values

    except:
        print("Failed to create day of week")

def get_season(month):
    if(month <= 2 or month == 12):
        return 2
    elif(month > 2 and month <= 5):
        return 1
    elif(month > 5 and month <= 8):
        return 0
    else:
        return 3

def add_season(data):
    date_index = data.columns.get_loc("Month")
    data.insert(date_index + 1, 'Season',data['Month'].apply(get_season))


def encode_holiday(x):
    if x == 'Public Holiday':
        return 1
    elif x == 'Easter':
        return 2
    elif x == 'Christmas':
        return 3
    else:
        return 0

def change_holiday(data):
    data['StateHoliday'] = data['StateHoliday'].apply(encode_holiday)

def day_to_after_holiday(data, holidays):
    days_to = []
    days_after = []
    holidays = holidays.sort_values(by=['Month','Day'], ascending='True')

    for index, row1 in data.iterrows():
        lower_month = 12
        lower_day = 26
        actual_month = 0
        month = row1['Month']
        day = row1['Day']
        for index, row in holidays.iterrows():
            if(month >= row['Month'] and day > row['Day']):
                lower_month = row["Month"]
                lower_day = row['Day']
            elif(month > row['Month']):
                lower_month = row["Month"]
                lower_day = row['Day']
            elif(month <= row['Month'] and day < row['Day']):
                actual_month = row['Month']
                actual_day = row['Day']
                break
                
        if(lower_month == 12):
            date1 = datetime.date(2009, lower_month, lower_day)
            date2 = datetime.date(2010, month, day)
            date3 = datetime.date(2010, actual_month, actual_day)
        else:
            date1 = datetime.date(2010, lower_month, lower_day)
            date2 = datetime.date(2010, month, day)
            date3 = datetime.date(2010, actual_month, actual_day)

        days_to_holiday = date2 - date1
        days_after_holiday = date3 - date2
        
        days_to.append(days_to_holiday.days)
        days_after.append(days_after_holiday.days)

    data['DaysToHoliday'] = days_to
    data['DaysAfterHoliday'] = days_after

def create_additional_datas(data):
    # Load Holiday References
    holiday_reference = load_df_from_csv('./models/holiday_reference.csv')

    # Classifiy Date
    data_cleaner = DataCleaner(data)
    data_cleaner.change_column_to_date_type('Date')
    data_cleaner.separate_date_column(date_column='Date')

    # Create Day of Week
    dayofweek(data)

    # Create WeekDay
    data_manipulator = DataManipulator(data)
    data_manipulator.add_week_day('DayOfWeek')

    # Create Season
    add_season(data)

    # change holiday
    change_holiday(data)

    # Create Month Timing
    data_manipulator.add_month_timing('Day')

    # Create Days to and after holiday
    day_to_after_holiday(data, holiday_reference)

    data = data[["Store",'DayOfWeek', 'WeekDay', 'Year', 'Month', 'Season', 'Day', 'MonthTiming', 'Open', 'Promo', 'StateHoliday', "DaysAfterHoliday","DaysToHoliday", "SchoolHoliday"]]

    return data

def add_store_value(data:pd.DataFrame, store_reference):
    final_dataframe = pd.merge(data, store_reference, on='Store')

    return final_dataframe

def get_actual_sale(value, prev_min, prev_max, new_min, new_max):
    actual_sale = ((value - prev_min) / (prev_max - prev_min)) * (new_max - new_min) + new_min
    return actual_sale

def predict(model, data, columns):
    result_df = data[['Store','Year','Month','Day']]
    data.drop('Store', axis=1, inplace=True)
    data.columns = columns
    predictions = []
    for index, row in data.iterrows():
        prediction = model.predict([data.iloc[index,:].values.tolist()])
        predictions.append(get_actual_sale(prediction[0], -1.5, 9.3, 0, 41551))
    
    result_df['Predicted Sales'] = predictions

    return result_df


def download_button(df):
    # if no filename is given, a string is returned
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="Store Predicitons.csv">Download Predicitions CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():

    # Load model column input reference
    model_columns = get_model_columns()

    # Import Model for the predicition
    model = import_model()

    # Load Store References
    store_reference = load_df_from_csv('./models/store_reference.csv')

    # Load Saved Results Data
    # model = load(filename='./models/satisfaction_scorer_model.pkl')

    st.title("Store Sales Predictor")

    method = st.radio("Options", ('Upload File', 'Manual'))

    if(method == 'Upload File'):
        test_file = st.file_uploader("Upload csv files", type=['csv'])
        test_csv = None

        if (test_file):
            test_csv = read_csv_without_index(test_file)
            st.dataframe(test_csv)

            if st.button('Predict'):
                st.write("Predict clicked")
                train_data = create_additional_datas(test_csv)
                st.dataframe(train_data)
                final_data = add_store_value(train_data, store_reference)
                st.dataframe(final_data)

                prediction = predict(model, final_data, model_columns)
                st.dataframe(prediction)

                download_button(prediction)

                
    else:

        store_id = int(st.selectbox(
            "Store Id", [i for i in range(1, 1116)]))
        # DayOfWeek	Date	Open	Promo	StateHoliday	SchoolHo
        date = st.date_input('Sale date')
        # sate_hoilday = st.selectbox(
        #     "Sate Holiday", list(holidays.keys()))
        state_hoilday = st.selectbox("Sate Holiday", ["Public Holiday", "Easter", "Christmas", "None"])
        school_holiday = int(st.checkbox("School Holiday"))
        promo = int(st.checkbox("Promotion Running"))
        is_open = int(st.checkbox("Store Open"))

        if st.button('Predict'):
            st.write('predict button clicked')
            # Create dataframe with the values
            data = {'Store': [store_id], 'Date': [date], 'StateHoliday': [state_hoilday],'SchoolHoliday': [school_holiday], 'Promo':[promo], 'Open':[is_open]}
            initial_data = pd.DataFrame(data=data)
            st.dataframe(initial_data)
            train_data = create_additional_datas(initial_data)
            st.dataframe(train_data)
            final_data = add_store_value(train_data, store_reference)
            st.dataframe(final_data)

            prediction = predict(model, final_data, model_columns)
            st.dataframe(prediction)

