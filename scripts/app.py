import pandas as pd
import streamlit as st
import holiday 
import bisect

# from streamlit_gallery.utils import readme
import pickle

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
class CustomMaxImputer(BaseEstimator,TransformerMixin):
    def fit(self, X, y=0):
        self.fill_value  = X.max()
        return self
    def transform(self, X,y=0):
        return np.where(X.isna(), self.fill_value, X)

@st.cache(allow_output_mutation=True)
def loadModel():
    file = open("model2.pkl",'rb')
    model = pickle.load(file)
    return model
@st.cache()
def load_store_data():
    store_df=pd.read_csv('store_cleaned.csv')

    return store_df
# s.dt.dayofweek
def read_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print("file read as csv")
        return df
    except FileNotFoundError:
        print("file not found")

def read_csv_without_index(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print("file read as csv")
        return df
    except FileNotFoundError:
        print("file not found")
def datToAndAfterHoliday(df,Column,holidays):
    
    to=[]
    after=[]
    for a in df[Column]:
        index=bisect.bisect(holidays,a)
        if len(holidays)==index:
            to.append(pd.Timedelta(0, unit='d') )
            after.append(a - holidays[index-1])
        else:
            after.append(holidays[index] - a)
            to.append(a -holidays[index-1])
    return to,after
def startMidEndMonth(x):
    if x<10:
        return 0
    elif x<20:
        return 1
    else:
        return 2
def isWeekend(x):
    if x<6:
        return 0
    else: 
        return 1
def dateExplode(df,column):
    try:
        df['Year'] = pd.DatetimeIndex(df[column]).year
        df['Month'] = pd.DatetimeIndex(df[column]).month
        df['Day'] = pd.DatetimeIndex(df[column]).day  
    except KeyError:
        print("Column couldn't be found")
        return
    return  df
    
def generate_features(df):
    
    df["Date"]=pd.to_datetime(df["Date"])
    
    df["weekend"]= df["DayOfWeek"].apply(isWeekend )
    df["MonthState"]=df["Day"].apply(startMidEndMonth)
    with open('dates.pickle', 'rb') as handle:
        dates = pickle.load(handle)
    df["To"],df["After"]=datToAndAfterHoliday(df,"Date",dates)
    
    df["After"]=pd.to_timedelta(df["After"])
    
    df["To"]=pd.to_timedelta(df["To"])

    df["After"]=pd.to_numeric(df['After'].dt.days, downcast='integer')
    df["To"]=pd.to_numeric(df['To'].dt.days, downcast='integer')

    return  df
    
def merge_store(df):
    store=read_csv("store_cleaned.csv")
    combined = pd.merge(df, store, on=["Store"])
    return combined

def predict(model,csv):
    csv_copy=csv.copy()
    csv_copy.drop("Store",axis=1,inplace=True)
    
    csv_copy.drop("Id",axis=1,inplace=True)
    
    csv_copy.drop("Date",axis=1,inplace=True)
    print(csv_copy.columns)
    prediction=model.predict(csv_copy)
    
    pred_df = csv.copy()

    pred_df["Sales-Prediction"] = prediction
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    return pred_df

def man_predict(model,csv):
    
    csv_copy=csv.copy()
    csv_copy.drop("Store",axis=1,inplace=True)
    
    
    csv_copy.drop("Date",axis=1,inplace=True)
    
    prediction=model.predict(csv_copy)
    pred_df = csv.copy()

    pred_df["Sales-Prediction"] = prediction
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    return pred_df

def main():
    dataset = "train.csv"

    df = load_store_data()
    # pr = gen_profile_report(df, explorative=True)
    model=loadModel()
    st.write(f"🔗Preictions based on  [Rossman Sales]({dataset})")
    st.markdown(f"### Sales Prediction")
    method = st.radio("method", ('Upload file', 'Manual'))
    if (method == "Upload file"):
        test_file = st.file_uploader("Upload csv files", type=['csv'])
        test_csv = None
        if (test_file):
            test_csv = read_csv_without_index(test_file)
            # st.write(test_csv)

            if st.button('Predict'):
                dateExplode(test_csv,column="Date")
                test_store=merge_store(test_csv)
                test_store=generate_features(test_store)
                # st.write(test_csv)
                prediction=predict(model,test_store)
                st.write(prediction)
    else:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        months = [i for i in range(12)]

        holidays = {
            "No Holiday": '0', "Public Holiday": "a", "Easter": "b", "Christmas": "c"}
        school_holiday = [0, 1]

        store_id = int(st.selectbox(
            "Store Id",df["Store"].unique()))
        # DayOfWeek	Date	Open	Promo	StateHoliday	SchoolHo
        date = st.date_input('Sale date')
        sate_hoilday = st.selectbox(
            "Sate Holiday", list(holidays.keys()))
        school_holiday = int(st.checkbox("School Holiday"))
        school_promo = int(st.checkbox("Promotion Running"))
        is_open = int(not st.checkbox("Store Closed"))

        if st.button('Predict'):
            man_test_data = pd.DataFrame()
            man_test_data["Store"] = [1]
            man_test_data["Date"] = [pd.to_datetime(date)]
            man_test_data["DayOfWeek"] = man_test_data.Date.dt.dayofweek + 1

            man_test_data["Open"] = [is_open]
            man_test_data["Promo"] = [school_promo]
            
            man_test_data["StateHoliday"] = [holidays[sate_hoilday]]
            man_test_data["SchoolHoliday"] = [school_holiday]

            man_test_data=dateExplode(man_test_data,"Date")
            man_test_data=merge_store(man_test_data)
            man_test_data=generate_features(man_test_data)
            print(man_test_data.columns)
            st.markdown("#### Data")
            st.write(man_test_data)
            st.markdown("#### Prediction")
            pred = man_predict( model,man_test_data)
            
            st.write(f"{pred['Sales-Prediction'].to_list()[0]:.2f}")

    # st.write(df)
    



    


@st.cache(allow_output_mutation=True)
def gen_profile_report(df, *report_args, **report_kwargs):
    return df.profile_report(*report_args, **report_kwargs)


if __name__ == "__main__":
    main()