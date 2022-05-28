import streamlit as st
import pandas as pd
import sys
from results_pickler import ResultPickler

sys.path.insert(0, '../scripts')

def app():

    # Load Saved Results Data
    results = ResultPickler()
    results.load_data(file_name='./data/exploration_info.pkl')
    results = results.get_data()

    st.title("Data Analysis Part I")

    st.header("Distribution of Promotion on Train and Test Data")
    st.subheader("Percentage")
    st.image('./data/comppromodist.png')

    st.subheader("Month Count")
    st.image('./data/comppromodist2.png')

    st.header("Sales Behaviours")
    st.subheader("Week Days")
    st.image('./data/dayofweeksales.png')

    st.subheader("Month")
    st.image('./data/monthpromosales.png')

    st.subheader("Holiday Time")
    st.image('./data/holidaytiming.png')

    st.subheader("Holiday Season")
    st.image('./data/holidayseason.png')

    st.header("Data Relations")
    st.subheader("Sales And Customers")
    st.image('./data/salecustrln.png')

    st.subheader("Sales and Promotion")
    st.image('./data/salepromorln.png')

    st.subheader("Customers and Promotion")
    st.image('./data/custpromo.png')

    st.header("Promotion Factor")
    st.subheader("Impact on Customer Based on Day of Week")
    st.image('./data/monthpromocat.png')

    st.subheader(
        "The average customer increase across all stores due to promotion is by: ")
    st.text('{:.2%}'.format(results['averageincreaseacross']))

    # st.subheader(
    #     "Customer Percentage Increase")
    # st.dataframe(results["cuspercincrease"])

    # st.subheader("Top 10 Stores")
    # st.dataframe(results["top10promocust"])

    st.subheader("Customers Increase in Stores")
    st.image('./data/cuspercincreasediag.png')

    st.header("Sales Based on Stores")
    st.subheader("Assortment Based")
    st.image('./data/assortmenttype.png')

    st.subheader("Store Type")
    st.image('./data/storetpye.png')

    st.header("Profitablity Of Promotion Based on Customers")
    # st.subheader("> 10% Increase in Customers")
    # st.dataframe(results["10percincrease"])

    # st.subheader("> 60% Increase in Customers")
    # st.dataframe(results["60percincrease"])

    # st.subheader("> 90% Increase in Customers")
    # st.dataframe(results["90percincrease"])

    st.header("Sales Difference Between Store Based on their Opening Schedule")
    st.subheader("All WeekDay vs Not All WeekDay")
    st.image('./data/alldaycomp.png')

    st.header("Difference Between Store Based on their Assortment")
    st.subheader("Assortments Comparisson in Sales and Customers")
    st.image('./data/storeassortmentclass.png')

    st.header("Relation Between Store and Competition")
    st.subheader("With Sales and Customers")
    st.image('./data/salescustcomprln.png')

    st.header("Change in Stores After Competiting Store Opened")
    # st.subheader("Change in Sales and Customers of each store")
    # st.dataframe(results["newcompcomingeffect"])

    # st.subheader("Stores with Decreasing Sales")
    # st.dataframe(results["salesdecrease"])

    # st.subheader("Stores with Decreasing Customers")
    # st.dataframe(results["custdecrease"])

    # st.subheader("Stores with Increasing Sales")
    # st.dataframe(results["salesincrease"])

    # st.subheader("Stores with Increasing Customers")
    # st.dataframe(results["custincrease"])
