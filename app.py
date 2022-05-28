import sys
sys.path.insert(0, './scripts')
from pages import EDA, EDA2, model_implementation
from multiapp import MultiApp
import streamlit as st

# import your app modules here

st.set_page_config(
    page_title="Rossmann Pharmaceuticals Sales Prediction Across Multiple Stores", layout="wide")

app = MultiApp()


st.sidebar.markdown("""
# Rossmann Pharmaceuticals Store Sales
### Multi-Page App
This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed by [Praneel Nihar](https://medium.com/@u.praneel.nihar). Also check out his [Medium article](https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4).
### Modifications
\t- Page Folder Based Access
\t- Presentation changed to SideBar
""")

# Add all your application here
app.add_app("Data Exploration and Analysis, PART I", EDA.app)
app.add_app("Data Exploration and Analysis, PART II", EDA2.app)
app.add_app("Sales Prediciton", model_implementation.app)

# The main app
app.run()
