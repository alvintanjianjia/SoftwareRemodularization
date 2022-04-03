import os
import streamlit as st
import numpy as np

from multipage import MultiPage
from pages import run_pipeline, results # import your pages here

# Create an instance of the app 
app = MultiPage()

# Add all your application here
app.add_page("Run Pipeline", run_pipeline.app)
app.add_page("View Results", results.app)

# The main app
app.run()