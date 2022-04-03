import streamlit as st
import numpy as np
import pandas as pd
import validators
import papermill as pm



def app():

    def execute_notebook():
        st.text("Running..... Please wait....")
        pm.execute_notebook(
                'Pipeline_Tool_Notebook.ipynb',
                'Pipeline_Tool_Notebook_Output.ipynb',
                parameters=dict(github_link=user_github_link)
            )
        return True


    user_github_link = st.text_input('GitHub URL', '')
    if validators.url(user_github_link) is True and 'github' in user_github_link:
        if st.button('Run Tool.'):
            completed = execute_notebook()
            if completed:
                st.text('Completed!')
        
        
        
    else:
        st.info(
            f"""
                👆 Please input a valid Github URL.
                """
        )

        st.stop()