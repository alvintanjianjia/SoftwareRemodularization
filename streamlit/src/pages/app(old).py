from turtle import onclick
import streamlit as st
import pandas as pd
import papermill as pm
import validators

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

###################################

from functionforDownloadButtons import download_button

def execute_notebook():
    # pm.execute_notebook(
    #         'Pipeline_Tool_Notebook.ipynb',
    #         'Pipeline_Tool_Notebook_Output.ipynb',
    #         parameters=dict(github_link=user_github_link)
    #     )
    return

###################################

def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="✂️", page_title="CSV Wrangler")

# st.image("https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/balloon_1f388.png", width=100)
st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/285/scissors_2702-fe0f.png",
    width=100,
)

st.title("Software Refactoring Tool")



c29, c30, c31 = st.columns([1, 6, 1])
shows = pd.DataFrame()
with c30:

    user_github_link = st.text_input('GitHub URL', '')
    #validators.url(user_github_link) is True and 'github' in user_github_link
    if st.button('Run Tool.'):
        shows = execute_notebook()
        st.text("Running..... Please wait....")
        
        
    else:
        st.info(
            f"""
                 👆 Please input a valid Github URL.
                 """
         )

        st.stop()
    # uploaded_file = st.file_uploader(
    #     "",
    #     key="1",
    #     help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    # )

    # if uploaded_file is not None:
    #     file_container = st.expander("Check your uploaded .csv")
    #     shows = pd.read_csv(uploaded_file)
    #     uploaded_file.seek(0)
    #     file_container.write(shows)

    # else:
    #     st.info(
    #         f"""
    #             👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
    #             """
    #     )

    #     st.stop()

from st_aggrid import GridUpdateMode, DataReturnMode
shows = pd.read_csv('pipeline_prediction_output.csv')
gb = GridOptionsBuilder.from_dataframe(shows)
# enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
gridOptions = gb.build()

st.success(
    f"""
        💡 Tip! Hold the shift key when selecting rows to select multiple rows at once!
        """
)

response = AgGrid(
    shows,
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=False,
)

df = pd.DataFrame(response["selected_rows"])


st.subheader("Filtered data will appear below 👇 ")
st.text("")

st.table(df)

st.text("")
try:
    total_predicted_time = df['predicted_time'].sum()
    st.text("Total predicted time: " + str(total_predicted_time))
except:
    st.text("Please select a refactoring operation.")

c29, c30, c31 = st.columns([1, 1, 2])

with c29:

    CSVButton = download_button(
        df,
        "File.csv",
        "Download to CSV",
    )

with c30:
    CSVButton = download_button(
        df,
        "File.csv",
        "Download to TXT",
    )