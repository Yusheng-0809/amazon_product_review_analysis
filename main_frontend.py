import streamlit as st
from main_analysis import read_data, preprocess_data, pre_analysis, get_statistics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

data_len = 20

st.title("AI-driven Review Analyzer")

# restrict file type and create a temporary path
uploaded_file_path = st.file_uploader("Import an Excel file here. Ensure the file contains the columns, '标题' and '内容'.", type='xlsx')

if uploaded_file_path is not None:
    data = read_data(uploaded_file_path)
    if isinstance(data, pd.DataFrame):
        st.write(data)
        # check data length
        if len(data) >= data_len:
            if st.button('Begin Analysis'):
                try:
                    begin_time = time.time()
                    with st.spinner('Analyzing...every 30 reviews takes one minute or so.'):
                        data_processed = preprocess_data(data)
                        data_result, summary_for_reviews = pre_analysis(data_processed)
                        statistical_result = get_statistics(data_result)
                    end_time = time.time()
                    elapsed_time = np.round(end_time - begin_time)
                    st.success(f'Analysis done, taking about {elapsed_time} seconds. The results are displayed below.')
                except Exception as e:
                    statistical_result = None
                    st.error(f'Analysis failed: {e}')

                def sort_dict(count_dict, with_example=False):
                    if with_example:
                        sorted_count_dict = dict(sorted(count_dict.items(), key=lambda item: item[1]['number'], reverse=True))
                        df = pd.DataFrame.from_dict(sorted_count_dict, orient='index')
                        df.reset_index(inplace=True)
                        df.columns = ['key', 'number', 'example']
                        df.insert(loc=2, column='ratio', value=(df['number']/df['number'].sum()).round(2))
                        return df
                    else:
                        df = pd.DataFrame(sorted(count_dict.items(), key=lambda x: x[1], reverse=True), columns = ['key', 'number'])
                        df.insert(loc=2, column='ratio', value=(df['number']/ df['number'].sum()).round(2))
                        return df

                def get_plot(name, data):
                    fig, ax = plt.subplots()
                    ax.bar(data['key'], data['number'], color='skyblue', edgecolor='black')
                    ax.set_xticklabels(data['key'], rotation=45)
                    ax.set_title(name)
                    fig.subplots_adjust(bottom=0.15)
                    return fig

                # rearrange the data in descending order and display them in pictures
                if isinstance(statistical_result, list):
                    purchase_time_cluster = sort_dict(statistical_result[0], with_example=True)
                    scenario_cluster = sort_dict(statistical_result[1], with_example=True)
                    user_cluster_counts = sort_dict(statistical_result[2])
                    motive_cluster_counts = sort_dict(statistical_result[3])
                    channel = sort_dict(statistical_result[4])
                    evaluation = sort_dict(statistical_result[5])
                    reason_point_positive = sort_dict(statistical_result[6])
                    reason_point_negative = sort_dict(statistical_result[7])
                    focus_cluster_counts = sort_dict(statistical_result[8])
                    statistical_result_new = {"Purchase Time": purchase_time_cluster,
                                              "Usage Scenario": scenario_cluster,
                                              "Target User": user_cluster_counts,
                                              "Purchase Motive": motive_cluster_counts,
                                              "Purchase Channel": channel,
                                              "Review Emotion": evaluation,
                                              "Why Positive": reason_point_positive,
                                              "Why Negative": reason_point_negative,
                                              "What Customers Care": focus_cluster_counts}

                    for name, table in statistical_result_new.items():
                        row = st.columns(2)
                        with row[0]:
                            st.write(table)
                        with row[1]:
                            ax = get_plot(name, table)
                            st.pyplot(ax)
        else:
            st.error(f'Data not sufficient (length: {len(data)}) for analysis')
    else:
        st.error('Imported data is not valid')