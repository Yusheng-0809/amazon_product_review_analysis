# streamlit
import streamlit as st

# title
# st.title("title")
# st.header("header")
# st.subheader("subheader")

# text
# st.markdown('''
# # 静夜思
# 床前**明月**光，疑是地上霜。
# 举头望**明月**，低头思故乡。
# ''')

# st.text('''
# 静夜思
# 床前明月光，疑是地上霜。
# 举头望明月，低头思故乡。
# ''')

# st.markdown('**以下为打印的代码：**')

# st.code('''
#         def main():
#             print(1)
        
#         if __name__ == "__main__":
#             main()
#         ''', language='python') 

# 可显示多种数据格式（text, num, list, dict, df, fig, combo）
# st.write('write测试')

# df
# import pandas as pd
# df = pd.DataFrame()
# st.dataframe(df.style.highlight_max(axis=0))
# st.table

# json
# st.json()

# pyplot 类似plt.show()
# st.pyplot

# map
# data = {
#     'latitude': [37.7749, 34.0522, 40.7128],
#     'longitude': [-122.4194, -118.2437, -74.0060],
#     'name': ['San Francisco', 'Los Angeles', 'New York']
# }

# st.map(data, zoom=4, use_container_width=True)

# st.image/st.video

# interaction mode
# button
if st.button("Click here"):
    st.text("Hi, there")

# 确认框
cb = st.checkbox('comfirm', value=False)
if cb:
    st.write('successfully confirmed')
else:
    st.write('unconfirmed')

    import streamlit as st

# 单选框/下拉框/多选框
sex = st.radio( # selectbox, multiselect
    label = '请输入您的性别',
    options = ('男', '女', '保密'), # 注意为tuple
    index = 2, # 默认选项, 在multiselect中为default(iterable)
    format_func = str,
    help = '如果您不想透露，可以选择保密'
    )

if sex == '男':
    st.write('男士您好!')
elif sex == '女':
    st.write('女士您好!')
else:
    st.write('您好!')

# 滑条
age = st.slider(label='请输入您的年龄', 
                min_value=0, 
                max_value=100, 
                value=0, 
                step=1, 
                help="请输入您的年龄")

st.write('您的年龄是', age)

# 输入框 
name = st.text_input('请输入用户名',  max_chars=100, help='最大长度为100字符') # number_input() text_area() date_input() time_input()

st.write('您的用户名是', name)

# 警告信息
st.error('错误信息')
st.warning('警告信息')
st.info('提示信息')
st.success('成功信息')
st.exception('异常信息')

# 程序运行信息
import time
progress_bar = st.empty()

for i in range(10):
    progress_bar.progress(i / 10, '进度')
    time.sleep(0.5)

with st.spinner('加载中...'):
    time.sleep(2)

# 缓存机制
# import time 
# import pandas as pd

# @st.cache_data
# def fetch_data(url):
#     time.sleep(5)
#     return pd.read_csv(url)

# url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
# d1 = fetch_data(url1)
# st.write(d1)

# url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
# d2 = fetch_data(url2)
# st.write(d2)

# st.cache_data.clear()

# url3 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
# d3 = fetch_data(url3)
# st.write(d3)