"""
    # 需求/UI功能：
    ## 上传按钮（每次上传都重新运行分析程序，带分析条）
    ## 主要展示分析数据量、用户画像、购买动机、好评差评集中点（带示例）、退货评论集中点（带示例）
    ## 支持时间调整（或者其他）
    ## 分析文件（excel、pdf）可以导出并可以重新导入分析

    # 正式程序
    ## 后端程序：带时间戳的分析结果（json）、返回分析图
    ## 前端程序：导入键、时间可调节、展示图、进度条、可以下载分析文件（excel、pdf）、接受已分析文件再导入
    ## 云端：配置环境并进行一定程度的压力测试
    ## 其他元素：并发处理、性能监测、错误记录（超时操作）和处理等、类型、注释、配置化、日志
"""

"""
    Author: Eason Liu
    Date: 2024.10.7
    Purpose: to deliver the analysis on customers' reviews in the way of images and figures (BACK END)
"""

import string
import time
import json
import configparser
import pandas as pd
import numpy as np
from openai import OpenAI
from collections import Counter
from threading import Thread
from queue import Queue
from fast_langdetect import detect
from translate import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# load models and parameters
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config.get('DEFAULT', 'ApiKey')
base_url = config.get('DEFAULT', 'BaseURL')
llm = config.get('DEFAULT', 'LLM')
temperature_ER = config.getfloat('DEFAULT', 'TemperatureER')
temperature_SUM = config.getfloat('DEFAULT', 'TemperatureSUM')
temperature_match = config.getfloat('DEFAULT', 'TemperatureMatch')
temperature_CLU1 = config.getfloat('DEFAULT', 'TemperatureCLU1')
temperatureCLU2 = config.getfloat('DEFAULT', 'TemperatureCLU2')

translator = Translator(from_lang='autodetect', to_lang='en')
stop_words = set(stopwords.words('english'))
client = OpenAI(api_key=api_key, base_url=base_url).chat.completions


def read_data(file_path:str):
    """
    Only accept docx file from local directory as input.
    The file must have certain columns. Return warning otherwise.
    """
    # read file
    try:
        data = pd.read_excel(file_path)
    except Exception as e:
        print(f'error out while reading the file: {e}') #### logging
        data = None
        return data

    # extract needed columns
    try:
        data = data[['标题','内容']]
        data.columns = ['title', 'review']
    except:
        print('error out while extracting needed columns')  #### logging
        data = None
        return data

    return data


def preprocess_data(data:pd.DataFrame):
    """
    This module applies to preprocess the input review
    by converting any non-English language to English and removing invalid data (empty or non-character)
    """

    begin_time = time.time()

    data = data.copy()
    
    # remove empty rows
    not_na_idx = data["review"].notna()
    data_not_na = data.loc[not_na_idx]
    data_not_na.reset_index(drop=True, inplace=True)

    # batch language type detection and translation
    ## define the functions
    def detect_with_trial(text):
        try:
            text_lang_type = detect(text.replace("\n", " "))['lang']
        except:
            text_lang_type = None
        return text_lang_type
    
    def translate_with_trial(text):
        text_trans = translator.translate(text)
        if text_trans == "PLEASE SELECT TWO DISTINCT LANGUAGES":
            return text
        else:
            return text_trans

    def main(text):
        text_type = detect_with_trial(text)
        if text_type == 'en':
            return text
        elif text_type is None:
            return None
        else:
            return translate_with_trial(text)
    
    def threaded_translate(texts):
        result_queue = Queue()

        def worker(index, text):
            translated_text = main(text)
            result_queue.put((index, translated_text)) # set index for retrieving specifc threading results with no need for threading lock

        threads = []
        for index, text in enumerate(texts):
            thread = Thread(target=worker, args=(index, text))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        translated_texts = [None] * len(texts)
        while not result_queue.empty():
            index, translated_text = result_queue.get()
            translated_texts[index] = translated_text
        
        return translated_texts

    ## threaded-process the input
    review_list = data_not_na['review'].to_list()
    review_trans_list = threaded_translate(review_list)
    data_not_na['review'] = review_trans_list

    ## remove invalid title and reviews
    data_final = data_not_na.dropna(subset=['review'], ignore_index=True, inplace=False)
    
    # log the elapsed time #### logging
    end_time = time.time()
    elapsed_time = np.round(end_time - begin_time)
    print(f'preprocessing took {elapsed_time} seconds')

    return data_final


# pre_analysis
def pre_analysis(data:pd.DataFrame):
    """
    The module is articulated to perform the entity-recognition task and other relevant LLM-needy tasks in preparation
    """

    data = data.copy()
    title_review_dict = data[['title', 'review']].to_dict(orient = 'records')

    # get entities
    begin_entity = time.time()


    def get_eneity(title_review):
        ER_role_setup = """
                You are a helpful expert for entity extraction, and are capable of performing the task below.
                
                You are required to extract key entities and summarize the text content from the Amazon product review. 

                Key entities include:
                1. Product purchase time (e.g., around birthday, before/after holidays, etc.)
                2. Usage scenario (e.g., birthday party, gift, workplace, etc.)
                3. Target user (e.g., children, elderly, etc.)
                4. Purchase motive (e.g., for education, as a gift)
                5. Information channel for discovering the product (e.g., recommended by others, redirected from other websites, etc.)

                If there is no corresponding entity in the text, return an empty value.

                The summary of the text content should include:
                1. User's emotional evaluation (positive, negative or neutral)
                2. Reason for the positive or negative review
                3. Product features that the user focuses on (e.g., color, size, etc.)

                Return the results in JSON format and lower case.

                Example text:
                "I bought this toy before Christmas as a gift for my nephew. He had a lot of fun with it at his birthday party, especially liking its color and size. I discovered this product through a friend's recommendation."

                JSON format return result example:
                {
                "entities": {
                    "purchase time": "before Christmas",
                    "usage scenario": "birthday party",
                    "target user": "nephew",
                    "purchase motive": "as a gift",
                    "information channel": "friend's recommendation"
                },
                "summary": {
                    "evaluation": "positive",
                    "reason": "nephew had a lot of fun at the birthday party",
                    "focus on product features": ["color", "size"]
                }
                }
                """
        messages = [{"role": "system", "content": ER_role_setup},
                    {"role": "user", "content": f"Please process the following review: title - {title_review['title']}; review - {title_review['review']}"}]
        response = client.create(
                                    model = llm,
                                    messages = messages,
                                    temperature = temperature_ER,
                                    response_format={'type':"json_object"})
        entity_set = json.loads(response.choices[0].message.content)
        return entity_set


    def treaded_ER(title_review_dict):
        result_queue = Queue()

        def worker(index, title_review):
            try:
                entity_set = get_eneity(title_review)
            except Exception as e: #### logging e.g., overtime
                entity_set = {
                            "entities": {
                                        "purchase time": "",
                                        "usage scenario": "",
                                        "target user": "",
                                        "purchase motive": "",
                                        "information channel": ""
                                        },
                            "summary": {
                                        "evaluation": "",
                                        "reason": "",
                                        "focus on product features": []
                                        }
                            }
            result_queue.put((index, entity_set))

        threads = []
        for index, title_review in enumerate(title_review_dict): #### 若运行时间不理想尝试异步
            thread = Thread(target=worker, args=(index, title_review))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        entity_set_dict = [None] * len(title_review_dict)
        while not result_queue.empty():
            index, entity_set = result_queue.get()
            entity_set_dict[index] = entity_set
        
        return entity_set_dict


    entity_set_dict = treaded_ER(title_review_dict)

    end_entity = time.time() #### logging
    elapsed_time = np.round(end_entity - begin_entity)
    print(f'getting entity took {elapsed_time} seconds')

    # summarize the reviews
    begin_entity = time.time()

    evaluation_reason_dict = [{"attribute": entity_set["summary"]["evaluation"], "reason": entity_set["summary"]["reason"]} for entity_set in entity_set_dict]
    evaluation_reasons_str = ""
    for idx, evaluation_reason in enumerate(evaluation_reason_dict):
        evaluation_reasons_str += f'{idx+1}. attribute: {evaluation_reason["attribute"]}, reason: {evaluation_reason["reason"]}\n'


    def get_summary(evaluation_reasons):
        role_function = '''
                        Role Description:
                            You are an expert at summarizing text. You are now required to complete the following task.

                        Task Description:
                            You would be provided with a series of review attributes of a particular product, including positive, negative, and neutral, along with the reasons for each. 
                            Your task is to concisely summarize the positive and negative points of these reviews.    

                        How:
                            1. Understand the attribute of each review, and the corresponding reason.
                            2. Summarize the negative and positive points, keeping the number of both the points as small as possible, and provide 1 to 3 short instances for each point.

                        Notes:
                            1. Ensure that the output is 100 percent alignment with the input regarding factuality.
                            2. Ensure that you don't miss out anything.
                            2. Return the points in JSON format.

                        Example input:
                                ```
                                1. attribute: positive, reason: crinkle sounds and tails keep the kiddo entertained, but the reviewer is confused about the bookmarks
                                2. attribute: positive, reason: absolutely adorable and great quality
                                3. attribute: negative, reason: expected a book but received a limp cloth item that resembled a chew toy
                                4. attribute: positive, reason: child loves to play with the product
                                5. attribute: positive, reason: the user found the product nice
                                6. attribute: positive, reason: the user simply states 'GOOD'
                                7. attribute: positive, reason: daughter loves the crinkling noise and holding on to all the different tails, very durable, easy to clean
                                8. attribute: positive, reason: mommy to be is very excited for her baby to have it
                                9. attribute: negative, reason: quality of the product is not very good and there should have been something inside that made other kind of noise
                                10. attribute: negative, reason: misleading picture and overpriced
                                ...
                                ```

                        Example output: 
                                ```
                                {
                                "positive": [
                                            "entertaining": ["crinkling", "loved by kids"],
                                            "good quality": ["cleanability", "durability"],
                                            ...
                                            ],
                                "negative": [
                                            "misleading package": ["bookmark confusion", "misleading picture"],
                                            "unreasonable price": ["overpriced"],
                                            ...
                                            ]
                                }
                                ```
                        '''
        messages = [{"role": "system", "content": role_function},
                    {"role": "user", "content": f"Following is a series of review attributes and corresponding reasons: {evaluation_reasons}. Please process it as required by your role function"}]
        response = client.create(
                                    model = llm,
                                    messages = messages,
                                    temperature = temperature_SUM,
                                    response_format={'type':"json_object"})
        summary_for_reviews = json.loads(response.choices[0].message.content)
        return summary_for_reviews


    summary_for_reviews = get_summary(evaluation_reasons_str)

    # match positives or negatives
    def match_review_point(evaluation_reason, summary_for_reviews):
        role_function = '''                
                        Role Description:
                            You are an expert at extracting key information from an user review for comparison.

                        Task Description:
                            Compare an given user review and compare it with a provided suammry of positive and negative points from user reviews, 
                            to determine whether the review contains any positive or negative points.

                        Steps:
                            1. Understand the given user review, including the corresponding emotional assessment and the reason for the assessment.
                            2. Compare the review and the summary semantically.
                            3. Determine the positive or negative points contained in the review.

                        Notes:
                            1. Ensure that the output is 100 percent from the input summary.
                            2. Do not fabricate the result if no alignment between the given inputs.
                            2. Return the result in JSON format.


                        Example Input:
                                ```
                                review summary:
                                        {
                                        'positive': [{'entertaining': ['crinkling',
                                                                            'loved by kids',
                                                                            "baby loves the crinkle noise and it's easy to hold"]},
                                                    {'good quality': ['cleanability',
                                                                            'durability',
                                                                            'well made']}],
                                        'negative': [{'misleading package': ['misleading picture',
                                                                            'misleading description',
                                                                            'misleading photo showing five books']},
                                                    {'unreasonable price': ['overpriced',
                                                                            'expensive for one book',
                                                                            'not worth the price']}]
                                        }
                                review attribute and reason:
                                        attribute: negative, reason: not worth the money
                                ```

                        Eample Output:
                                ```
                                {"point": ['unreasonable price']}
                                ```
                        '''
        messages = [{"role": "system", "content": role_function},
                    {"role": "user", "content": f"Process the following summary and review. Review attribute and reason: {evaluation_reason}; Review summary: {summary_for_reviews}"}]
        response = client.create(
                                    model = llm,
                                    messages = messages,
                                    temperature = temperature_match,
                                    response_format={'type':"json_object"})
        points = json.loads(response.choices[0].message.content)
        return points


    def threaded_match(evaluation_reason_dict):
        result_queue = Queue()

        def worker(index, evaluation_reason):
            try:
                point_set = match_review_point(evaluation_reason, summary_for_reviews)
            except Exception as e: #### logging e.g., overtime
                point_set = {"point": []}
            result_queue.put((index, point_set))

        threads = []
        for index, evaluation_reason in enumerate(evaluation_reason_dict):
            thread = Thread(target=worker, args=(index, evaluation_reason))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        point_set_dict = [None] * len(evaluation_reason_dict)
        while not result_queue.empty():
            index, point_set = result_queue.get()
            point_set_dict[index] = point_set
        
        return point_set_dict


    point_set_dict = threaded_match(evaluation_reason_dict)

    end_entity = time.time() #### logging
    elapsed_time = np.round(end_entity - begin_entity)
    print(f'summarization took {elapsed_time} seconds')

    # preprocess entities surrounding target user and motive
    user_list = [entity_set["entities"]["target user"] for entity_set in entity_set_dict]
    motive_list = [entity_set["entities"]["purchase motive"] for entity_set in entity_set_dict]


    def bundled_process(entity_list):
        def remove_punct(text):
            # 使用string.punctuation获取所有英文标点符号
            translator = str.maketrans('', '', string.punctuation)
            # 去除标点符号
            text_without_punct = text.translate(translator)
            return text_without_punct

        def remove_stopwords(ls):
            return [item for item in ls if item not in stop_words]

        entity_list = list(map(remove_punct, entity_list))
        entity_list = list(map(word_tokenize, entity_list)) #### 未来可优化
        entity_list = list(map(remove_stopwords, entity_list)) 

        return entity_list


    user_list_processed = bundled_process(user_list)
    motive_list_processed = bundled_process(motive_list)

    #### 加入json验证
    data['purchase_time'] = [i['entities']['purchase time'] for i in entity_set_dict]
    data['scenario'] = [i['entities']['usage scenario'] for i in entity_set_dict]
    data['user'] = [i['entities']['target user'] for i in entity_set_dict]
    data['motive'] = [i['entities']['purchase motive'] for i in entity_set_dict]
    data['channel'] = [i['entities']['information channel'] for i in entity_set_dict]
    data['evaluation'] = [i['summary']['evaluation'] for i in entity_set_dict]
    data['reason'] = [i['summary']['reason'] for i in entity_set_dict]
    data['focus'] = [i['summary']['focus on product features'] for i in entity_set_dict]
    data['reason_point'] = [i['point'] for i in point_set_dict]
    data['user_processed'] = user_list_processed
    data['motive_processed'] = motive_list_processed

    return data, summary_for_reviews


# perform the analysis and return the adjustable result
def get_statistics(data:pd.DataFrame, period=None, variant=None): #### 随后再加入这两个变量（类型）
    """
    The module accepts data, period, product variant (adjustable) as input and returns entity statistics as output
    """

    data = data.copy()
    
    # return the statistics of each type of entity
    ## purchase_time, scenario
    begin = time.time()
    purchase_time = [i for i in set(data['purchase_time'].to_list()) if len(i) != 0]
    scenario = [i for i in set(data['scenario'].to_list()) if len(i) != 0]


    def get_cluster1(name, entity_set):
            entity_set_str = "、".join(entity_set)
            role_function = '''
                            角色说明：
                                你是一个擅长数据分析的数据分析师，现在你需要完成以下任务。

                            任务说明：
                                我会给你某一产品用户购买特征的一系列描述（如，购买时间、使用场景等），请将相似的描述归类，并统计每个类别的数量。
                            
                            操作步骤：
                            1. 阅读并理解每条描述。
                            2. 根据描述内容，识别并归类相似的描述。
                            3. 对每个类别进行计数。
                            4. 输出每个类别、对应的数量以及数个例子。

                            注意事项：
                            1. 归类时主要考虑描述的相似性（比如说“1st birthday”和“for his 1st birthday”可以归为“生日”类别，“the number of books”和“quantity”可以归类为商品数量）。
                            2. 类别名称不能过于笼统，比如不能出现“书籍特征”这种类别，因为无法从“书籍特征”中得知具体的书籍特征。
                            3. 输出的例子个数为2~10个左右，注意不能重复输出。
                            4. 统计时注意不要遗漏或重复计数，总量要与输入一致。
                            5. 返回json格式，语言为英语。  

                            示例输入：
                                    ```                                                
                                    shower,
                                    a few months ago,
                                    for his 1st birthday,
                                    during Easter2023,
                                    Easter,
                                    first Christmas,
                                    for Christmas,
                                    when baby #2 was 4 months and daughter was 2yrs,
                                    baby’s first Christmas,
                                    a few days ago,
                                    since Christmas,
                                    at Christmas,
                                    birthday,
                                    for a Christmas present,
                                    for Christmas,
                                    early Christmas,
                                    1st birthday,
                                    1st birthday,
                                    1st birthday,
                                    Christmas,
                                    1st birthday,
                                    Christmas,
                                    for Christmas,
                                    when she was 5 1/2 months old,
                                    when he was about 3 months
                                    ```

                            示例输出：
                                    ```
                                    {
                                    "birthday": {"number":7,"example":["1st birthday","birthday",...]},  #### 应该可以再简略点
                                    "Christmas": {"number":10,"example":["for Christmas","Christmas",...]},
                                    "baby growth": {"number":2,"example":["when he was about 3 months","when she was 5 1/2 months old"]},
                                    "Halloween": {"number":1,"example":["Easter"]}
                                    }
                                    ```
                    '''
            messages = [{"role": "system", "content": role_function},
                        {"role": "user", "content": f"以下是某款产品购买用户的{name}，大概有{len(entity_set)}个，请按照你的角色功能设定对以下一系列描述进行处理：{entity_set_str}"}]
            response = client.create(
                                        model = llm,
                                        messages = messages,
                                        temperature = temperature_CLU1,
                                        response_format={'type':"json_object"})
            entity_set = json.loads(response.choices[0].message.content)
            return entity_set


    try:
        purchase_time_cluster = get_cluster1("购买时间", purchase_time) #### prompt改为英文
    except: #### logging
        purchase_time_cluster = {"time_error": {"number":0,"example":[]}}
    try:
        scenario_cluster = get_cluster1("使用场景", scenario)
    except:
        scenario_cluster = {"scenario_error": {"number":0,"example":[]}}

    end = time.time()
    elapsed_time = np.round(end - begin)
    print(f'clustering 1 took {elapsed_time} seconds') #### logging

    ## user, motive, focus, reason_point (by positive/negative or by whether returned) #### 后续再加入return
    begin = time.time()

    user = Counter([y for x in data['user_processed'].to_list() for y in x if len(y) != 0])
    motive = Counter([y for x in data['motive_processed'].to_list() for y in x if len(y) != 0])
    focus = Counter([y for x in data['focus'].to_list() for y in x if len(y) != 0])
    reason_point_positive = dict(Counter([x for i in data.query("evaluation == 'positive'")['reason_point'].to_list() if len(i) != 0 for x in i]))
    reason_point_negative = dict(Counter([x for i in data.query("evaluation == 'negative'")['reason_point'].to_list() if len(i) != 0 for x in i]))


    def get_cluster2(name, entity_set_count):
        role_function = '''
                        角色说明：
                            你是一个擅长文字理解和数据分析的专家，现在你需要完成以下任务。

                        任务说明：
                            我会给你从某一产品的用户评论中抽取出来和购买特征相关的一系列关键词（如用户类别、购买动机、用户对产品的关注点等）以及统计数量，我需要你把形式上和语义上相似的关键词进一步归类并返回。
                        
                        操作步骤：
                        1. 阅读并理解每个关键词。
                        2. 根据关键词形式上和语义上的相似度进行归类。
                        3. 输出每个类别以及该类别的关键词和数量。

                        注意事项：
                        1. 归类时主要考虑描述的形似和语义相似性（比如说“quantity”和“number”可以归为“数量”类别，“babies”、“baby”和“infant”可以归为“婴儿”类别）。
                        2. 并且根据我给出的关键词的属性对关键词序列进行筛选，忽略无关的关键词。比如，我告诉你关键词属于“用户的类别”，关键词序列为
                        （{'baby': 150, 'old': 76, 'month': 44, 'babies': 38, 'grandson': 29, 'little': 25}），的“old”、“month”、“little”可以忽略（因为这些数量词性质上不是用户）只需要返回“baby”、“babies”、“grandson”
                        3. 保证返回的关键词和统计数量与输入一致，不能出现原序列里不存在的关键词。
                        4. 以json格式返回，语言为英语。  

                        示例输入：
                                ```
                                keyword attribute: 用户对产品的关注点
                                keyword series:                                                
                                    {'tails': 68,
                                    'books': 50,
                                    'price': 50,
                                    'quality': 48,
                                    'quantity': 48,
                                    'colors': 37,
                                    'size': 37,
                                    'number': 34,
                                    'crinkle': 28,
                                    'pages': 28,
                                    'color': 23,
                                    'crinkly': 23,
                                    'cute': 23,
                                    'sound': 22,
                                    'book': 19,
                                    'durability': 18,
                                    'colorful': 17,
                                    'different': 15,
                                    'material': 15,
                                    ...}
                                ```

                        示例输出：
                                其中，关键词“different”与属性“用户对产品的关注点”无关，则直接忽略
                                ```
                                {
                                "book": {"books":50, "book":19},
                                "color": {"colors":37, "colorful":17},
                                "material": {"crinkly":23, "crinkle":28, "material":15},
                                "number": {"number":34, "quantity":48},
                                "quality": {"quality":48, "durability":18},
                                "sound": {"sound": 22},
                                "appearance": {"cute": 23, "tails":68},
                                "size": {"size": 37, "durability":18},
                                "price": {"price": 50},
                                ...
                                }
                                ```
                '''
        messages = [{"role": "system", "content": role_function},
                    {"role": "user", "content": f"以下序列是某款产品的{name}（keyword attribute），请按照你的角色功能设定进行处理：{entity_set_count}（keyword series）"}]
        response = client.create(
                                    model = llm,
                                    messages = messages,
                                    temperature = temperatureCLU2,
                                    response_format={'type':"json_object"})
        entity_set = json.loads(response.choices[0].message.content)
        return entity_set

    try:
        user_cluster = get_cluster2("用户的类别", user)
    except:
        user_cluster = {"user_error": {"user":0}}
    try:
        motive_cluster = get_cluster2("用户购买动机", motive)    
    except:
        motive_cluster = {"motive_error": {"motive":0}}
    try:
        focus_cluster = get_cluster2("用户对产品的关注点", focus)
    except:
        focus_cluster = {"focus_error": {"focus":0}}
    user_cluster_counts = {}
    motive_cluster_counts = {}
    focus_cluster_counts = {}
    for main_category, sub_categories in user_cluster.items():
        user_cluster_counts[main_category] = sum(sub_categories.values())
    for main_category, sub_categories in motive_cluster.items():
        motive_cluster_counts[main_category] = sum(sub_categories.values())
    for main_category, sub_categories in focus_cluster.items():
        focus_cluster_counts[main_category] = sum(sub_categories.values())

    ## channel, evaluation
    channel = dict(Counter(sorted([i for i in data['channel'] if len(i) != 0])))
    evaluation = dict(Counter(sorted(data['evaluation'].to_list())))

    end = time.time() #### logging
    elapsed_time = np.round(end - begin)
    print(f'clustering 2 took {elapsed_time} seconds')

    return [purchase_time_cluster, scenario_cluster, user_cluster_counts, motive_cluster_counts, channel, evaluation, reason_point_positive, reason_point_negative, focus_cluster_counts]


if __name__ == "__main__":
    begin_time = time.time()
    file_path = './B07WZT8DTD-US-Reviews-240910-73648_test.xlsx'
    data = read_data(file_path)

    if isinstance(data, pd.DataFrame):
        # check data length
        if len(data) >= 20:
            try:
                data_processed = preprocess_data(data)
            except Exception as e:
                print(f'error amid pre-processing: {e}')

            try:
                data_result, summary_for_reviews = pre_analysis(data_processed)
            except Exception as e:
                print(f'error amid analysis: {e}')

            try:
                statistical_result = get_statistics(data_result)
                with open('test.txt', 'w', encoding='utf-8') as f:
                    f.write(str(statistical_result))
            except Exception as e:
                print(f'error amid statistical analysis: {e}')

        else:
            print('data not enough')
    else:
        print('file content error')

    end_time = time.time()
    elapsed_time = np.round(end_time - begin_time)
    print(f'the full process took {elapsed_time} seconds')