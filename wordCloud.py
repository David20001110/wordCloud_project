import numpy as np
import pandas as pd
import json
from pathlib import Path
import string
from typing import List, Tuple
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import nltk

# 下載停用詞
nltk.download('stopwords')
# 下載詞型還原器
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# 做出reviews的七位數匯總
def seven_number_summary(file_path):
    reviews_length = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            length = len(data['text'])
            reviews_length.append(length)
    reviews_series = pd.Series(reviews_length)

    return reviews_series


# 讀取檔案
def read_amazon_reviews(file_path):
    count = 0
    positive_review = []
    negative_review = []
    positive_reviews_count = 0
    negative_reviews_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            count += 1
            if count<50:
                data = json.loads(line)
                text = data['text']
                label = data['label']
                if label == 1:
                    positive_review.append(text)
                    positive_reviews_count += 1
                else:
                    negative_review.append(text)
                    negative_reviews_count += 1
            else:
                break
    return positive_review, negative_review, positive_reviews_count, negative_reviews_count

    # with open(file_path, 'r') as f:
    #     data = json.load(f)
    #     texts = []
    #     labels = []
    #     positive_reviews = []
    #     negative_reviews = []
    #     for item in data:
    #         texts.append(item['text'])
    #         labels.append(item['label'])
    #         if item['label'] == 1:
    #             positive_reviews.append(item['text'])
    #         else:
    #             negative_reviews.append(item['text'])
    #     # num_positive = labels.count(1)
    #     # num_negative = labels.count(0)
    #     return positive_reviews, negative_reviews


# 處理文字
def tidy_process_word(text: str):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word.isalpha())
    stop_word = set(stopwords.words("english"))
    text = ' '.join(word for word in text.split() if word not in stop_word)
    # 還原型態
    wordlemmatizer = WordNetLemmatizer()
    text = ' '.join(wordlemmatizer.lemmatize(word) for word in text.split())
    return text


# 標註詞性提取V、N、adj的詞性
def pos_tagging_extract_word(text: str) -> Tuple[list, list, list, list]:
    alls = []
    verbs = []
    nonus = []
    adjective = []
    # 將string拆解成一個list裡面是單詞
    words = word_tokenize(text)
    # 標註詞性
    words = pos_tag(words)
    for word, tag in words:
        alls.append(word)
        if tag.startswith('JJ'):
            adjective.append(word)
        elif tag.startswith('V'):
            verbs.append(word)
        elif tag.startswith('N'):
            nonus.append(word)
    return nonus, verbs, adjective, alls


def count_word(list1: List[str]):
    word_count = Counter()
    for text in list1:
        word_list = text.split()
        word_count.update(word_list)
    # 提取前200個(有問題)
    # word_count type is list
    word_count = word_count.most_common(200)
    # 轉成dict再轉成Counter
    word_count = Counter(dict(word_count))
    return word_count


# 繪製wordcloud
def generate_wordCloud(words_count, title: str, output_path):
    wordcloud = WordCloud(width=800, height=800, background_color='black', max_words=100).generate_from_frequencies(
        words_count)
    plt.figure(figsize=(8, 8))
    # interpolation表示圖像的顯示方式這裡使用雙線性內插法
    plt.imshow(wordcloud, interpolation='bilinear')
    # 隱藏坐標軸
    plt.axis('off')
    plt.title(title, fontsize=30)
    plt.savefig(output_path)


if __name__ == '__main__':
    # json轉為dict
    json_data = Path('amazon_polarity.json')
    # 回傳好與壞評論

    summary = seven_number_summary(json_data)
    print("min :", summary.min(), " max :", summary.max())
    print("10% :", int(summary.quantile(0.1)), ' , 25% :', int(summary.quantile(0.25)), " , 50% :",
          int(summary.quantile(0.5)), " , 75% :", int(summary.quantile(0.75)), " , 90% :", int(summary.quantile(0.9)))
    positive_reviews, negative_reviews, positive_reviews_numbers, negative_reviews_numbers = read_amazon_reviews(
        json_data)
    print('正面的論共有', positive_reviews_numbers, '條')
    print('負面的論共有', positive_reviews_numbers, '條')

    # 將處理好的好與壞的文字評論存入positive_reviews_processed以及negative_reviews_processed type為list
    positive_reviews_processed = tidy_process_word(str(positive_reviews))
    negative_reviews_processed = tidy_process_word(str(negative_reviews))

    # print('好的評論 : ', positive_reviews_processed)
    # print('不好的評論 : ', negative_reviews_processed, '\n')

    positive_n_word, positive_v_word, positive_adj_word, positive_all_word = pos_tagging_extract_word(
        positive_reviews_processed)
    negative_n_word, negative_v_word, negative_adj_word, negative_all_word = pos_tagging_extract_word(
        negative_reviews_processed)
    # print('好評論的名詞 : ', positive_n_word)
    # print('不好評論的名詞 : ', negative_n_word, '\n')

    # 把文字出現的次數計算出來存成一個collection.Counter 裡面的格式像是dict
    positive_n_numbers = count_word(positive_n_word)
    positive_v_numbers = count_word(positive_v_word)
    positive_adj_numbers = count_word(positive_adj_word)
    positive_all_numbers = count_word(positive_all_word)
    negative_n_numbers = count_word(negative_n_word)
    negative_v_numbers = count_word(negative_v_word)
    negative_adj_numbers = count_word(negative_adj_word)
    negative_all_numbers = count_word(negative_all_word)
    # print('好評論名詞的單字統計', positive_n_numbers)
    # print('不好評論名詞的單字統計', negative_n_numbers)

    # 產生文字雲
    generate_wordCloud(positive_n_numbers, 'positive_review_nn', Path('output/positive_review_nn'))
    generate_wordCloud(negative_n_numbers, 'negative_review_nn', Path('output/negative_review_nn'))
    generate_wordCloud(positive_v_numbers, 'positive_review_vv', Path('output/positive_review_vv'))
    generate_wordCloud(negative_v_numbers, 'negative_review_vv', Path('output/negative_review_vv'))
    generate_wordCloud(positive_adj_numbers, 'positive_review_adj', Path('output/positive_review_adj'))
    generate_wordCloud(negative_adj_numbers, 'negative_review_adj', Path('output/negative_review_adj'))
    generate_wordCloud(positive_all_numbers, 'positive_review', Path('output/positive_review'))
    generate_wordCloud(negative_all_numbers, 'negative_review', Path('output/negative_review'))

    # 好評和壞平的數量
    positive_review_count = len(positive_reviews)
    negative_review_count = len(negative_reviews)
