import json
import string
from collections import Counter
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 設定檔案路徑和目標資料夾
data_path = Path('exercises/python_exercises/project_exercises/amazon_product_review/data/amazon_polarity.json')
output_folder = Path('exercises/python_exercises/project_exercises/amazon_product_review/output')

# 讀取 JSON 檔案並解析為字典格式
with open(data_path, 'r') as f:
    reviews = [json.loads(line) for line in f]

# 計算好評和差評的數量
positive_count = sum(review['label'] == '1' for review in reviews)
negative_count = len(reviews) - positive_count

print(f'Positive reviews: {positive_count}')
print(f'Negative reviews: {negative_count}')


# 定義一個函數來處理文字
def preprocess_text(text):
    # 將文字轉換為小寫字母
    text = text.lower()

    # 移除標點符號
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 移除非英文單詞
    text = ' '.join(word for word in text.split() if word.isalpha())

    # 移除停用詞
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text


# 定義一個函數來生成文字雲圖片
def generate_wordcloud(words, output_path):
    # 設定字體和背景顏色
    wc = WordCloud(background_color='white', max_words=100, colormap='tab10', font_path='arial.ttf')

    # 生成文字雲
    wc.generate_from_frequencies(words)

    # 顯示圖片
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

    # 儲存圖片
    plt.savefig(output_path)


# 處理好評和差評的評論
positive_text = ' '.join(preprocess_text(review['text']) for review in reviews if review['label'] == '1')
negative_text = ' '.join(preprocess_text(review['text']) for review in reviews if review['label'] == '0')

# 計算好評和差評中的單詞出現次數
positive_word_counts = Counter(word_tokenize(positive_text))
negative_word_counts = Counter(word_tokenize(negative_text))

# 生成文字雲圖片

