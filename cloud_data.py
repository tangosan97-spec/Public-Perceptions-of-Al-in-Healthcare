import os
import json
import pandas as pd
from collections import Counter
import re
from tqdm import tqdm

# 输入路径
csv_path = '/home/zyc/project/Sentiment/post_process/sentiment_reclassified.csv'
json_dir = '/home/zyc/project/Sentiment/new_finished_posts'
output_dir = '/home/zyc/project/Sentiment/post_process/word'

# 停用词（可扩展）
stop_words = set([
    'the', 'and', 'for', 'you', 'that', 'with', 'this', 'from', 'are', 'but',
    'was', 'have', 'not', 'they', 'has', 'had', 'all', 'can', 'would', 'there',
    'been', 'were', 'what', 'when', 'will', 'your', 'just', 'out', 'how', 'who',
    'get', 'about', 'one', 'like', 'them', 'more', 'because', 'some', 'could',
    'should', 'did', 'which', 'than', 'their', 'then', 'she', 'him', 'her', 'his',
    'its', 'our', 'also', 'into', 'why', 'where', 'too', 'any', 'does', 'i', 'we',  'people',  'think', 'don', 'even', 'make', 'going', 'years', 'being', 'see', 'way', 'very', 'other', 'really', 'know', 'want', 'things', 'use', 'world', 'much', 'most', 'https', 'etc'
])

# 加载CSV
df = pd.read_csv(csv_path)

# 词频容器
sentiment_words = {
    'Positive': Counter(),
    'Neutral': Counter(),
    'Negative': Counter()
}

post_cache = {}

def tokenize_and_clean(text):
    # 简单英文分词 + 清洗
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [
        word for word in words
        if len(word) >= 3 and word not in stop_words
    ]

# 主处理逻辑
for _, row in tqdm(df.iterrows(), total=len(df)):
    sid = row['id']
    pid = row['parent_id']
    sentiment = row['reclassified_sentiment']
    if sentiment not in sentiment_words:
        continue

    # 解析 post id
    post_id = sid if pd.isna(pid) or pid == 'None' else pid.replace("t3_", "")

    # 加载 JSON
    if post_id not in post_cache:
        json_path = os.path.join(json_dir, f"{post_id}.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                post_data = json.load(f)
            post_cache[post_id] = post_data
        except Exception:
            continue
    else:
        post_data = post_cache[post_id]

    text = ""
    if pd.isna(pid) or pid == 'None':  # post
        title = post_data.get('title', '')
        body = post_data.get('selftext', '')
        text = f"{title} {body}"
    else:  # comment
        comments = post_data.get('comments', [])
        comment = next((c for c in comments if c.get('id') == sid), None)
        if comment:
            text = comment.get('body', '')

    words = tokenize_and_clean(text)
    sentiment_words[sentiment].update(words)

# 保存每类情感的词频到CSV
for senti, counter in sentiment_words.items():
    freq_df = pd.DataFrame(counter.most_common(100), columns=['word', 'count'])
    save_path = os.path.join(output_dir, f'word_freq_{senti.lower()}.csv')
    freq_df.to_csv(save_path, index=False)
    print(f"{senti} 词频前100已保存：{save_path}")
