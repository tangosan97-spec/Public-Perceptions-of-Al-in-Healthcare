import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# 输入路径和输出目录
input_dir = '/home/zyc/project/Sentiment/post_process/word'
fig_dir = os.path.join(input_dir, 'fig')
os.makedirs(fig_dir, exist_ok=True)

# 情感类别
sentiments = ['positive', 'neutral', 'negative']

# 生成词云图
for senti in sentiments:
    csv_path = os.path.join(input_dir, f'word_freq_{senti}.csv')
    img_path = os.path.join(fig_dir, f'wordcloud_{senti}.png')
    
    # 加载词频
    df = pd.read_csv(csv_path)
    word_freq = dict(zip(df['word'], df['count']))

    # 生成词云
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    # 绘图保存
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud - {senti.capitalize()}")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    print(f"词云图已保存：{img_path}")
