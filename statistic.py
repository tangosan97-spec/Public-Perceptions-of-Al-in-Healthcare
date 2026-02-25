import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import matplotlib.font_manager as fm
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = fm.FontProperties(fname=font_path)

# 用法一：全局设置字体
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块
# 设置文件路径
csv_path = '/home/zyc/project/Sentiment/code/sentiment.csv'
output_dir = './sentiment_analysis_output'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(csv_path)

# ===================
# 控制台输出部分
# ===================
print("=== 数据概览 ===")
print(df.head())
print("\n=== 总数据量 ===")
print(len(df))

# 判断是否为post
df['is_post'] = df['parent_id'].isna()

# 情感分布
sentiment_counts = df['sentiment'].value_counts()
print("\n=== 情感分布 ===")
print(sentiment_counts)

# topic分布
topic_counts = df['topic'].value_counts()
print("\n=== 话题分布 ===")
print(topic_counts)

# post与comment数量
post_count = df['is_post'].sum()
comment_count = len(df) - post_count
print(f"\n=== Post数量: {post_count}，Comment数量: {comment_count} ===")

# 每个话题下的情感分布（透视表）
topic_sentiment_pivot = pd.pivot_table(df, index='topic', columns='sentiment', aggfunc='size', fill_value=0)
print("\n=== 每个话题下的情感分布 ===")
print(topic_sentiment_pivot)

# ===================
# 图像保存部分
# ===================

# 情感分布图
plt.figure()
sentiment_counts.plot(kind='bar', title='Sentiment Distribution')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))

# topic分布图
plt.figure(figsize=(10, 5))
topic_counts.plot(kind='barh', title='Topic Distribution')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'topic_distribution.png'))

# Post与Comment数量图
plt.figure()
plt.bar(['Post', 'Comment'], [post_count, comment_count])
plt.title('Post vs Comment Count')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'post_comment_distribution.png'))

# 每个话题下的情感热力图
plt.figure(figsize=(12, 8))
sns.heatmap(topic_sentiment_pivot, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Sentiment per Topic')
plt.ylabel('Topic')
plt.xlabel('Sentiment')
plt.savefig(os.path.join(output_dir, 'sentiment_per_topic_heatmap.png'))

print(f"\n图像已保存到: {os.path.abspath(output_dir)}")
