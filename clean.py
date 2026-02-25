import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/zyc/project/Sentiment/crawl/sentiment.csv')

# 第一步：删除 sentiment 为 Meaningless 和 topic 为 无关 的行
df = df[(df['sentiment'] != 'Meaningless') & (df['topic'] != '无关')]

# 第二步：处理时间，保留年份信息（从时间戳转换为年份）
df['year'] = pd.to_datetime(df['time'], unit='s').dt.year

# 仅保留2023和2024年的 Neutral 数据
neutral_df = df[(df['sentiment'] == 'Neutral') & (df['year'].isin([2023, 2024]))]

# 统计每年的 neutral 数量
year_counts = neutral_df['year'].value_counts()
total = year_counts.sum()

# 计算各年份要删除的数量（按比例）
delete_counts = (year_counts / total * 10000).round().astype(int)

# 创建要删除的索引集合
to_drop_idx = []
for year in [2023, 2024]:
    n_delete = delete_counts.get(year, 0)
    temp = neutral_df[neutral_df['year'] == year].sort_values(by='score').head(n_delete)
    to_drop_idx.extend(temp.index)

# 删除这些记录
df_cleaned = df.drop(index=to_drop_idx)

# 可选：保存清洗后的数据
df_cleaned.to_csv('/home/zyc/project/Sentiment/post_process/sentiment_cleaned.csv', index=False)

print(f"原始记录数: {len(df)}, 清洗后记录数: {len(df_cleaned)}")
