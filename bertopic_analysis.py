import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# -------------------
# 路径配置
# -------------------
post_dir = "/home/zyc/project/Sentiment/new_finished_posts"
output_dir = "/home/zyc/project/Sentiment/post_process/fig"
os.makedirs(output_dir, exist_ok=True)

# -------------------
# 加载数据
# -------------------
documents = []
post_ids = []

for file in tqdm(os.listdir(post_dir)):
    if file.endswith(".json"):
        with open(os.path.join(post_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            post_text = data.get("selftext", "")
            if post_text.strip():
                documents.append(post_text.strip())
                post_ids.append(data.get("id"))

# -------------------
# 构建嵌入模型与主题模型
# -------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    verbose=True
)

topics, probs = topic_model.fit_transform(documents)

# -------------------
# 保存主题关键词表格
# -------------------
topic_info = topic_model.get_topic_info()
topic_keywords = []

for topic_id in topic_info.Topic:
    if topic_id == -1:
        continue
    keywords = [word for word, _ in topic_model.get_topic(topic_id)]
    topic_keywords.append({
        "topic_id": topic_id,
        "name": f"Topic {topic_id}",
        "keywords": ", ".join(keywords[:10])
    })

df_keywords = pd.DataFrame(topic_keywords)
df_keywords.to_csv(os.path.join(output_dir, "topic_keywords.csv"), index=False)

# -------------------
# 保存各类图像
# -------------------
topic_model.visualize_topics().write_html(os.path.join(output_dir, "visualize_topics.html"))
topic_model.visualize_documents(documents, topics=topics).write_html(os.path.join(output_dir, "visualize_documents.html"))
topic_model.visualize_barchart(top_n_topics=10).write_html(os.path.join(output_dir, "barchart_top10.html"))
topic_model.visualize_hierarchy().write_html(os.path.join(output_dir, "hierarchy.html"))
topic_model.visualize_heatmap().write_html(os.path.join(output_dir, "heatmap.html"))
topic_model.visualize_distribution(probs).write_html(os.path.join(output_dir, "distribution.html"))

# -------------------
# 生成每个主题关键词折线图和示例文档保存
# -------------------
for topic_id in topic_model.get_topics().keys():
    if topic_id == -1:
        continue

    topic_words = topic_model.get_topic(topic_id)
    words = [w for w, _ in topic_words]
    scores = [s for _, s in topic_words]

    # 关键词贡献折线图
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(scores)), scores, marker="o")
    plt.xticks(range(len(words)), words, rotation=45, ha="right")
    plt.title(f"Topic {topic_id} - Word Importance")
    plt.xlabel("Word Rank in Topic")
    plt.ylabel("c-TF-IDF Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"topic_{topic_id}_keywords.png"))
    plt.close()

    # 示例文档保存
    indices = [i for i, t in enumerate(topics) if t == topic_id]
    example_docs = [documents[i] for i in indices[:5]]
    with open(os.path.join(output_dir, f"topic_{topic_id}_examples.txt"), "w", encoding="utf-8") as f:
        f.write(f"Topic {topic_id} Example Documents:\n\n")
        for idx, doc in enumerate(example_docs):
            f.write(f"--- Document {idx + 1} ---\n{doc}\n\n")

print(f"所有图像与表格已保存在: {output_dir}")
