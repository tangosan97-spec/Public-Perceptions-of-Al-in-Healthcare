import praw
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
import requests

# 设置代理环境变量
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 测试代理是否生效（访问 Google）
try:
    response = requests.get("https://www.google.com", timeout=5)
    print("代理连接成功，Google 响应状态码:", response.status_code)
except Exception as e:
    print("代理连接失败，错误信息:", e)
# ==== 请替换为你自己的凭证 ====
REDDIT_CLIENT_ID = ""
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT = ""

# ==== 初始化 praw ====
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# ==== 文件路径设置 ====
csv_path = r"F:\Work\reddit爬虫\all_reddit_posts_merged_cleaned.csv"
save_dir = r"F:\Work\reddit爬虫\posts"
os.makedirs(save_dir, exist_ok=True)

# ==== 读取帖子URL与ID ====
df = pd.read_csv(csv_path)
total_posts = len(df)

# ==== 找出已处理的帖子 ====
existing_ids = set(f.split(".")[0] for f in os.listdir(save_dir) if f.endswith(".json"))

# ==== 开始处理 ====
start_time = time.time()
processed = 0

for idx, row in df.iterrows():
    post_id = row['id']
    post_url = row['URL']

    if post_id in existing_ids:
        continue

    try:
        submission = reddit.submission(url=post_url)
        submission.comments.replace_more(limit=None)
        comments = []

        for comment in submission.comments.list():
            comments.append({
                "id": comment.id,
                "body": comment.body,
                "author": str(comment.author),
                "created_utc": comment.created_utc,
                "created_beijing": datetime.utcfromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                "score": comment.score,
                "parent_id": comment.parent_id,
                "depth": comment.depth
            })

        post_data = {
            "id": submission.id,
            "title": submission.title,
            "url": submission.url,
            "created_utc": submission.created_utc,
            "created_beijing": datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
            "selftext": submission.selftext,
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "author": str(submission.author),
            "subreddit": str(submission.subreddit),
            "is_self": submission.is_self,
            "link_flair_text": submission.link_flair_text,
            "comments": comments
        }

        # 保存文件
        save_path = os.path.join(save_dir, f"{post_id}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(post_data, f, indent=2, ensure_ascii=False)

        processed += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / processed
        remaining = total_posts - len(existing_ids) - processed
        eta = timedelta(seconds=int(remaining * avg_time))
        progress_pct = 100 * (processed + len(existing_ids)) / total_posts

        print(f"[✓] Saved {post_id} ({processed + len(existing_ids)}/{total_posts}) "
              f"[{progress_pct:.2f}%] | ETA: {eta} | Avg: {avg_time:.2f}s")

    except Exception as e:
        print(f"[!] Failed to fetch post {post_id}: {e}")
        continue
