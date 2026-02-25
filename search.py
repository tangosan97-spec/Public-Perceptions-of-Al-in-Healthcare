import requests
import csv
from datetime import datetime
import pytz
import time

posts_num = 0
comments_num = 0
# search_terms = [
#     "Medical AI", "Healthcare AI", "Artificial Intelligence in Healthcare", 
#     "AI in Medicine", "AI in Healthcare", "AI Diagnosis", 
#     "Medical Technology AI", "AI in Hospitals", "AI for Doctors", 
#     "AI for Patients", "AI Diagnosis Tools", "AI Surgery Assistance", 
#     "AI Medical Imaging", "AI Cancer Detection", "AI in Radiology", 
#     "AI in Pathology", "AI in Drug Discovery", "AI in Mental Health", 
#     "ChatGPT for Healthcare", "AI powered Medical Devices", 
#     "Trust in Medical AI", "Ethics of AI in Healthcare", "AI and Patient Privacy", 
#     "AI replacing Doctors", "Fear of AI in Medicine", "AI in Healthcare Opinions", 
#     "AI Bias in Medicine", "AI and Medical Errors", "AI and Patient Safety", 
#     "Doctors vs AI", "AI and Medical Careers", "AI and Nurses", 
#     "AI Impact on Healthcare Jobs", "Future of Medicine with AI", "GPT4 in Medicine", 
#     "Chatbots in Healthcare", "AI powered Healthcare Apps", "AI for Telemedicine",  
#     "AI vs Human Doctors", "Patient Perspective on AI", "Doctors on AI", 
#     "Public Opinion on AI in Healthcare", "AI in Medicine Pros and Cons", 
#     "Fear of AI in Medicine"
# ]
# search_terms = [
#     "AI in Drug Discovery", "AI in Mental Health", 
#     "ChatGPT for Healthcare", "AI powered Medical Devices", 
#     "Trust in Medical AI", "Ethics of AI in Healthcare", "AI and Patient Privacy", 
#     "AI replacing Doctors", "Fear of AI in Medicine", "AI in Healthcare Opinions", 
#     "AI Bias in Medicine", "AI and Medical Errors", "AI and Patient Safety", 
#     "Doctors vs AI", "AI and Medical Careers", "AI and Nurses", 
#     "AI Impact on Healthcare Jobs", "Future of Medicine with AI", "GPT4 in Medicine", 
#     "Chatbots in Healthcare", "AI powered Healthcare Apps", "AI for Telemedicine", 
#     "AI vs Human Doctors", "Patient Perspective on AI", "Doctors on AI", 
#     "Public Opinion on AI in Healthcare", "AI in Medicine Pros and Cons", 
#     "Fear of AI in Medicine"
# ]

# AI-related concepts group (English)
ai_terms = (
    '(AI OR "Artificial Intelligence" OR "Machine Learning" OR ML OR "Deep Learning" OR DL OR '
    'NLP OR "Natural Language Processing" OR "Computer Vision")'
)

# Healthcare-related concepts group (English)
health_terms = (
    '(healthcare OR medical OR medicine OR clinical OR patient OR diagnostic OR diagnosis OR '
    'treatment OR therapy OR hospital OR health OR pharma OR pharmaceutical OR drug OR '
    'radiology OR pathology OR genomics OR "digital health" OR healthtech OR medtech)'
)

# search_query = f"{ai_terms} {health_terms}"
search_query = f'title: (AI OR "Artificial Intelligence") AND (health OR medical) '


# 保存帖子到 CSV
def save_to_csv(csv_name, posts):
    with open(csv_name, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'Title', 'URL', 'Created (UTC)', 'Created (Beijing)', 'Score', 'Num Comments']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 如果是第一次写入，写入表头
        if csvfile.tell() == 0:
            writer.writeheader()

        for post in posts:
            writer.writerow(post)

def is_reddit_post(post_data):
    # 检查 domain 是否包含 "reddit"
    if "reddit" in post_data["domain"]:
        return True
    # 或者检查 URL 是否来自 Reddit
    elif post_data["url"].startswith("https://www.reddit.com/"):
        return True
    return False

# 主循环
def fetch_reddit_posts(keyword):
    # query = 'title:' + '"' + keyword + '"'
    query = keyword

    # Reddit API 搜索 URL
    url = "https://www.reddit.com/r/all/search.json"
    params = {
        "q": query,  # 搜索关键字
        "sort": "new",  # 按时间新旧排序
        "t": "all",  # 限制为过去所有时间的帖子
        "limit": 100,  # 每次请求最多返回 10 个帖子
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }

    # 北京时间（UTC+8）
    beijing_tz = pytz.timezone("Asia/Shanghai")
    after = None  # 初始没有 after 参数
    while True:
        # 如果有 after 参数，添加到请求参数中
        if after:
            params["after"] = "t3_" + after
        
        try:
            # 发送请求获取数据
            response = requests.get(url, params=params, headers=headers)
            data = response.json()

            if not data["data"]["children"]:
                print("No more posts to fetch.")
                break

            posts_to_save = []
            earliest_post = None

            # 处理返回的每个帖子
            for post in data["data"]["children"]:
                post_data = post["data"]
                
                # 获取帖子的创建时间并转为北京时间
                created_utc = post_data["created_utc"]
                created_beijing = datetime.utcfromtimestamp(created_utc).replace(tzinfo=pytz.utc).astimezone(beijing_tz)

                # 判断帖子是否来自 Reddit（domain == 'self.reddit'）
                if is_reddit_post(post_data):
                    # 保存有效帖子信息
                    post_info = {
                        'id' : post_data['id'],
                        'Title': post_data['title'],
                        'URL': post_data['url'],
                        'Created (UTC)': created_utc,
                        'Created (Beijing)': created_beijing.strftime('%Y-%m-%d %H:%M:%S'),
                        'Score': post_data['score'],
                        'Num Comments': post_data['num_comments']
                    }
                    posts_to_save.append(post_info)
                    global posts_num, comments_num
                    posts_num += 1
                    comments_num += int(post_data['num_comments'])


                # 找到 Unix 时间戳最小的帖子作为 next 'after' 参数
                if not earliest_post or created_utc < earliest_post["created_utc"]:
                    earliest_post = {
                        "fullname": post_data['id'],
                        "created_utc": created_utc
                    }

            # 保存有效的 Reddit 帖子
            # save_to_csv('./posts_url/' + query[7:-1] + '.csv',posts_to_save)
            save_to_csv('./posts_url/' + query + '.csv',posts_to_save)

            # 判断是否更新 after 参数
            if earliest_post:
                if after != "t3_" + earliest_post["fullname"]:
                    after = earliest_post["fullname"]
                    print(f"Next 'after' parameter: {after}")
                else:
                    print("No new posts to fetch, stopping.")
                    break
            else:
                print("No valid posts found in this batch.")
                break
        except requests.exceptions.RequestException as e:
            retry_count = 0
            print(f"Request failed with exception: {e}. Retrying in 5 seconds...")
            time.sleep(10)  # 等待 5 秒钟后重试
            retry_count += 1
            if retry_count >= 50:  # 如果重试超过 5 次，终止
                print("Maximum retry attempts reached. Exiting...")
                break


for keyword in search_terms:
    print('-' * 30)
    print("Keyword:", keyword)



    # 运行获取帖子
    fetch_reddit_posts(keyword)
    print("Keyword:", keyword, "Complete")
    print("total posts number:", posts_num)
    print("total comments number:", comments_num)    
    print('-' * 30)
