import os
import json
from tqdm import tqdm
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import psutil
from openai import OpenAI
from functools import lru_cache
import re
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(filename='processing.log', level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration Parameters
DATA_DIR = "/home/zyc/project/Sentiment/new_posts"
BUFFER_SIZE = 10  # Batch size for API calls
TOTAL_NUM = 0


# ------------------------------ Data Cleaning ------------------------------ #
search_terms1 = [
    "AI", "Artificial Intelligence", "GPT", "LLM", "Assistance", "ChatGPT", "GPT-4", "AI-powered", "Chatbots"
]


search_terms2 = [
    "Medical", "Healthcare", "Pathology", "Diagnosis", "Doctors", "Patients", "Cancer", "Radiology", "AI"
]

garbage_keywords = [
    # General Ads and Promo
    "ad", "ads", "advertisement", "promo", "promotion", "buy", "sell", "offer", "sale", "discount", "deal",
    "clearance", "cheap", "free", "giveaway", "bargain", "coupon", "limited time", "special offer",
    
    # Financial Scams
    "scam", "fraud", "investment", "get rich", "quick money", "make money", "easy cash", "earn", 
    "profit", "passive income", "pyramid scheme", "binary option", "forex", "crypto", "bitcoin", "ethereum", "nft",
    
    # Link Bait & Spam
    "click here", "visit", "link", "subscribe", "follow", "check out", "http", "https", "www", "signup",
    "register", "join now", "don't miss", "exclusive", "limited", "access now", "download", "watch now",
    "buy now", "order now", "reserve your spot", "act now",
    
    # Adult Content
    "onlyfans", "porn", "xxx", "nsfw", "adult", "dating", "nude", "explicit", "sex", "escort", "sugar daddy",
    "camgirl", "hot singles", "live chat", "sexting", "fetish", "webcam", "milf", "bdsm", "18+", "premium content",
    
    # Gambling
    "casino", "poker", "bet", "lottery", "jackpot", "win big", "gambling", "slots", "wager", "bingo",
    "sports betting", "roulette", "blackjack", "scratch card", "free spin",
    
    # Counterfeit products
    "fake", "replica", "knockoff", "counterfeit", "cheap version", "bootleg", "copy", "scammer",
    "imitation", "low-quality", "phishing",
    
    # Health Scams
    "miracle", "magic", "weight loss", "diet pill", "fat burner", "cure", "detox", "anti-aging", 
    "supplement", "boost", "enhance", "no side effects", "natural remedy", "herbal", "guaranteed results",
    "instant relief", "health scam",
    
    # Gaming Spam
    "free game", "game cheat", "hack", "mod", "cheat code", "unlimited", "unlock", "skins", "aimbot", 
    "wallhack", "free gems", "game booster",
    
    # Contact Bait
    "email me", "contact us", "phone number", "dm me", "message me", "inbox me", "reach out",
    
    # Clickbait
    "unbelievable", "amazing", "incredible", "shocking", "breaking news", "viral", "must see", "trending",
    "life-changing", "you won't believe", "never seen before",
    
    # Platform specific
    "survey", "poll", "pay-to-win", "cash app", "venmo", "paypal", "zelle", "gift card", "voucher",
    "rebate", "rewards", "free trial", "sign up", "join us", "membership", "vip",
    
    # Job Scams
    "job opportunity", "work from home", "remote job", "freelance", "part-time job", "easy job",
    "no experience needed", "start today", "be your own boss", "startup kit", "earn online", "career opportunity",
    "training", "online course", "certification",
    
    # Misc Spam
    "spam", "junk", "hoax", "viral content", "fake news", "chain message", "forward this", "share now",
    "copy-paste", "spread the word", "unverified"
]

def contains_links(post_data):
    title = post_data.get("title", "")
    selftext = post_data.get("selftext", "")
    
    # Regex to check for URLs
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    
    # If title or body contains too many links, consider it spam
    if len(re.findall(url_pattern, title)) > 2 or len(re.findall(url_pattern, selftext)) > 1:
        return True
    return False

def contains_garbage_keywords(post_data, garbage_keywords, threshold=2):
    """
    Check if the post contains garbage/spam keywords.
    """
    title = post_data.get("title", "").lower()
    selftext = post_data.get("selftext", "").lower()

    garbage_count = 0

    for term in garbage_keywords:
        # Match whole words only
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        
        if re.search(pattern, title, re.IGNORECASE):
            garbage_count += 1
        if re.search(pattern, selftext, re.IGNORECASE):
            garbage_count += 1

    if contains_links(post_data):
        return True

    return garbage_count >= threshold

def contains_keywords(post_data, search_terms):
    """
    Check if the post contains relevant search terms.
    """
    title = post_data.get("title", "").lower()
    selftext = post_data.get("selftext", "").lower()

    # Filter out extremely long posts
    if len(selftext) >= 1500:
        return False

    # Filter out spam
    if contains_garbage_keywords(post_data, garbage_keywords=garbage_keywords, threshold=3):
        return False

    # Match whole words for search terms
    search_pattern = r'\b(' + '|'.join(re.escape(term.lower()) for term in search_terms) + r')\b'

    if re.search(search_pattern, title) or re.search(search_pattern, selftext):
        return True

    return False

# ------------------------------ Sentiment Analysis ------------------------------ #

client = OpenAI(api_key="YOUR_KEY", base_url="YOUR_BASE_URL")

SYSTEM_PROMPT = """
#### Role
- Name: AI Healthcare Sentiment Analysis Expert
- Task: Analyze social media sentiment regarding AI in Healthcare.

#### Capabilities
- Relevance: Identify content directly related to AI in medical contexts.
- Context: Analyze relationships between posts and comments.
- Classification: Accurately determine author's sentiment.

#### Sentiment Criteria
- Positive: Benefits, progress, or success stories.
- Neutral: Factual statements without clear bias.
- Negative: Risks, ethical concerns, or failures.
- Meaningless: Irrelevant content (ads, games, etc.).

#### Output Instructions
- Input: Social media text (post or comment).
- Output: JSON format. 
  - sentiment: (Positive/Neutral/Negative)
  - topic: One word (Application, Privacy, Patient-Doctor, Medicine, Employment, Safety, Newcomer, Ethics, Future, Fairness, Education, Cost, Bias, Mental)
  - relevance: Integer between 0 and 10.

#### Special Notes
- For comments, evaluate based on the thread context.
- relevance: 0 for irrelevant/ads, 10 for highly focused medical AI discussion.

EXAMPLE JSON OUTPUT:
{
    "results": [
        {
            "id": "1awin1",
            "sentiment": "Positive",
            "topic": "Application",
            "relevance": 8
        },
        {
            "id": "1bekw2",
            "sentiment": "Negative",
            "topic": "Privacy",
            "relevance": 2
        }
    ]
}
"""

def sentiment(prompt):
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    stream=False
    )
    return response

def get_single_prompt(id, type, title, selftext, parent_body, body):
    prompt = f"""
        id: {id}
        Type: {type}
        Post Title: {title}
        Post Body: {selftext}
        Parent Comment: {parent_body}
        Comment Body: {body}
    """
    return prompt

def get_prompt(prompt_list):
    return "".join(prompt_list)

def get_api_result(buffer, writer):
    retry = 5
    post_start_time = time.time()
    prompts = [item["prompt"] for item in buffer]
    full_prompt = get_prompt(prompts)
    
    response = None
    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                    ],
                stream=False,
                response_format={'type': 'json_object'}
            )
            break
        except Exception as e:
            time.sleep(1)
            print(f"Error sending API request: {e}")
            with open("./error_log.txt", "a") as error_file:
                error_file.write(f"API Error: {e}, Retry {attempt+1}\n")
    
    if not response:
        return

    content = response.choices[0].message.content
    results = []
    try:
        data = json.loads(content)
        # Extract results regardless of the top-level key used by LLM
        for key in data.keys():
            if isinstance(data[key], list):
                results.extend(data[key])
    except Exception as e:
        print(f"JSON Parsing error: {str(e)}")

    for result in results:
        if not isinstance(result, dict):
            logging.warning(f"Invalid result format: {result}")
            continue
            
        for item in buffer:
            if str(item["id"]) == str(result.get('id')):
                try:
                    item["sentiment"] = result.get('sentiment')
                    item["topic"] = result.get('topic')
                    newline = {
                        "id": item["id"],
                        "time": item["time"],
                        "score": item["score"],
                        "parent_id": item["parent_id"],
                        "sentiment": item["sentiment"],
                        "topic": item["topic"],
                        "relevance": item.get("relevance", 0)
                    }
                    writer.writerow(newline)
                except Exception as e:
                    print(f"Error processing ID {item['id']}: {e}")
                    logging.error(f"Processing error for ID {item.get('id')}: {e}")

    end_time = time.time()
    global TOTAL_NUM
    TOTAL_NUM -= 10
    print("{:{fill}^{width}}".format("Duration {time:.2f}s, Estimated remaining {remain:.2f} min".format(time=end_time-post_start_time, remain=max(0, (TOTAL_NUM/10)*(end_time-post_start_time)/60)), fill="+", width=50)) 

def load_posts(path):
    print("{:{fill}^{width}}".format("Start loading posts and comments", fill="=", width=70)) 
    total = 0
    posts = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    for filename in tqdm(os.listdir(path)):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    post_data = json.load(f)
                    if post_data.get("is_self") and contains_keywords(post_data=post_data, search_terms=search_terms1) and contains_keywords(post_data=post_data, search_terms=search_terms2):
                        total += post_data.get("num_comments", 0)
                        posts.append(post_data)  
            except Exception as e:
                print(f"Warning: Error reading {filename} - {str(e)}")
    
    global TOTAL_NUM
    TOTAL_NUM = total
    print(f"Total comments to process: {TOTAL_NUM}")
    return posts  

def main(posts, output_path, csv_headers):
    with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
        buffer = []
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        post_process_count = 0
        for post in posts:
            post_id = post["id"]
            post_title = post["title"]
            post_body = post["selftext"].replace("\n", "").replace("\r", "")[:500]
            post_time = post["created_utc"]
            post_score = post["score"]
            post_numcomments = post["num_comments"]

            buffer.append({
                "prompt": get_single_prompt(post_id, "post", post_title, post_body, "None", "None"), 
                "id": post_id, 
                "time": post_time,
                "score": post_score, 
                "parent_id": "None", 
                "sentiment": None, 
                "topic": None, 
                "relevance": 0
            })
            
            if len(buffer) == BUFFER_SIZE:
                get_api_result(buffer, writer)
                buffer = []

            if post_numcomments > 0:
                id2com_dict = {c["id"]: c["body"] for c in post.get("comments", [])}
                for comment in post.get("comments", []):
                    comment_id = comment["id"]
                    comment_body = comment["body"].replace("\n", "").replace("\r", "")[:500]
                    comment_time = comment["created_utc"]
                    comment_score = comment["score"]
                    comment_parent = comment["parent_id"]
                    comment_depth = comment["depth"]

                    buffer.append({
                        "prompt": get_single_prompt(comment_id, "comment", post_title, post_body, "None" if comment_depth == 0 else id2com_dict.get(comment_parent[3:]), comment_body), 
                        "id": comment_id, 
                        "time": comment_time, 
                        "score": comment_score, 
                        "parent_id": comment_parent, 
                        "sentiment": None, 
                        "topic": None, 
                        "relevance": 0
                    })
                    if len(buffer) == BUFFER_SIZE:
                        get_api_result(buffer, writer)
                        buffer = []

            try:
                # Move finished files
                target_dir = os.path.join(os.path.dirname(DATA_DIR), "new_finished_posts")
                os.makedirs(target_dir, exist_ok=True)
                os.rename(os.path.join(DATA_DIR, f"{post_id}.json"), os.path.join(target_dir, f"{post_id}.json"))
            except Exception:
                continue
                
            post_process_count += 1
            print(f"Logged post {post_id}, Progress: {post_process_count}/{len(posts)}")
            csvfile.flush()

if __name__ == "__main__":
    output_file = "./sentiment.csv"
    csv_headers = ["id", "time", "score", "parent_id", "sentiment", "topic", "relevance"]
    
    if not os.path.isfile(output_file):
        print("{:{fill}^{width}}".format("No existing file, writing headers", fill="-", width=70)) 
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
    else:
        print("{:{fill}^{width}}".format("Existing file detected", fill="-", width=70))

    posts = load_posts(DATA_DIR)
    main(posts=posts, output_path=output_file, csv_headers=csv_headers)