# Public Perceptions of AI in Healthcare: A Large-Scale Analysis of Reddit Discussions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the data, source code, and analysis pipeline for the manuscript **"Public Perceptions of AI in Healthcare: A Large-Scale Analysis of Reddit Discussions"**. 

The study systematically investigates public discourse on medical AI by analyzing 36,555 Reddit comments across 36 subreddits from March 1, 2020, to March 31, 2025. We employ BERTopic for dynamic topic modeling and an LLM-based zero-shot classification approach for sentiment analysis.

## Repository Structure

The repository is organized to facilitate the full reproduction of the data pipeline.

```text
.
├── posts/                      # Directory containing raw JSON data for individual Reddit posts
├── posts.csv                   # Cleaned and processed dataset of Reddit posts and metadata
├── sentiment.csv               # Final dataset containing text with sentiment labels and scores
├── search.py                   # Main scraping script for retrieving Reddit posts
├── praw_try.py                 # Deprecated attempt using PRAW (included for reference only, not used in the final pipeline)
├── clean.py                    # Data preprocessing (noise removal, NLTK stop words, bot filtering)
├── bertopic_analysis.py        # Implementation of the BERTopic modeling pipeline
├── sentiment_analysis.py       # LLM zero-shot sentiment classification script
├── statistic.py                # Script for temporal sentiment trends
├── cloud_data.py               # Calculates term frequencies and TF-IDF across sentiment categories
├── cloud_figure.py             # Generates Word Cloud visualizations
└── README.md                   # This document

```

## System Requirements & Installation

### Prerequisites

* Operating System: Windows, macOS, or Linux
* Python: version 3.8 or higher

### Dependencies

Install the required Python packages using `pip`:

```bash
pip install praw pandas numpy matplotlib scikit-learn scipy requests tqdm nltk bertopic sentence-transformers umap-learn hdbscan wordcloud
```

*Note: The script `sentiment_analysis.py` utilizes an external LLM API. Ensure you have the appropriate API keys and base URLs configured in the script before running.*

## Reproduction Pipeline

Follow these steps to reproduce the analytical workflow presented in the manuscript.

### 1. Data Collection
Run the main scraping script to retrieve posts from the designated subreddits.
```bash
python search.py
```

*Raw JSON outputs are saved in the `posts/` directory.*

### 2. Data Preprocessing

Run the cleaning script to anonymize data, remove URLs, filter non-English content, and exclude bot-generated spam.

```bash
python clean.py
```

*Outputs the structured `posts.csv` file.*

### 3. Topic Modeling

Execute the BERTopic pipeline to generate the semantic clusters.

```bash
python bertopic_analysis.py
```

### 4. Sentiment Analysis

Run the LLM-based zero-shot classifier to categorize the emotional tone of each entry.

```bash
python sentiment_analysis.py
```

*Outputs the `sentiment.csv` file containing sentiment labels.*

### 5. Statistics & Visualization

Generate the quantitative results, sentiment trend area charts, and word clouds.

```bash
# Calculate TF-IDF and term frequencies (outputs tf_idf_results.csv and word_frequencies.csv)
python cloud_data.py

# Generate word cloud visualizations (outputs .png files)
python cloud_figure.py

# Run temporal trend analysis and significance testing (outputs sentiment_trends.png)
python statistic.py
```

## Data Availability & Privacy Statement

In strict adherence to ethical guidelines and the Reddit User Agreement, all data provided in this repository (`posts.csv` and `sentiment.csv`) have been thoroughly anonymized. Usernames and identifiable metadata have been removed. Furthermore, all representative quotes used in the manuscript have been semantically paraphrased to prevent traceability and protect user privacy.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
