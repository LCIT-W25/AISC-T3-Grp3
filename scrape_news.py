import requests
from pymongo import MongoClient
from bs4 import BeautifulSoup
from datetime import datetime
import time

# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "newsDB"
COLLECTION_NAME = "articles"

# Initialize MongoDB connection
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Function to scrape news from Google News for a specific category
def scrape_google_news(category):
    base_url = f"https://news.google.com/rss/headlines/section/topic/{category.upper()}?hl=en-CA&gl=CA&ceid=CA:en"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "xml")

    news_items = []
    items = soup.find_all("item")[:5]  # Limit to 5 news items per category

    for item in items:
        news = {
            "Source": "Google News",
            "Author": "N/A",  # Google News RSS doesn't provide author
            "Title": item.title.text,
            "Description": item.description.text,
            "Content": "N/A",  # RSS feed doesn't provide full content
            "URL": item.link.text,
            "PublishTime": item.pubDate.text,
        }
        news_items.append(news)

    return news_items

# Function to fetch news from NewsAPI for a specific category
def fetch_newsapi_news(category):
    API_KEY = "5ddd6ecb574548749c05c1e966b38168" 
    NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"

    params = {
        "apiKey": API_KEY,
        "country": "ca",
        "category": category.lower(),
        "pageSize": 5,
    }

    response = requests.get(NEWSAPI_URL, params=params)
    news_data = response.json()

    # Print the API response to debug
    print(f"NewsAPI response for category {category}: {news_data}")

    news_items = []
    if news_data["status"] == "ok":
        for article in news_data["articles"]:
            news = {
                "Source": "NewsAPI",
                "Author": article.get("author", "N/A"),
                "Title": article["title"],
                "Description": article.get("description", "N/A"),
                "Content": article.get("content", "N/A"),
                "URL": article["url"],
                "PublishTime": article["publishedAt"],
            }
            news_items.append(news)

    return news_items

# Function to save news to MongoDB
def save_to_mongo(news_items):
    # Add a timestamp to track when the news was added
    for news in news_items:
        news["SavedAt"] = datetime.utcnow()

        # Insert into MongoDB collection
        result = collection.insert_one(news)
        print(f"Saved news: {news['Title']} with ID: {result.inserted_id}")

    # Check if the new data is in the collection
    count = collection.count_documents({})
    print(f"Documents in collection after insertion: {count}")

# Main execution
if __name__ == "__main__":
    # Delete old news only once at the start
    print("Deleting old news from the database...")
    collection.delete_many({})

    # Add a small delay to ensure the deletion is processed
    time.sleep(2)
    
    # Fetch and save news from Google News
    categories = ["CANADA", "WORLD", "BUSINESS", "TECHNOLOGY", "ENTERTAINMENT", "SPORTS", "SCIENCE", "HEALTH"]

    print("Fetching news from Google News...")
    for category in categories:
        google_news = scrape_google_news(category)
        save_to_mongo(google_news)

    # Fetch and save news from NewsAPI
    print("Fetching news from NewsAPI...")
    for category in categories:
        newsapi_news = fetch_newsapi_news(category)
        save_to_mongo(newsapi_news)

    print("All news has been saved to MongoDB.")

    # Close MongoDB connection after saving all data
    client.close()

