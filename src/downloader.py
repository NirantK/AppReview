import fire
import json
from google_play_scraper import Sort, reviews_all, reviews

def download(app_id="com.ubercab", country="us", count=1000):
    result, continuation_token = reviews(
        app_id=app_id,
        lang='en', # defaults to 'en'
        country=country, # defaults to 'us'
        sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
        count=count, # defaults to 1000
        filter_score_with=None # defaults to None(means all score)
    )
    with open(f"{app_id}_{country}_reviews.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(result)

if __name__ == '__main__':
    fire.Fire(download)