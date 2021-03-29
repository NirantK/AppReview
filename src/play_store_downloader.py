import json
from pathlib import Path

import fire
from app_store_scraper import AppStore
from google_play_scraper import Sort, reviews, reviews_all

data_path = Path("src/data")


def download(app_id="com.ubercab", country="us", count=1000):
    result, continuation_token = reviews(
        app_id=app_id,
        lang="en",  # defaults to 'en'
        country=country,  # defaults to 'us'
        sort=Sort.MOST_RELEVANT,  # defaults to Sort.MOST_RELEVANT
        count=count,  # defaults to 1000
        filter_score_with=None,  # defaults to None(means all score)
    )
    with open(str(data_path / f"{app_id}_{country}_play_store_reviews.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(result)


if __name__ == "__main__":
    fire.Fire(download)
