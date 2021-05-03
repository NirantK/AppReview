import json
from pathlib import Path

import fire
from app_store_scraper import AppStore

data_path = Path("data")


def download(app_name: str, country: str = "us", count: int = 200):
    print(f"App Name : {app_name}")
    print(f"Country : {country}")
    print(f"count : {count}")
    try:
        app = AppStore(country=country, app_name=app_name)
        app.review(how_many=count)
        result = app.reviews
        file_path = data_path / f"{app_name}_{country}_app_store_reviews.json"
        with file_path.open("w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        print(f"Couln't fetch the reviews for {app_name} with following error : {e}")


if __name__ == "__main__":
    fire.Fire(download)
