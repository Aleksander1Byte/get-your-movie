from get_your_movie import recommend_movies

data_dir = "."
favorites = ["Amélie", "Das Leben der Anderen", "La vita è bella"]
targets = ["western", "china", "hongkong", "japan"]

for region in targets:
    recs = recommend_movies(
        favorite_titles=favorites,
        target_region=region,
        n=5,
        data_dir=data_dir,
        alpha=0.85,
    )
    print(f"\n=== {region.upper()} (favorites: {', '.join(favorites)}) ===")
    print(
        recs[
            [
                "title",
                "rating",
                "genres",
                "countries",
                "release_date",
                "similarity",
                "score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
