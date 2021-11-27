import functools
import catboost as cb
import pandas as pd
import numpy as np
from catboost import CatBoostRanker
from sklearn.model_selection import train_test_split


def preprocess_data(members, songs, songs_extra):
    members["membership_days"] = (
        members["expiration_date"].subtract(members["registration_init_time"]).dt.days.astype(int)
    )

    members["registration_year"] = members["registration_init_time"].dt.year
    members["registration_month"] = members["registration_init_time"].dt.month
    members["registration_date"] = members["registration_init_time"].dt.day

    members["expiration_year"] = members["expiration_date"].dt.year
    members["expiration_month"] = members["expiration_date"].dt.month
    members["expiration_date"] = members["expiration_date"].dt.day
    members = members.drop(["registration_init_time"], axis=1)

    def isrc_to_year(isrc):
        if type(isrc) == str:
            if int(isrc[5:7]) > 17:
                return 1900 + int(isrc[5:7])
            else:
                return 2000 + int(isrc[5:7])
        else:
            return np.nan

    songs_extra["song_year"] = songs_extra["isrc"].apply(isrc_to_year)
    songs_extra.drop(["isrc", "name"], axis=1, inplace=True)

    return members, songs, songs_extra


def create_ds(df, songs, members, songs_extra):
    df = df.merge(songs, on="song_id", how="left")
    df = df.merge(members, on="msno", how="left")
    df = df.merge(songs_extra, on="song_id", how="left")

    df.song_length.fillna(200000, inplace=True)
    df.song_length = df.song_length.astype(np.uint32)
    df.song_id = df.song_id.astype("category")

    return df


def lyricist_count(x):
    if x == "no_lyricist":
        return 0
    else:
        return sum(map(x.count, ["|", "/", "\\", ";"])) + 1


def composer_count(x):
    if x == "no_composer":
        return 0
    else:
        return sum(map(x.count, ["|", "/", "\\", ";"])) + 1


def is_featured(x):
    if "feat" in str(x):
        return 1
    return 0


def artist_count(x):
    if x == "no_artist":
        return 0
    else:
        return x.count("and") + x.count(",") + x.count("feat") + x.count("&")


# is song language 17 or 45.
def song_lang_boolean(x):
    if "17.0" in str(x) or "45.0" in str(x):
        return 1
    return 0


def smaller_song(x, mean_song_length):
    if x < mean_song_length:
        return 1
    return 0


def process_df(df):
    df_agg = df.groupby("msno").aggregate("count")
    df_agg = df_agg[df_agg["song_id"] < 1024]

    users = set(df_agg.index)

    df = df[df["msno"].isin(users)]

    df["lyricist"] = df["lyricist"].cat.add_categories(["no_lyricist"])
    df["lyricist"].fillna("no_lyricist", inplace=True)
    df["lyricists_count"] = df["lyricist"].apply(lyricist_count).astype(np.int8)

    df["composer"] = df["composer"].cat.add_categories(["no_composer"])
    df["composer"].fillna("no_composer", inplace=True)
    df["composer_count"] = df["composer"].apply(composer_count).astype(np.int8)

    df["artist_name"] = df["artist_name"].cat.add_categories(["no_artist"])
    df["artist_name"].fillna("no_artist", inplace=True)
    df["is_featured"] = df["artist_name"].apply(is_featured).astype(np.int8)

    df["artist_count"] = df["artist_name"].apply(artist_count).astype(np.int8)

    # if artist is same as composer
    df["artist_composer"] = (np.asarray(df["artist_name"]) == np.asarray(df["composer"])).astype(np.int8)

    # if artist, lyricist and composer are all three same
    df["artist_composer_lyricist"] = (
        (np.asarray(df["artist_name"]) == np.asarray(df["composer"]))
        & np.asarray((df["artist_name"]) == np.asarray(df["lyricist"]))
        & np.asarray((df["composer"]) == np.asarray(df["lyricist"]))
    ).astype(np.int8)

    df["song_lang_boolean"] = df["language"].apply(song_lang_boolean).fillna(0).astype(np.int8)

    mean_song_length = np.mean(df["song_length"])

    df["smaller_song"] = (
        df["song_length"].apply(functools.partial(smaller_song, mean_song_length=mean_song_length)).astype(np.int8)
    )

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("nan").astype('category')

    df = df.sort_values(by="msno")

    y = df.target
    q = df.msno
    X = df.drop(["target", "msno"], axis=1)

    return X, y, q


def train_cb(train, val):
    default_parameters = {
        'custom_metric': ["NDCG", "QueryAUC", "AUC", 'AverageGain:top=10'],
        'random_seed': 42,
        "loss_function": "YetiRank",
        "train_dir": "YetiRank",
        "metric_period": 100
    }

    parameters = {
        "learning_rate": 0.1,
        "iterations": 500,
        "task_type": "GPU",
        **default_parameters
    }

    model = CatBoostRanker(**parameters)
    model.fit(train, eval_set=val, plot=True, verbose=1, )

    metrics = model.eval_metrics(data=val, metrics=["NDCG", "QueryAUC", "AUC"])

    return model, metrics


def main():
    data_path = "../data/"
    train = pd.read_csv(
        data_path + "train.csv",
        dtype={
            "msno": "category",
            "source_system_tab": "category",
            "source_screen_name": "category",
            "source_type": "category",
            "target": np.uint8,
            "song_id": "category",
        },
    )
    songs = pd.read_csv(
        data_path + "songs.csv",
        dtype={
            "genre_ids": "category",
            "language": "category",
            "artist_name": "category",
            "composer": "category",
            "lyricist": "category",
            "song_id": "category",
        },
    )
    members = pd.read_csv(
        data_path + "members.csv",
        dtype={"city": "category", "bd": np.uint8, "gender": "category", "registered_via": "category"},
        parse_dates=["registration_init_time", "expiration_date"],
    )
    songs_extra = pd.read_csv(data_path + "song_extra_info.csv")

    train, val = train_test_split(train, train_size=0.8, random_state=42)

    print("Preprocessing data...")
    members, songs, songs_extra = preprocess_data(members, songs, songs_extra)

    print("Merging data...")
    train = create_ds(train, songs, members, songs_extra)
    val = create_ds(val, songs, members, songs_extra)

    print("Processing data...")
    X_train, y_train, q_train = process_df(train)
    X_val, y_val, q_val = process_df(val)

    cat_features = list(X_train.select_dtypes(["category"]))

    arr = [col for col in cat_features if X_train[col].isnull().values.any()]

    X_train[arr] = X_train[arr].astype(str).fillna("nan").astype("category")
    X_val[arr] = X_val[arr].astype(str).fillna("nan").astype("category")

    train = cb.Pool(
        data=X_train,
        label=y_train,
        group_id=q_train,
        cat_features=cat_features
    )

    val = cb.Pool(
        data=X_val,
        label=y_val,
        group_id=q_val,
        cat_features=cat_features
    )

    print("Fitting model...")
    model, metrics = train_cb(train, val)

    for name, vals in metrics.items():
        print(f"{name}: {vals[-1]: .4f}")

    print("Done!")


if __name__ == "__main__":
    main()
