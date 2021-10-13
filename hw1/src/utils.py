import numpy as np


def calc_auc(model, R):
    R = R.toarray()
    auc = 0
    for u_id, u in enumerate(R):
        if np.count_nonzero(u != 0) == 0:
            continue

        pos_mask = u > 0
        neg_mask = ~u

        preds = model.I @ model.U[u_id]
        comp_matrix = preds[pos_mask].reshape(-1, 1) > preds[neg_mask]
        u_auc = np.count_nonzero(comp_matrix) / (comp_matrix.shape[0] * comp_matrix.shape[1])

        auc += u_auc

    return auc / R.shape[0]


def test_model(model, implicit_ratings, movie_info):
    get_similars = lambda item_id, model: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string() for x in model.similar_items(item_id)
    ]
    print("Similars:")
    for r in get_similars(1, model):
        print(r)

    get_user_history = lambda user_id, implicit_ratings: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string()
        for x in implicit_ratings[implicit_ratings["user_id"] == user_id]["movie_id"]
    ]

    print("User's history:")
    for f in get_user_history(4, implicit_ratings):
        print(f)

    get_recommendations = lambda user_id, model: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string() for x in model.recommend(user_id)
    ]

    print("Recommendations:")
    for r in get_recommendations(4, model):
        print(r)
