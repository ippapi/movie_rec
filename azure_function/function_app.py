import azure.functions as func
from surprise import dump
import pandas as pd
import traceback

blueprint = func.Blueprint()
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

#========== Globals ==========
MODEL_PATH = "/home/chauhb/Projects/movie_rec/storage/pretrained_model/model.pkl"
USER_MAP_PATH = "/home/chauhb/Projects/movie_rec/storage/prepared_data/user_map.parquet"
MOVIE_MAP_PATH = "/home/chauhb/Projects/movie_rec/storage/prepared_data/movie_map.parquet"

algo = None
user2id, id2user, item2id, id2item = None, None, None, None

#========== Load model ==========
def load_model():
    global algo
    if algo is None:
        print("Loading model...")
        _, algo = dump.load(MODEL_PATH)
        print("Model loaded")

#========== Load mapping ==========
def get_mapping():
    global user2id, id2user, item2id, id2item
    print("Loading user/item mappings...")
    if user2id == None:
        df_user = pd.read_parquet(USER_MAP_PATH)
        df_item = pd.read_parquet(MOVIE_MAP_PATH)

        user2id = dict(zip(df_user["user_id"], df_user["user_idx"]))
        id2user = dict(zip(df_user["user_idx"], df_user["user_id"]))

        item2id = dict(zip(df_item["movie_id"], df_item["movie_idx"]))
        id2item = dict(zip(df_item["movie_idx"], df_item["movie_id"]))
        print(f"{len(user2id)} users and {len(item2id)} items loaded")

@blueprint.route(route="recommend", auth_level=func.AuthLevel.ANONYMOUS)
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    try:
        print("Received recommendation request")
        load_model()
        get_mapping()

        user_id = int(req.params.get("user_id"))
        print(f"Request user_id: {user_id}")

        if not user_id:
            return func.HttpResponse("User_id missing", status_code=400)

        if user_id not in user2id:
            return func.HttpResponse(f"User_id not found", status_code=400)

        user_idx = user2id[user_id]
        all_items = list(item2id.values())
        preds = []

        for iid in all_items:
            try:
                pred = algo.predict(user_idx, iid)
                preds.append((iid, pred.est))
            except Exception as e:
                print(f"Prediction failed for item {iid}: {e}")

        top5 = sorted(preds, key=lambda x: x[1], reverse=True)[:5]
        top5_ids = [id2item[iid] for iid, _ in top5]
        print(f"Top 5 recommendations for {user_id}: {top5_ids}")

        return func.HttpResponse(str(top5_ids))

    except Exception as e:
        print("Error in recommend function:", e)
        traceback.print_exc()
        return func.HttpResponse("Internal server error", status_code=500)

app.register_blueprint(blueprint)