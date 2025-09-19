# Register this blueprint by adding the following line of code 
# to your entry point file.  
# app.register_functions(blueprint) 
# 
# Please refer to https://aka.ms/azure-functions-python-blueprints


import azure.functions as func
from surprise import dump
import pandas as pd
import logging

blueprint = func.Blueprint()
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

#========== Load pretrain model ========== 
MODEL_PATH = "/mnt/models/svd_model.pkl"
algo = None

def load_model():
    global algo
    if algo is None:
        _, algo = dump.load(MODEL_PATH)

#========== Load parquet mapping ==========
USER_MAP_PATH = "/mnt/data/user_map.parquet"
MOVIE_MAP_PATH = "/mnt/data/movie_map.parquet"

df_user = pd.read_parquet(USER_MAP_PATH)
df_item = pd.read_parquet(MOVIE_MAP_PATH)

user2id = dict(zip(df_user["user_id"], df_user["user_idx"]))
id2user = dict(zip(df_user["user_idx"], df_user["user_id"]))

item2id = dict(zip(df_item["movie_id"], df_item["movie_idx"]))
id2item = dict(zip(df_item["movie_idx"], df_item["movie_id"]))


@blueprint.route(route="recommend", auth_level=func.AuthLevel.ANONYMOUS)
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = user2id[req.params.get("user_id")]
    if not user_id or user_id not in user2id:
        return func.HttpResponse("User_id got no recommendation", status_code=400)

    all_items = list(item2id.values())
    preds = []
    for iid in all_items:
        pred = model.predict(user_id, iid)
        preds.append((iid, pred))
    top5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    top5 = [id2item[item_idx] for item_idx in top5]

    return func.HttpResponse(str(top5))