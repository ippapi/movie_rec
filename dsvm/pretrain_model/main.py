import argparse
import pandas as pd
import time
from collections import defaultdict
from surprise import Dataset, Reader, SVD, NMF, KNNWithMeans, BaselineOnly, SlopeOne, CoClustering
from surprise import dump
from surprise import accuracy
import sys
import math
import numpy as np

class FileLogger:
    def __init__(self, logfile=None):
        self.logfile = logfile

    def _log(self, level, msg):
        t = time.strftime("%H:%M", time.localtime())
        line = f"[{t}][{level}] {msg}\n"
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def info(self, msg):
        self._log("INFO", msg)

    def warning(self, msg):
        self._log("WARNING", msg)

    def error(self, msg):
        self._log("ERROR", msg)

# -----------------------------
# Top-k metrics với negative sampling
# -----------------------------
def metrics_at_k_with_negatives(algo, test_df, train_df, k=10, n_neg=99):
    """
    test_df: DataFrame user_id, movie_id, rating (positive)
    train_df: DataFrame để biết items user đã xem (avoid sampling seen items)
    n_neg: số negative item cho mỗi user
    """
    all_items = set(train_df['movie_id'].unique())
    user_train_dict = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()

    precisions, recalls, ndcgs = [], [], []

    for uid, group in test_df.groupby('user_id'):
        pos_items = list(group['movie_id'])
        seen_items = user_train_dict.get(uid, set())
        neg_candidates = list(all_items - seen_items - set(pos_items))
        if len(neg_candidates) >= n_neg:
            neg_items = np.random.choice(neg_candidates, n_neg, replace=False)
        else:
            neg_items = neg_candidates

        items_to_predict = pos_items + list(neg_items)
        ratings_true = [1]*len(pos_items) + [0]*len(neg_items)

        preds = []
        for item, r_true in zip(items_to_predict, ratings_true):
            est = algo.predict(uid, item).est
            preds.append((est, r_true))

        # sort top-k
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]

        n_rel_total = sum(ratings_true)
        n_rel_topk = sum(r for (_, r) in top_k)

        precisions.append(n_rel_topk / len(top_k) if top_k else 0)
        recalls.append(n_rel_topk / n_rel_total if n_rel_total else 0)

        rels = [r for (_, r) in top_k]
        ideal_rels = sorted(ratings_true, reverse=True)[:k]
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(rels))
        idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    avg_precision = sum(precisions)/len(precisions) if precisions else 0
    avg_recall = sum(recalls)/len(recalls) if recalls else 0
    avg_ndcg = sum(ndcgs)/len(ndcgs) if ndcgs else 0

    return avg_precision, avg_recall, avg_ndcg

# -----------------------------
# Main function
# -----------------------------
def main(args):
    log = FileLogger(args.log_path)
    start_time = time.time()

    if args.load_model:
        log.info(f"Loading model from {args.load_model} ...")
        _, algo = dump.load(args.load_model)
    else:
        if args.model_name.lower() == "svd":
            algo = SVD(n_factors=args.n_factor, n_epochs=args.n_epoch, reg_all=args.reg_all)
        elif args.model_name.lower() == "nmf":
            algo = NMF(n_factors=args.n_factor, n_epochs=args.n_epoch)
        elif args.model_name.lower() == "knn":
            algo = KNNWithMeans(k=args.k_neighbors, sim_options={'name': 'cosine', 'user_based': True})
        elif args.model_name.lower() == "baseline":
            algo = BaselineOnly(bsl_options={'method': 'als', 'reg': args.reg_all})
        elif args.model_name.lower() == "slopeone":
            algo = SlopeOne()
        elif args.model_name.lower() == "cocluster":
            algo = CoClustering(n_cltr_u=args.n_cltr_u, n_cltr_i=args.n_cltr_i, n_epochs=args.n_epoch)
        else:
            raise ValueError(f"Unknown model_name: {args.model_name}")


    if args.mode == "train" and args.traindir:
        log.info("Loading training data...")
        train_df = pd.read_parquet(args.traindir)
        reader = Reader(rating_scale=(1,5))
        train_data = Dataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader)
        train_set = train_data.build_full_trainset()
        log.info("Training model...")
        algo.fit(train_set)
        log.info(f"Training done in {time.time() - start_time:.2f}s")

        train_predictions = algo.test(train_set.build_testset())
        train_rmse = accuracy.rmse(train_predictions)
        log.info(f"RMSE on training set: {train_rmse:.4f}")

        if args.save_model:
            log.info(f"Saving model to {args.save_model} ...")
            dump.dump(args.save_model, algo=algo)

    if args.mode == "eval" and args.testdir and args.traindir:
        log.info("Loading testing data...")
        test_df = pd.read_parquet(args.testdir)
        train_df = pd.read_parquet(args.traindir)
        log.info("Evaluating model with negative sampling...")

        rmse = accuracy.rmse(algo.test(list(zip(test_df['user_id'], test_df['movie_id'], test_df['rating']))))
        avg_precision, avg_recall, avg_ndcg = metrics_at_k_with_negatives(
            algo, test_df, train_df, k=args.k, n_neg=99
        )

        log.info(f"Average Precision@{args.k}: {avg_precision:.4f}")
        log.info(f"Average Recall@{args.k}: {avg_recall:.4f}")
        log.info(f"Average NDCG@{args.k}: {avg_ndcg:.4f}")

# -----------------------------
# Argparse
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir", type=str)
    parser.add_argument("--testdir", type=str)
    parser.add_argument("--model_name", default="svd", choices=["svd", "nmf", "knn"])
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train", choices=["train","eval"])
    parser.add_argument("--n_factor", type=int, default=20)
    parser.add_argument("--n_epoch", type=int, default=15)
    parser.add_argument("--reg_all", type=float, default=0.1)
    parser.add_argument("--k_neighbors", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=4.0)

    args = parser.parse_args()
    main(args)
