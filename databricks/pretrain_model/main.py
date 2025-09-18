import argparse
import pandas as pd
import time
from surprise import Dataset, Reader, SVD, NMF, KNNWithMeans
from surprise import dump
from surprise import accuracy
import sys

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
        else:
            raise ValueError(f"Unknown model_name: {args.model_name}")
        log.info(f"Created new model: {args.model_name}")

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

    if args.mode == "eval" and args.testdir:
        log.info("Loading testing data...")
        test_df = pd.read_parquet(args.testdir)
        log.info("Evaluating model...")
        test_set = list(zip(test_df['user_id'], test_df['movie_id'], test_df['rating']))
        predictions = algo.test(test_set)
        rmse = accuracy.rmse(predictions)
        log.info(f"RMSE on eval set: {rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir", type=str)
    parser.add_argument("--testdir", type=str)
    parser.add_argument("--model_name", default = "svd", choices = ["svd", "nmf", "knn"])
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train", choices=["train","eval"])
    parser.add_argument("--n_factor", type=int, default=20)
    parser.add_argument("--n_epoch", type=int, default=15)
    parser.add_argument("--reg_all", type=float, default=0.1)
    parser.add_argument("--k_neighbors", type = int, default = 20)

    args = parser.parse_args()
    main(args)

