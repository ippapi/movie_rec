import argparse
import pandas as pd
import time
import logging
from surprise import Dataset, Reader, SVD
from surprise import dump
from surprise import accuracy
from collections import defaultdict

class CustomLogger:
    def __init__(self, name=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s',
                                          datefmt='%H:%M')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

def get_top_k(predictions, k=10):
    top_k = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_k[uid].append((iid, est))
    for uid in top_k:
        top_k[uid] = sorted(top_k[uid], key=lambda x: x[1], reverse=True)[:k]
    return top_k

def main(args):
    log = CustomLogger()
    logging.info("Loading data...")
    train_df = pd.read_parquet(args.traindir)
    test_df = pd.read_parquet(args.testdir)

    reader = Reader(rating_scale=(1,5))
    train_data = Dataset.load_from_df(train_df[['user_idx','item_idx','rating']], reader)
    trainset = train_data.build_full_trainset()

    if args.load_model:
        logging.info(f"Loading model from {args.load_model} ...")
        algo, _ = dump.load(args.load_model)
    else:
        algo = SVD(n_factors=20, n_epochs=15, reg_all=0.1)

    start_time = time.time()
    logging.info("Training model...")
    algo.fit(trainset)
    logging.info(f"Training done in {time.time() - start_time:.2f}s")

    if args.save_model:
        logging.info(f"Saving model to {args.save_model} ...")
        dump.dump(args.save_model, algo=algo)

    if args.mode == "eval" and not test_df.empty:
        logging.info("Evaluating model...")
        testset = list(zip(test_df['user_id'], test_df['movie_id'], test_df['rating']))
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions)
        logging.info(f"RMSE on eval set: {rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir", type = str, required=True)
    parser.add_argument("--evaldir", type = str)
    parser.add_argument("--save_model", type = str, default=None)
    parser.add_argument("--load_model", type = str, default=None)
    parser.add_argument("--log_path", type = str)
    parser.add_argument("--mode", type=str, default="train", choices=["train","eval"])
    parser.add_argument("--n_factor", default = 20, type = int)
    parser.add_argument("--n_epoch", default = 15, type = int)
    parser.add_argument("--rag_all", default = 0.1, type = float)

    args = parser.parse_args()

    main(args)
