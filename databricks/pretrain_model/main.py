import argparse
import pandas as pd
import time
from surprise import Dataset, Reader, SVD
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
        algo, _ = dump.load(args.load_model)
    else:
        algo = SVD(n_factors=args.n_factor, n_epochs=args.n_epoch, reg_all=args.reg_all)

    if args.mode == "train" and args.traindir:
        log.info("Loading training data...")
        train_df = pd.read_parquet(args.traindir)
        log.info("Training model...")
        algo.fit(trainset)
        log.info(f"Training done in {time.time() - start_time:.2f}s")

        if args.save_model:
            log.info(f"Saving model to {args.save_model} ...")
            dump.dump(args.save_model, algo=algo)

    if args.mode == "eval" and args.testdir:
        log.info("Loading testing data...")
        test_df = pd.read_parquet(args.testdir)
        log.info("Evaluating model...")
        testset = list(zip(test_df['user_id'], test_df['movie_id'], test_df['rating']))
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions)
        log.info(f"RMSE on eval set: {rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir", type=str)
    parser.add_argument("--testdir", type=str)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train", choices=["train","eval"])
    parser.add_argument("--n_factor", type=int, default=20)
    parser.add_argument("--n_epoch", type=int, default=15)
    parser.add_argument("--reg_all", type=float, default=0.1)

    args = parser.parse_args()
    main(args)

