import numpy as np
import pandas as pd
import ast
from collections import defaultdict

def data_retrieval(data_dir):
    num_users = 0
    num_movies = 0
    num_ratings = 0
    train = defaultdict(list)
    validation = defaultdict(list)
    test = defaultdict(list)

    def load_train(path, storage):
        nonlocal num_users, num_movies, num_ratings
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            user = int(row['user_id'])
            movies = ast.literal_eval(row['feature'])
            movies = [x + 1 for x in movies]
            ratings = ast.literal_eval(row['rating'])
            pairs = list(zip(movies, ratings))
            storage[user].extend(pairs)
            num_users = max(num_users, user)
            if movies:
                num_movies = max(num_movies, max(movies))
                num_ratings = max(num_ratings, max(ratings))

    def load_single_label_file(path, label_column, rating_column, storage):
        nonlocal num_users, num_movies, num_ratings
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            user = int(row['user_id'])
            movie = int(row[label_column]) + 1
            rating = int(row[rating_column])
            storage[user].append((movie, rating))
            num_users = max(num_users, user)
            num_movies = max(num_movies, movie)
            num_rating = max(num_ratings, rating)

    load_train(f'{data_dir}/train.parquet', train)
    load_single_label_file(f'{data_dir}/dev.parquet', 'dev_label', 'dev_rating', validation)
    load_single_label_file(f'{data_dir}/test.parquet', 'test_label', 'test_rating', test)

    return [train, validation, test, num_users + 1, num_movies, num_ratings]

class Sampler:
    def __init__(self, users_interacts, num_users, num_movies, num_ratings, batch_size=64, sequence_size=10):
        self.users_interacts = users_interacts
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_ratings = num_ratings
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.user_ids = np.arange(0, self.num_users, dtype=np.int32)
        np.random.seed(1601)
        np.random.shuffle(self.user_ids)
        self.index = 0

    def random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def sample(self, user_id):
        while len(self.users_interacts[user_id]) <= 1:
            user_id = np.random.randint(0, self.num_users)

        seq_movie = np.zeros([self.sequence_size], dtype=np.int32)
        pos_movie = np.zeros([self.sequence_size], dtype=np.int32)
        neg_movie = np.zeros([self.sequence_size], dtype=np.int32)
        seq_rating = np.zeros([self.sequence_size], dtype=np.int32)
        next_movie = self.users_interacts[user_id][-1][0]
        next_rating = self.users_interact[user_id][-1][1]
        next_id = self.sequence_size - 1

        movie_set = {m for m, _ in self.users_interacts[user_id]}
        for movie, rating in reversed(self.users_interacts[user_id][:-1]):
            seq_movie[next_id] = movie
            seq_rating[next_id] = rating

            pos_movie[next_id] = next_movie
            pos_rating[next_id] = next_rating

            if next_movie != 0:
                neg_movie[next_id] = self.random_neq(1, self.num_movies + 1, movie_set)

            next_movie, next_rating = movie, rating
            next_id -= 1
            if next_id == -1:
                break

        return user_id, seq_movie, seq_rating, pos_movie, pos_rating, neg_movie

    def next_batch(self):
        if self.index + self.batch_size > len(self.user_ids):
            np.random.shuffle(self.user_ids)
            self.index = 0

        batch_user_ids = self.user_ids[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        batch = [self.sample(uid) for uid in batch_user_ids]
        return list(zip(*batch)) 