import sys
import copy
import numpy as np
import random

def evaluate(model, dataset, sequence_size = 10, k = 1):
    [train, validation, test, num_users, num_movies, num_ratings] = copy.deepcopy(dataset)

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0
    valid_user = 0.0

    users = range(0, num_users)

    for user in users:
        if len(train[user]) < 1 or len(test[user]) < 1:
            continue

        seq_movie = np.zeros([sequence_size], dtype=np.int32)
        seq_rating = np.zeros([sequence_size], dtype=np.int32)
        next_index = sequence_size - 1
        seq_movie[next_index] = validation[user][0][0] if len(validation[user]) > 0 else 0
        seq_rating[next_index] = validation[user][0][1] if len(validation[user]) > 0 else 0
        next_index -= 1
        for movie, rating in reversed(train[user]):
            seq_movie[next_index] = movie
            seq_rating[next_index] = rating
            next_index -= 1
            if next_index == -1:
                break

        interacted_movies = set(train[user])
        interacted_movies.add(0)
        predict_movies = [test[user][0]]

        all_movies = set(range(1, num_movies + 1))
        available_movies = list(all_movies - interacted_movies - set(predict_movies))
        num_needed = 100 - len(predict_movies)
        predict_movies += random.sample(available_movies, min(num_needed, len(available_movies)))

        predictions = -model.predict(
            np.array([user]),                
            np.array([seq_movie]),           
            np.array([seq_rating]),          
            np.array(predict_movies)         
        )
        predictions = predictions[0]

        rank = (predictions > predictions[0]).sum().item()
        valid_user += 1

        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
            RECALL += 1

        if valid_user % 10000 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user != 0:
        return {
            "NDCG@k": NDCG / valid_user,
            "Hit@k": HIT / valid_user,
            "Recall@k": RECALL / valid_user
        }
    else:
        return {
            "NDCG@k": 0.0,
            "Hit@k": 0.0,
            "Recall@k": 0.0
        }

def evaluate_validation(model, dataset, sequence_size = 10, k = 1):
    [train, validation, test, num_users, num_movies] = copy.deepcopy(dataset)

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0
    valid_user = 0.0

    users = range(0, num_users)

    for user in users:
        if len(train[user]) < 1 or len(test[user]) < 1:
            continue

        seq_movie = np.zeros([sequence_size], dtype=np.int32)
        seq_rating = np.zeros([sequence_size], dtype=np.int32)
        next_index = sequence_size - 1
        for movie, rating in reversed(train[user]):
            seq_movie[next_index] = movie
            seq_rating[next_index] = rating
            next_index -= 1
            if next_index == -1:
                break

        interacted_movies = set(train[user])
        interacted_movies.add(0)
        predict_movies = [validation[user][0]]

        all_movies = set(range(1, num_movies + 1))
        available_movies = list(all_movies - interacted_movies - set(predict_movies))
        num_needed = 100 - len(predict_movies)
        predict_movies += random.sample(available_movies, min(num_needed, len(available_movies)))

        predictions = -model.predict(
            np.array([user]),                
            np.array([seq_movie]),           
            np.array([seq_rating]),          
            np.array(predict_movies)         
        )
        predictions = predictions[0]

        rank = (predictions > predictions[0]).sum().item()
        valid_user += 1

        if rank < k:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
            RECALL += 1

        if valid_user % 10000 == 0:
            print('.', end="")
            sys.stdout.flush()

    if valid_user != 0:
        return {
            "NDCG@k": NDCG / valid_user,
            "Hit@k": HIT / valid_user,
            "Recall@k": RECALL / valid_user
        }
    else:
        return {
            "NDCG@k": 0.0,
            "Hit@k": 0.0,
            "Recall@k": 0.0
        }