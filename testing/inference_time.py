import time
import requests
import argparse

def main(port):
    users = ["U1", "U2", "U3", ...]

    start = time.time()
    for u in users:
        requests.get(f"http://localhost:{port}/api/recommend?user_id={u}")
    end = time.time()

    print("Average per user:", (end-start)/len(users), "seconds")

if __name__ == "__main__":
    
    main()
