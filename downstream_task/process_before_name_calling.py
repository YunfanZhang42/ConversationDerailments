import json
import os
import argparse
import random
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process before name calling")
    parser.add_argument("--sampled-thread-folder", type=str, default="/home/yunfan/Downloads/sampled-threads-ah-delta-context3")
    parser.add_argument("--all-posts-folder", type=str, default="/home/yunfan/Downloads/cmv-full-2017-09-22")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output-folder", type=str, default="./data_collection/final_data/bnc")
    args = parser.parse_args()

    posts_to_add = {}

    random.seed(23663)

    # First, we need to add the first round of benign posts to benigh threads
    # Iterate through all JSON files in the sampled thread folder
    for file in os.listdir(args.sampled_thread_folder):
        if file.endswith(".json"):
            # Split the file name by the underscore character
            split_file = file.split("_")
            thread_type = split_file[1]
            if thread_type == "delta":
                # Read the first line as json
                with open(os.path.join(args.sampled_thread_folder, file), "r") as f:
                    post = json.loads(f.readline())
                    if "parent_id" in post:
                        parent_id = post["parent_id"]
                        posts_to_add[parent_id] = None

    # Iterate through all JSON files in the all posts folder
    for file in os.listdir(args.all_posts_folder):
        if file.endswith(".json"):
            # Read every line as json
            with open(os.path.join(args.all_posts_folder, file), "r") as f:
                for line in f:
                    post = json.loads(line)
                    post_id = post["name"]
                    if post_id in posts_to_add:
                        posts_to_add[post_id] = post

    print(f"Finished adding posts")

    # Then, we need to iterate the sampled posts again to add the benign posts to the benign threads
    all_threads = []
    for file in os.listdir(args.sampled_thread_folder):
        if file.endswith(".json"):
            split_file = file.split("_")
            thread_type = split_file[1]
            # Read every line as json
            with open(os.path.join(args.sampled_thread_folder, file), "r") as f:
                thread = []
                for line in f:
                    post = json.loads(line)
                    thread.append(post)

                if thread_type == "delta":
                    parent_id = thread[0]["parent_id"]
                    # Prepend the benign posts to the thread
                    if parent_id in posts_to_add and posts_to_add[parent_id] is not None:
                        thread.insert(0, posts_to_add[parent_id])

                episode_title = ".".join([i["name"] for i in thread])
                conversations = [{"role": i["author_name"], "content": re.sub(r'\n+', '\n', i["body"]).strip()} for i in thread]
                toxic_label = True if thread_type == "ah" else False

                all_threads.append({
                    "episode_title": episode_title,
                    "conversations": conversations,
                    "toxic_label": toxic_label
                })

    print(f"Finished creating threads")

    # Split the threads into train, val, and test sets
    random.shuffle(all_threads)
    train_cutoff = int(len(all_threads) * args.train_ratio)
    val_cutoff = int(len(all_threads) * (args.train_ratio + args.val_ratio))

    train_threads = all_threads[:train_cutoff]
    val_threads = all_threads[train_cutoff:val_cutoff]
    test_threads = all_threads[val_cutoff:]

    # Write the threads to the output folder as JSONL files
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "train.jsonl"), "w") as f:
        for thread in train_threads:
            f.write(json.dumps(thread) + "\n")

    with open(os.path.join(args.output_folder, "val.jsonl"), "w") as f:
        for thread in val_threads:
            f.write(json.dumps(thread) + "\n")

    with open(os.path.join(args.output_folder, "test.jsonl"), "w") as f:
        for thread in test_threads:
            f.write(json.dumps(thread) + "\n")
