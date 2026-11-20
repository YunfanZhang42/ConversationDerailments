import csv
import json
import os

if __name__ == "__main__":
    social_orientation_labels = {}

    # List all csv files in the directory
    csv_files = [f for f in os.listdir("./data_collection/final_data/cga_wiki/gpt-4-cga-social-orientation-labels") if f.endswith(".csv")]

    for csv_file in csv_files:
        with open(f"./data_collection/final_data/cga_wiki/gpt-4-cga-social-orientation-labels/{csv_file}", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "conversation_id":
                    continue
                if row[0] not in social_orientation_labels:
                    social_orientation_labels[row[0]] = {}

                social_orientation_labels[row[0]][int(row[2]) - 1] = row[4]

    number_of_turns = []
    data = []
    with open("./data_collection/final_data/cga_wiki/train_circumplex.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    for item in data:
        for i, turn in enumerate(item["conversations"]):
            if item["episode_title"] in social_orientation_labels and i in social_orientation_labels[item["episode_title"]]:
                turn["social_orientation"] = social_orientation_labels[item["episode_title"]][i]

        # Count the number of turns in each conversation
        number_of_turns.append(len(item["conversations"]))

    # Summarize the number of turns in each conversation by frequency
    print({i: number_of_turns.count(i) for i in set(number_of_turns)})


    with open("./data_collection/final_data/cga_wiki/train_circumplex.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    data = []
    with open("./data_collection/final_data/cga_wiki/val_circumplex.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    for item in data:
        for i, turn in enumerate(item["conversations"]):
            if item["episode_title"] in social_orientation_labels and i in social_orientation_labels[item["episode_title"]]:
                turn["social_orientation"] = social_orientation_labels[item["episode_title"]][i]

    with open("./data_collection/final_data/cga_wiki/val_circumplex.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    data = []
    with open("./data_collection/final_data/cga_wiki/test_circumplex.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    for item in data:
        # Do not add social orientation labels for the last turn in the test set
        for i, turn in enumerate(item["conversations"]):
            if item["episode_title"] in social_orientation_labels and i in social_orientation_labels[item["episode_title"]]:
                turn["social_orientation"] = social_orientation_labels[item["episode_title"]][i]

    with open("./data_collection/final_data/cga_wiki/test_circumplex.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
