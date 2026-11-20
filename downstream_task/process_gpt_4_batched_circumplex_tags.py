import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./data_collection/final_data/cga_wiki/val_intermediate.jsonl")
    parser.add_argument("--openai-batch-output", type=str, default="/home/yunfan/Downloads/val_batched_results.jsonl")
    parser.add_argument("--result-file", type=str, default="./data_collection/final_data/cga_wiki/val_new_circumplex.jsonl")
    parser.add_argument("--skip-last-turn", action="store_true")
    args = parser.parse_args()

    # Load the dataset
    dataset = {}
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset[data["episode_title"]] = data

    success_count = 0

    # Load the OpenAI batch output
    with open(args.openai_batch_output, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)

                finish_status = data["response"]["body"]["choices"][0]["finish_reason"]

                if finish_status != "stop":
                    continue

                id = data["custom_id"]
                result_str = data["response"]["body"]["choices"][0]["message"]["content"]
                result_str = result_str.strip()

                circumplex_tags = result_str.split("\n")

                # Iterate over the circumplex tags, delete everything before and including ":"
                circumplex_tags = [tag.split(":")[-1].strip() for tag in circumplex_tags]

                # Only keep the non-empty tags
                circumplex_tags = [tag for tag in circumplex_tags if tag]

                if args.skip_last_turn:
                    assert len(circumplex_tags) == len(dataset[id]["conversations"]) - 1, f"Expected {len(dataset[id]['conversations']) - 1} tags, got {len(circumplex_tags)}"

                circumplex_tags_iter = iter(circumplex_tags)

                # Update the dataset with the circumplex tags
                for turn in dataset[id]["conversations"]:
                    if "social_orientation" not in turn or not turn["social_orientation"]:
                        try:
                            turn["social_orientation"] = next(circumplex_tags_iter)
                        except StopIteration:
                            break

                success_count += 1

            except Exception as e:
                print(e)
                print(line)
                continue

    # Save the updated dataset
    with open(args.result_file, "w", encoding="utf-8") as f:
        for item in dataset.values():
            f.write(json.dumps(item) + "\n")

    print(f"Successfully updated {success_count} items. Total items: {len(dataset)}")
