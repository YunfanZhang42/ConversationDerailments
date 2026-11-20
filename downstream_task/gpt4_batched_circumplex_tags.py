import argparse
import json
import random
import tiktoken


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(tiktoken.encoding_for_model(encoding_name).encode(string))
    return num_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", type=str, default="./data_collection/final_data/cga_wiki/train_intermediate.jsonl", help="The input JSON file to be converted to OpenAI batch format"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./data_collection/final_data/cga_wiki/train_batched.jsonl",
        help="The output file to save the converted data in OpenAI batch format",
    )
    parser.add_argument("--system-prompt-file", type=str, default="./downstream_task/gpt4_default_system_prompt.txt")
    parser.add_argument("--question-prompt-file", type=str, default="./downstream_task/social_orientation_user_prompt_pass_2_cga_wiki_new.txt")
    parser.add_argument("--api-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--skip-last-turn", action="store_true")
    parser.add_argument("--populate-existing-circumplex-tags", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=423680)
    args = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(args.seed)

    # Read the input JSON file. Allow for UTF-8 encoding.
    with open(args.input_file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    # Read the system and question prompts
    with open(args.system_prompt_file, "r", encoding="utf-8") as file:
        system_prompt_template = file.read().strip()

    with open(args.question_prompt_file, "r", encoding="utf-8") as file:
        question_prompt_template = file.read().strip()

    # Prepare the data for the OpenAI batch format
    openai_batch_data = []
    for comment in data:
        turns = comment["conversations"]
        if args.skip_last_turn:
            turns = turns[:-1]

        for turn in turns:
            if "content" in turn:
                turn["content"] = turn["content"].replace("\n", " ")

        comment_text = "\n".join([f"Turn {i+1}: {turn['role']}: {turn['content']}" for i, turn in enumerate(turns)])
        comment_text = comment_text.strip()

        if args.populate_existing_circumplex_tags:
            # Extract the circumplex tags from the existing tags
            circumplex_tags = [turn.get("social_orientation", "") for turn in turns]
            circumplex_tags = [tag for tag in circumplex_tags if tag]
            circumplex_tags = [f"Turn {i+1}: {tag}" for i, tag in enumerate(circumplex_tags)]
            circumplex_tags = "\n".join(circumplex_tags)
        else:
            circumplex_tags = ""

        # Skip the comment if there are no conversations
        if len(comment_text) == 0:
            continue

        question_prompt = question_prompt_template.format(
            comment=comment_text, tags=circumplex_tags
        )
        question_prompt = question_prompt.strip()

        openai_batch_data.append(
            {
                "custom_id": comment["episode_title"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [{"role": "system", "content": system_prompt_template}, {"role": "user", "content": question_prompt}],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                },
            }
        )

    # Randomize the order of the requests
    random.shuffle(openai_batch_data)

    # Save the requests in JSONL format
    with open(args.output_file, "w", encoding="utf-8") as file:
        for comment in openai_batch_data:
            file.write(json.dumps(comment, ensure_ascii=False) + "\n")

    # Print the number of tokens in the system and question prompts
    total_tokens = 0
    system_prompt_tokens = num_tokens_from_string(system_prompt_template, args.model)
    for i, comment in enumerate(openai_batch_data):
        question_prompt_tokens = num_tokens_from_string(comment["body"]["messages"][1]["content"], args.model)
        total_tokens += system_prompt_tokens + question_prompt_tokens

    print(f"Total samples: {len(openai_batch_data)}")
    print(f"Total tokens: {total_tokens}")
