import argparse
import time
import json
import os
import random

import torch
import numpy as np
from dotmap import DotMap
from torch.utils.data import Dataset
from transformers import AutoTokenizer


HF_LOSS_IGNORE_TOKEN_ID = -100


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        is_training=True,
        is_generation=False,
        tokenizer=None,
        context_length=2048,
        min_turns=4,
        max_turns=6,
        task_prompt_file_path="",
        use_speaker_profile=False,
        use_channel_name=False,
        use_episode_title=False,
        generation_min_input_turns=1,
        generation_max_input_turns=1,
    ):
        super().__init__()
        # Load the JSONL file
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.is_training = is_training
        self.is_generation = is_generation
        self.use_speaker_profile = use_speaker_profile
        self.use_channel_name = use_channel_name
        self.use_episode_title = use_episode_title
        self.generation_min_input_turns = generation_min_input_turns
        self.generation_max_input_turns = generation_max_input_turns

        # Set up padding and truncation options for the tokenizer
        self.tokenizer.truncation_side = "right"
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.task_prompt = ""
        if task_prompt_file_path:
            with open(task_prompt_file_path, "r") as f:
                self.task_prompt = f.read().strip()

        self.sections = []
        for conversation in data:
            # Retrieve the channel name.
            channel_name = ""
            if self.use_channel_name:
                channel_name = conversation["channel_name"].strip()

            # Retrieve the episode title.
            episode_title = ""
            if self.use_episode_title:
                episode_title = conversation["episode_title"].strip()

            # Retrieve the conversation level speaker profile.
            speaker_profile = ""
            if self.use_speaker_profile and conversation["speaker_profile"]:
                speaker_profile = conversation["speaker_profile"].strip()

            # Remove the turn numbers from the conversation to make it compatible with the hugging face tokenizer
            section = conversation["conversations"]
            section = [{"role": turn["role"], "content": turn["content"].strip()} for turn in section]

            # If the current mode is training, we will need to re-chop the data
            # Chop the conversation into sections by selecting a random number of turns between "min_turns" and "max_turns"
            if self.is_training:
                start = 0
                while start < len(section):
                    end = min(start + random.randint(min_turns, max_turns), len(section))
                    self.sections.append(
                        {"conversations": section[start:end], "speaker_profile": speaker_profile, "channel_name": channel_name, "episode_title": episode_title}
                    )
                    start = end
            # If the current mode is generation, we will need to take the first [generation_min_input_turns, generation_max_input_turns] turns
            elif self.is_generation:
                num_turns = random.randint(self.generation_min_input_turns, self.generation_max_input_turns)
                self.sections.append(
                    {
                        "conversations": section[:num_turns],
                        "speaker_profile": speaker_profile,
                        "channel_name": channel_name,
                        "episode_title": episode_title,
                        "chapter_uuid": conversation["chapter_uuid"],
                    }
                )
            # If the current mode is validation, we will use the whole conversation
            else:
                self.sections.append(
                    {"conversations": section, "speaker_profile": speaker_profile, "channel_name": channel_name, "episode_title": episode_title}
                )

    def __len__(self):
        return len(self.sections)

    def __getitem__(self, idx):
        section = self.sections[idx]

        # Formulate the prompt based on whether the task prompt and speaker profile are used. We do not calculate loss for the prompt.
        prompt = ""
        if self.task_prompt:
            prompt += self.task_prompt + "\n\n"

        if self.use_channel_name:
            prompt += "Channel Name: " + section["channel_name"] + "\n\n"

        if self.use_episode_title:
            prompt += "Episode Title: " + section["episode_title"] + "\n\n"

        if self.use_speaker_profile:
            prompt += "Speaker Profiles:\n" + section["speaker_profile"] + "\n\n"

        # Formulate the target based on the conversations, we only calculate loss for the target.
        # We use the special token "\n\n" as the stop generation sequence for conversation turns, so it must be appended to the end of each turn.
        target = "Conversations:\n" + "\n\n".join([f"{turn['role']}: {turn['content']}" for turn in section["conversations"]]) + "\n\n"

        # Concatenate the prompt and target
        input_text = prompt + target

        if self.is_generation:
            # If the current mode is generation, we will not tokenize and pad, so we return the input text as a string.
            data = section
            data["input_text"] = input_text
        else:
            # For training/validation, we will tokenize and pad the input text and calculate the loss for the target.
            # Tokenize the input text
            data = self.tokenizer(
                input_text,
                max_length=self.context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Remove the first dimension of the input_ids, attention_mask.
            data["input_ids"] = data["input_ids"].squeeze(0)
            data["attention_mask"] = data["attention_mask"].squeeze(0)

            # Initialize labels with ignore token id for all tokens
            labels = torch.full(data["input_ids"].shape, HF_LOSS_IGNORE_TOKEN_ID)

            # Calculate the number of tokens in the prompt by subtracting the number of tokens in the target from the total number of tokens
            target_tokens = self.tokenizer.encode(target, max_length=self.context_length, truncation=True, add_special_tokens=False)
            prompt_length = data["input_ids"].size(0) - len(target_tokens)
            labels[prompt_length:] = data["input_ids"][prompt_length:]
            data["labels"] = labels

        return data


class CGADataset(Dataset):
    def __init__(
        self,
        dataset_path,
        is_training=True,
        is_generation=False,
        tokenizer=None,
        context_length=2048,
        predict_label=False,
        use_first_n_turns=1000,
        use_social_orientation=False,
        generation_initial_context_length=1536,
        do_not_count_section_headers=False,
    ):
        super().__init__()
        # Load the JSONL file
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.is_training = is_training
        self.is_generation = is_generation
        self.predict_label = predict_label
        self.use_first_n_turns = use_first_n_turns
        self.use_social_orientation = use_social_orientation
        self.generation_initial_context_length = generation_initial_context_length
        self.do_not_count_section_headers = do_not_count_section_headers

        # Set up padding and truncation options for the tokenizer
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sections = []
        for conversation in data:
            # Remove the turn numbers from the conversation to make it compatible with the hugging face tokenizer
            section = conversation["conversations"]
            section = [{"role": turn["role"], "content": turn["content"].strip(), "social_orientation": turn.get("social_orientation", "Unknown"), "is_section_header": turn.get("is_section_header", False)} for turn in section]

            self.sections.append({"conversations": section, "episode_title": conversation["episode_title"], "toxic_label": conversation["toxic_label"]})

    def __len__(self):
        return len(self.sections)

    def __getitem__(self, idx):
        section = self.sections[idx]

        data = {}

        conversations = section["conversations"]

        last_turn_social_orientation = conversations[-1].get("social_orientation", "Unknown")
        last_turn_speaker = conversations[-1]["role"]
        last_turn_content = conversations[-1]["content"]

        conversations = conversations[:-1]

        if self.do_not_count_section_headers:
            section_header_count = sum([1 for turn in conversations if turn.get("is_section_header", False)])
        else:
            section_header_count = 0

        first_n_turns = conversations[:self.use_first_n_turns + section_header_count]
        rest_turns = conversations[self.use_first_n_turns + section_header_count:]

        text_to_complete = []

        # Use the first n turns as the input text
        if self.use_social_orientation:
            input_text = "\n\n".join([f"{turn['role']} ({turn['social_orientation']}): {turn['content']}" for turn in first_n_turns])
        else:
            input_text = "\n\n".join([f"{turn['role']}: {turn['content']}" for turn in first_n_turns])

        # Use the rest of the turns as the text to complete
        if self.use_social_orientation:
            text_to_complete.append("\n\n".join([f"{turn['role']} ({turn['social_orientation']}): {turn['content']}" for turn in rest_turns]))
        else:
            text_to_complete.append("\n\n".join([f"{turn['role']}: {turn['content']}" for turn in rest_turns]))

        # The last turn is always included in the text to complete
        if self.use_social_orientation:
            text_to_complete.append(f"(Last Comment) {last_turn_speaker} ({last_turn_social_orientation}): {last_turn_content}")
        else:
            text_to_complete.append(f"(Last Comment) {last_turn_speaker}: {last_turn_content}")

        if self.predict_label:
            toxic_verdict_label = "yes" if section["toxic_label"] == 1 else "no"
            text_to_complete.append(f"Are these comments toxic: {toxic_verdict_label}.")

        # Only keep non-empty text to complete
        text_to_complete = [text for text in text_to_complete if text]

        text_to_complete = "\n\n" + "\n\n".join(text_to_complete)

        if self.is_training:
            input_text = input_text + text_to_complete

            # Tokenize the context
            data = self.tokenizer(
                input_text,
                max_length=self.context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # Remove the first dimension of the input_ids, attention_mask.
            data["input_ids"] = data["input_ids"].squeeze(0)
            data["attention_mask"] = data["attention_mask"].squeeze(0)

            # Initialize labels with ignore token id for all tokens
            labels = torch.full(data["input_ids"].shape, HF_LOSS_IGNORE_TOKEN_ID)

            # Supervise only on tokens in the text to complete
            text_to_complete_tokens = self.tokenizer.encode(text_to_complete, max_length=self.context_length, truncation=True, add_special_tokens=False)
            labels[-len(text_to_complete_tokens) + 1:] = data["input_ids"][-len(text_to_complete_tokens) + 1:]

            data["labels"] = labels

        if self.is_generation:
            # If we only predict the last turn, hint the model that this should be the last turn
            # Otherwise, only hint the model that there should be a next turn
            if self.use_first_n_turns >= 10000:
                input_text += "\n\n(Last Comment)"
            else:
                input_text += "\n\n"

            data = self.tokenizer(
                input_text,
                max_length=self.generation_initial_context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            data["input_text"] = input_text
            data["toxic_label"] = section["toxic_label"]
            data["episode_title"] = section["episode_title"]
            data["text_to_complete"] = text_to_complete

        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test loading natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./llm/config_mistral_cga_wiki_first_2_turns.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    # Test and load the train set
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=True)
    train_dataset = CGADataset(
        dataset_path=os.path.join(args.dataset_dir, args.train_filename),
        is_training=True,
        is_generation=False,
        tokenizer=tokenizer,
        context_length=args.max_context_length,
        predict_label=args.predict_label,
        use_first_n_turns=args.use_first_n_turns,
        use_social_orientation=args.use_social_orientation,
        do_not_count_section_headers=args.do_not_count_section_headers,
    )
    torch.set_printoptions(threshold=10_000)

    # Print the first few examples
    for i in range(10):
        # print(train_dataset[i])
        # Translate the token IDs back to text, including the special tokens
        print(tokenizer.decode(train_dataset[i]["input_ids"]))
        print("Calculate loss for the following tokens:")
        print(tokenizer.decode(train_dataset[i]["labels"][train_dataset[i]["labels"] != HF_LOSS_IGNORE_TOKEN_ID]))

    print("Length of train set:", len(train_dataset))

    # Iterate over the whole train set and print the statistics of valid, not padded tokens
    valid_tokens = []
    for i in range(len(train_dataset)):
        valid_tokens.append(train_dataset[i]["attention_mask"].sum().item())

    print(f"Average number of valid tokens: {np.mean(valid_tokens)}")
    print(f"Standard deviation of valid tokens: {np.std(valid_tokens)}")
    print(f"Minimum number of valid tokens: {np.min(valid_tokens)}")
    print(f"Maximum number of valid tokens: {np.max(valid_tokens)}")

    # Print the percentage of samples with >= max context length tokens
    print(f"Percentage of samples exceeding max context length tokens: {np.mean(np.array(valid_tokens) >= args.max_context_length) * 100}")

    # Print the top 1% of valid tokens
    print(f"Top 1% of valid tokens: {np.percentile(valid_tokens, 99)}")
    # Print the top 5% of valid tokens
    print(f"Top 5% of valid tokens: {np.percentile(valid_tokens, 95)}")
    # Print the top 10% of valid tokens
    print(f"Top 10% of valid tokens: {np.percentile(valid_tokens, 90)}")
