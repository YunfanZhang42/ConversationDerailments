import json
import os
import argparse

import torch
from dotmap import DotMap
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random


def remove_last_occurrence(s, sub, replacement=""):
    # Find the last occurrence of the substring
    index = s.rfind(sub)
    if index == -1:
        return s  # Substring not found, return the original string
    # Remove the substring
    return s[:index] + replacement + s[index + len(sub) :]


class ConversationClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        tokenizer=None,
        context_length=1024,
        text_source="real",
        label_type="episode_title",
        keep_first_turn=True,
        keep_first_n_turns=10000,
        keep_last_turn=True,
        data_fraction=1.0,
        use_social_orientation=False,
        do_not_count_section_headers=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length

        # Set up padding and truncation options for the tokenizer

        if "bert" in self.tokenizer.name_or_path:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side = "right"
        elif "bart" in self.tokenizer.name_or_path:
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side = "right"
        elif "mistral" in self.tokenizer.name_or_path or "Qwen2" in self.tokenizer.name_or_path:
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.task_prompt = ""

        self.dataset = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                data = json.loads(line)
                label = data[label_type]
                if "real" in text_source:
                    if "conversations" in data:
                        if not keep_last_turn:
                            data["conversations"] = data["conversations"][:-1]

                        if do_not_count_section_headers:
                            section_header_count = sum([1 for turn in data["conversations"] if turn.get("is_section_header", False)])
                        else:
                            section_header_count = 0

                        data["conversations"] = data["conversations"][:keep_first_n_turns + section_header_count]

                        if use_social_orientation:
                            # Add the social orientation to the label
                            text = "\n\n".join([f"{turn['role']} ({turn.get('social_orientation', 'Unknown')}): {turn['content']}" for turn in data["conversations"]])
                        else:
                            text = "\n\n".join([f"{turn['role']}: {turn['content']}" for turn in data["conversations"]])
                        text = text.strip()
                        self.dataset.append({"text": text, "label": label, "episode_title": data["episode_title"]})
                    elif "all_text" in data:
                        # Remove "Are these comments toxic: no." and "Are these comments toxic: yes." from the end of the text
                        data["all_text"] = remove_last_occurrence(data["all_text"], "Are these comments toxic: no.")
                        data["all_text"] = remove_last_occurrence(data["all_text"], "Are these comments toxic: yes.")
                        # Replace the last "(Last Comment) Commenter" with "Commenter "
                        data["all_text"] = remove_last_occurrence(data["all_text"], "(Last Comment) Commenter", "Commenter ")
                        # Strip the text
                        data["all_text"] = data["all_text"].strip()

                        self.dataset.append({"text": data["all_text"], "label": label, "episode_title": data["episode_title"]})
                if "synthetic" in text_source and "synthetic_conversations" in data:
                    for conversation in data["synthetic_conversations"]:
                        conversation = conversation if keep_first_turn else conversation[1:]
                        text = "\n\n".join([f"{turn['role']}: {turn['content']}" for turn in conversation])
                        text = text.strip()
                        self.dataset.append({"text": text, "label": label, "episode_title": data["episode_title"]})

        # Optionally subsample the dataset
        if data_fraction < 1.0:
            self.dataset = self.dataset[: int(data_fraction * len(self.dataset))]

        # Find all unique labels in this dataset
        self.labels = list(set([sample["label"] for sample in self.dataset]))

    def __len__(self):
        return len(self.dataset)

    def set_label_mapping(self, all_labels):
        # Must be called before __getitem__ to set the label mapping
        # First sort the labels
        all_labels.sort()
        # Then create a mapping from label to index
        self.label_mapping = {label: i for i, label in enumerate(all_labels)}

    def __getitem__(self, idx):
        item = self.dataset[idx]

        data = self.tokenizer(
            item["text"],
            max_length=self.context_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Remove the first dimension of the input_ids, attention_mask.
        data["input_ids"] = data["input_ids"].squeeze(0)
        data["attention_mask"] = data["attention_mask"].squeeze(0)

        # Calculate labels
        data["labels"] = self.label_mapping[item["label"]]

        # Add the episode title to the data
        data["episode_title"] = item["episode_title"]

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test loading natural instructions dataset.")
    parser.add_argument("--config", type=str, default="downstream_task/mistral_cga_wiki_first_4_turns_aug.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    # Test and load the train set
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=True)
    train_dataset = ConversationClassificationDataset(
        dataset_path=os.path.join(args.dataset_dir, args.val_filename),
        tokenizer=tokenizer,
        context_length=args.max_context_length,
        text_source=args.train_text_source,
        label_type=args.label_type,
        keep_first_turn=args.keep_first_turn,
        data_fraction=args.data_fraction,
        use_social_orientation=args.use_social_orientation,
        keep_first_n_turns=args.keep_first_n_turns,
        keep_last_turn=args.keep_last_turn,
        do_not_count_section_headers=args.do_not_count_section_headers,
    )
    torch.set_printoptions(threshold=10_000)

    # Set the label mapping
    train_dataset.set_label_mapping(train_dataset.labels)

    # Print a random subset of samples
    num_samples = 10
    random_samples = random.sample(range(len(train_dataset)), num_samples)
    for i in random_samples:
        # Translate the token IDs back to text, including the special tokens
        print(tokenizer.decode(train_dataset[i]["input_ids"]))
        # Print the number of effective tokens
        print(f"Number of effective tokens: {train_dataset[i]['attention_mask'].sum()}")
        print(f"Label: {train_dataset[i]['labels']}")
