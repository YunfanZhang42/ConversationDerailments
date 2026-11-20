import json
import random
import os
import re
import argparse
from convokit import Corpus, download


speaker_indices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def clean_text_regex(text):
    # Remove all new lines in the text, and replace adjacent new lines with a single space
    text = re.sub(r"\s+", " ", text)
    # Replace HTML entity with empty string
    text = re.sub(r'&gt;', '', text)
    # Replace curly single quotes
    text = re.sub(r'[\u2018\u2019]', "'", text)
    # Replace curly double quotes
    text = re.sub(r'[\u201c\u201d]', '"', text)
    text = re.sub(r'[\u00a0]', "", text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="conversations-gone-awry-corpus")
    # parser.add_argument("--annotation-years", type=int, nargs="+", default=[])
    parser.add_argument("--label-name", type=str, default="conversation_has_personal_attack")
    parser.add_argument("--output-path", type=str, default="./data_collection/final_data/cga_wiki/")
    args = parser.parse_args()
    corpus = Corpus(filename=download(args.dataset))

    # If we are only interested in the original CGA dataset, we can filter the corpus by the annotation year
    # corpus = corpus.filter_conversations_by(lambda convo: convo.meta['annotation_year'] == '2018')

    print(corpus.print_summary_stats())

    train_conversations = []
    val_conversations = []
    test_conversations = []

    for convo in corpus.iter_conversations():
        speakers = list(set([speaker.id for speaker in convo.iter_speakers()]))
        turns = list(convo.iter_utterances())

        conversation = []

        conversation_id = convo.id

        for i, turn in enumerate(turns):
            conversation.append({"role": turn.speaker.id, "content": turn.text.strip(), "is_section_header": turn.meta["is_section_header"]})

            # If the turn is not the last turn but has a personal attack, print
            if i < len(turns) - 1 and turn.meta["comment_has_personal_attack"]:
                print("Personal attack found in the middle of the conversation.")
                print(turn.text)

        if convo.meta["split"] == "train":
            train_conversations.append({"episode_title": conversation_id, "conversations": conversation, "toxic_label": convo.meta[args.label_name]})
        elif convo.meta["split"] == "val":
            val_conversations.append({"episode_title": conversation_id, "conversations": conversation, "toxic_label": convo.meta[args.label_name]})
        elif convo.meta["split"] == "test":
            test_conversations.append({"episode_title": conversation_id, "conversations": conversation, "toxic_label": convo.meta[args.label_name]})

    # Shuffle the training data
    random.shuffle(train_conversations)

    # Save the data
    with open(os.path.join(args.output_path, "train_username_2018.jsonl"), "w", encoding="utf-8") as f:
        for item in train_conversations:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(args.output_path, "val_username_2018.jsonl"), "w", encoding="utf-8") as f:
        for item in val_conversations:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(args.output_path, "test_username_2018.jsonl"), "w", encoding="utf-8") as f:
        for item in test_conversations:
            f.write(json.dumps(item) + "\n")
