import argparse
import time
import random
import os
import json
import gc
from collections import defaultdict, Counter
from dotmap import DotMap
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, MistralForSequenceClassification
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

from data import ConversationClassificationDataset


def evaluate_model(model, dataloader, device, use_majority_voting=False, intermediate_result_path=None):
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    episode_data = defaultdict(lambda: {"text": [], "predictions": [], "label": None})

    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            episode_titles = batch["episode_title"]
            text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)

            total_loss += float(outputs.loss.item())

            top_1_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            for title, t, pred, label in zip(episode_titles, text, top_1_preds, labels.cpu().numpy()):
                episode_data[title]['predictions'].append(int(pred))
                episode_data[title]['text'].append(t)
                # Store label (it should be consistent for each title)
                if episode_data[title]['label'] is None:
                    episode_data[title]['label'] = int(label)

    ground_truth_labels = []
    predictions = []

    if use_majority_voting:
        for data in episode_data.values():
            # Calculate the most common predictions for each unique episode
            most_common_prediction = Counter(data['predictions']).most_common(1)[0][0]
            ground_truth_labels.append(data['label'])
            predictions.append(most_common_prediction)
    else:
        for data in episode_data.values():
            ground_truth_labels.append(data['label'])
            predictions.append(data['predictions'][0])

    # Calculate the accuracy, F1, precision, and recall
    correct = sum([1 for i, j in zip(ground_truth_labels, predictions) if i == j])
    accuracy = correct / len(ground_truth_labels)

    # Precision - Assuming label 1 is the positive class
    true_positives = sum([1 for i, j in zip(ground_truth_labels, predictions) if i == 1 and j == 1])
    false_positives = sum([1 for i, j in zip(ground_truth_labels, predictions) if i == 0 and j == 1])
    precision = true_positives / (true_positives + false_positives)

    # Recall - Assuming label 1 is the positive class
    false_negatives = sum([1 for i, j in zip(ground_truth_labels, predictions) if i == 1 and j == 0])
    recall = true_positives / (true_positives + false_negatives)

    # F1 score
    if precision + recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # Calculate the average loss
    average_loss = total_loss / len(dataloader)

    if intermediate_result_path is not None:
        # Save the intermediate results as a JSON file
        with open(intermediate_result_path, "w") as f:
            json.dump(episode_data, f)

    return average_loss, accuracy, f1, precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for conversation classification.")
    parser.add_argument("--config", type=str, default="./downstream_task/mistral_bnc_wiki_generation_jul_2025.json", help="Path to config file.")
    parser.add_argument("--seed-override", type=int, default=None, help="Override the seed in the config file.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the model.")
    parser.add_argument("--intermediate-result-path", type=str, default="./bnc_jul_2025.json", help="Path to save intermediate results.")
    args = parser.parse_args()

    seed_override = args.seed_override
    eval_only = args.eval_only
    intermediate_result_path = args.intermediate_result_path

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    seed = args.seed if seed_override is None else seed_override
    args.num_epochs = 0 if eval_only else args.num_epochs

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Finished setting up random seeds and TF32 on rank")

    log_writer = SummaryWriter(os.path.join(args.tensorboard_log_dir, f"{args.model_name}_{seed}"))

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    print(f"Finished initializing tokenizer")

    # Initialize the datasets
    train_dataset = ConversationClassificationDataset(
        dataset_path=os.path.join(args.dataset_dir, args.train_filename),
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

    val_dataset = ConversationClassificationDataset(
        dataset_path=os.path.join(args.dataset_dir, args.val_filename),
        tokenizer=tokenizer,
        context_length=args.max_context_length,
        text_source=args.val_text_source,
        label_type=args.label_type,
        keep_first_turn=args.keep_first_turn,
        use_social_orientation=args.use_social_orientation,
        keep_first_n_turns=args.keep_first_n_turns,
        keep_last_turn=args.keep_last_turn,
        do_not_count_section_headers=args.do_not_count_section_headers,
    )

    test_dataset = ConversationClassificationDataset(
        dataset_path=os.path.join(args.dataset_dir, args.test_filename),
        tokenizer=tokenizer,
        context_length=args.max_context_length,
        text_source=args.test_text_source,
        label_type=args.label_type,
        keep_first_turn=args.keep_first_turn,
        use_social_orientation=args.use_social_orientation,
        keep_first_n_turns=args.keep_first_n_turns,
        keep_last_turn=args.keep_last_turn,
        do_not_count_section_headers=args.do_not_count_section_headers,
    )

    # Set the label mapping
    all_labels = list(set(train_dataset.labels + val_dataset.labels + test_dataset.labels))

    for i in [train_dataset, val_dataset, test_dataset]:
        i.set_label_mapping(all_labels)

    # Initialize the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.gradient_accumulation_steps,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // args.gradient_accumulation_steps,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size // args.gradient_accumulation_steps,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )
    print(f"Finished loading datasets, training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")

    # Initialize the model and optimizer
    # Figure out data type
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32

    device = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_type, device_map=device, torch_dtype=torch_dtype, num_labels=len(all_labels), use_flash_attention_2=args.flash_attn)
    if args.load_model:
        model_weights = torch.load(args.load_model, map_location=device)

        if not args.load_classification_head:
            model_weights = {k: v for k, v in model_weights.items() if "classification_head" not in k}
        model.load_state_dict(model_weights, strict=False)

        print(f"Loaded model from {args.load_model}")

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_lora:
        if args.load_lora:
            model = PeftModel.from_pretrained(model, args.load_lora, is_trainable=True)
            model = model.merge_and_unload()
            print(f"Loaded Lora model from {args.load_lora}")

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )
        model = get_peft_model(model, lora_config)
        print(f"Finished initializing Lora model")
        model.print_trainable_parameters()

    if "mistral" in args.model_type or "Qwen2" in args.model_type:
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Finished loading model.")

    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, fused=True)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=int(args.cycle_epochs * len(train_dataloader)),
        cycle_mult=1.0,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=int(args.warmup_epochs * len(train_dataloader)),
        gamma=args.lr_gamma,
    )
    print(f"Initialized optimizer and LR scheduler.")

    best_top_1_accuracy = -1

    batch_loss = 0.0
    step_count = 0
    batch_count = 0
    last_batch_time = time.time()

    for epoch in range(args.num_epochs):
        model.train()
        gc.collect()
        torch.cuda.empty_cache()

        for step_count, batch in enumerate(train_dataloader):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss /= args.gradient_accumulation_steps

            loss.backward()

            batch_loss += float(loss.item())

            if (step_count + 1) % args.gradient_accumulation_steps == 0:
                # Perform gradient clipping and optimizer step
                clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()
                batch_count += 1

                print(f"Epoch {epoch}, Batch {batch_count}, Loss {batch_loss}, samples/sec {args.batch_size / (time.time() - last_batch_time)}")
                log_writer.add_scalar("train_loss", batch_loss, batch_count)

                batch_loss = 0.0
                last_batch_time = time.time()

        model.eval()
        if args.use_lora:
            model.save_pretrained(os.path.join(args.saved_model_path, args.model_name + "_latest"))
        else:
            torch.save(model.state_dict(), os.path.join(args.saved_model_path, f"{args.model_name}_{seed}_latest.pth"))
        print(f"Saved latest model, batch {batch_count}")

        eval_start_time = time.time()
        eval_loss, top_1_accuracy, _, _, _ = evaluate_model(model, valid_dataloader, device, use_majority_voting=args.majority_voting)
        log_writer.add_scalar("eval_loss", eval_loss, batch_count)
        log_writer.add_scalar("eval_top_1_accuracy", top_1_accuracy, batch_count)
        print(f"Validation model name: {args.model_name}, loss: {eval_loss}, top-1 accuracy: {top_1_accuracy}, samples per second: {len(val_dataset) / (time.time() - eval_start_time)}")

        if top_1_accuracy > best_top_1_accuracy:
            best_top_1_accuracy = top_1_accuracy
            if args.use_lora:
                model.save_pretrained(os.path.join(args.saved_model_path, args.model_name + "_best"))
            else:
                torch.save(model.state_dict(), os.path.join(args.saved_model_path, f"{args.model_name}_{seed}_best.pth"))
            print(f"Saved best model, batch {batch_count}")


    # Load the final model and evaluate on the test set
    if args.test_on_best_model:
        if args.use_lora:
            model = None
            model = AutoModelForSequenceClassification.from_pretrained(args.model_type, device_map=device, torch_dtype=torch_dtype, num_labels=len(all_labels), use_flash_attention_2=args.flash_attn)
            model.config.pad_token_id = model.config.eos_token_id
            if args.load_lora:
                model = PeftModel.from_pretrained(model, args.load_lora, is_trainable=True)
                model = model.merge_and_unload()
            model = PeftModel.from_pretrained(model, os.path.join(args.saved_model_path, args.model_name + "_best"), is_trainable=False)
            model = model.merge_and_unload()
        else:
            model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"{args.model_name}_{seed}_best.pth")))
    else:
        if args.use_lora:
            model = None
            model = AutoModelForSequenceClassification.from_pretrained(args.model_type, device_map=device, torch_dtype=torch_dtype, num_labels=len(all_labels), use_flash_attention_2=args.flash_attn)
            if args.load_lora:
                model = PeftModel.from_pretrained(model, args.load_lora, is_trainable=True)
                model = model.merge_and_unload()
            model = PeftModel.from_pretrained(model, os.path.join(args.saved_model_path, args.model_name + "_latest"), is_trainable=False)
            model = model.merge_and_unload()
        else:
            model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f"{args.model_name}_{seed}_latest.pth")))

    test_start_time = time.time()
    test_loss, top_1_accuracy, f1, precision, recall = evaluate_model(model, test_dataloader, device, use_majority_voting=args.majority_voting, intermediate_result_path=intermediate_result_path)

    log_writer.add_scalar("test_loss", test_loss, batch_count)
    log_writer.add_scalar("test_top_1_accuracy", top_1_accuracy, batch_count)
    log_writer.add_scalar("test_f1", f1, batch_count)
    log_writer.add_scalar("test_precision", precision, batch_count)
    log_writer.add_scalar("test_recall", recall, batch_count)

    print(f"Test model name: {args.model_name}, loss: {test_loss}, top-1 accuracy: {top_1_accuracy}, F1: {f1}, precision: {precision}, recall: {recall}, samples per second: {len(test_dataset) / (time.time() - test_start_time)}")

    log_writer.close()
