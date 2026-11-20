import argparse
import time
import random
import os
import json
import gc
import multiprocessing as mp

from dotmap import DotMap
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, LlamaForCausalLM, MistralForCausalLM
from peft import PeftModel

from data import ConversationDataset, CGADataset


def main(rank, world_size, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Finished setting up random seeds and TF32.")

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.generation_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Finished initializing tokenizer.")

    if args.dataset_type == "podcast":
        val_dataset = ConversationDataset(
            dataset_path=os.path.join(args.dataset_dir, args.generation_dataset),
            is_training=False,
            is_generation=True,
            tokenizer=tokenizer,
            context_length=args.max_context_length,
            task_prompt_file_path=args.task_prompt_file_path,
            use_speaker_profile=args.use_speaker_profile,
            use_channel_name=args.use_channel_name,
            use_episode_title=args.use_episode_title,
            generation_min_input_turns=args.generation_min_input_turns,
            generation_max_input_turns=args.generation_max_input_turns,
        )
        tokenizer.truncation_side = "right"
    elif args.dataset_type == "cga_wiki":
        val_dataset = CGADataset(
            dataset_path=os.path.join(args.dataset_dir, args.generation_dataset),
            is_training=False,
            is_generation=True,
            tokenizer=tokenizer,
            context_length=args.max_context_length,
            use_first_n_turns=args.use_first_n_turns,
            use_social_orientation=args.use_social_orientation,
            generation_initial_context_length=args.generation_initial_context_length,
            do_not_count_section_headers=args.do_not_count_section_headers,
        )
        tokenizer.truncation_side = "left"
    else:
        raise NotImplementedError(f"Dataset type {args.dataset_type} not implemented.")

    # Initialize the dataloaders
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=args.generation_per_device_batch_size,
        shuffle=False,
        num_workers=0,
        sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=False),
    )
    print(f"Finished loading datasets, validation samples: {len(val_dataset)}.")

    # Initialize the model and optimizer
    # Figure out data type
    if args.generation_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.generation_dtype == "float32":
        torch_dtype = torch.float32

    device = torch.device(rank)
    if "llama" in args.generation_model:
        model = LlamaForCausalLM.from_pretrained(args.generation_model, device_map=device, torch_dtype=torch_dtype, use_flash_attention_2=args.flash_attn)
    elif "mistral" in args.generation_model or "zephyr" in args.generation_model:
        model = MistralForCausalLM.from_pretrained(args.generation_model, device_map=device, torch_dtype=torch_dtype, use_flash_attention_2=args.flash_attn)

    if args.generation_load_lora:
        model = PeftModel.from_pretrained(model, args.generation_load_lora, is_trainable=True)
        model = model.merge_and_unload()
        print(f"Finished loading Lora model from {args.generation_load_lora}. Lora model merged and unloaded.")

    print(f"Finished loading model and lora.")

    gc.collect()
    torch.cuda.empty_cache()
    model.eval()

    results = []

    for epoch in range(args.generation_epochs):
        # Set the epoch for the sampler
        valid_dataloader.sampler.set_epoch(epoch)

        for batch in valid_dataloader:
            start_time = time.time()
            input_text = batch["input_text"]
            inputs = tokenizer(input_text, return_tensors="pt", padding="longest", truncation=True, max_length=args.max_context_length)
            outputs = model.generate(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                stop_strings=args.stop_sequence,
            )

            for i in range(len(outputs)):
                # Decode the generated ids, keep only the new tokens within the max_new_tokens limit
                new_tokens = outputs[i][inputs.input_ids.shape[1] :]
                answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                sample = {
                    "episode_title": str(batch["episode_title"][i]),
                    "toxic_label": bool(batch["toxic_label"][i].item()),
                    "all_text": str(input_text[i] + answer),
                }

                results.append(sample)

            print(f"Samples per second: {args.generation_per_device_batch_size / (time.time() - start_time)}")

    # Save the results to a temporary file
    print(f"Finished generating samples. Saving the results.")
    with open(os.path.join(args.dataset_dir, f"{args.save_generation_to}_{rank}"), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Finished saving the results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune models on natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./llm/config_mistral_cga_wiki_all_but_last_turns.json", help="Path to config file")
    parser.add_argument("--generation-dataset-override", type=str, default=None, help="Generation dataset override")
    parser.add_argument("--generation-epochs-override", type=int, default=None, help="Generation epochs override")
    parser.add_argument("--save-generation-to-override", type=str, default=None, help="Path to save generation results")
    args = parser.parse_args()

    generation_dataset_override = args.generation_dataset_override
    generation_epochs_override = args.generation_epochs_override
    save_generation_to_override = args.save_generation_to_override

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    if generation_dataset_override is not None:
        args.generation_dataset = generation_dataset_override

    if save_generation_to_override is not None:
        args.save_generation_to = save_generation_to_override

    if generation_epochs_override is not None:
        args.generation_epochs = generation_epochs_override

    mp.set_start_method("forkserver")

    world_size = torch.cuda.device_count()

    children = []
    for rank in range(world_size):
        p = mp.Process(target=main, args=(rank, world_size, args))
        p.start()
        children.append(p)

    for p in children:
        p.join()

    print("Aggregating the results.")

    # I could not make the Queue work with PyTorch and CUDA environment, so I will save the results to a temporary file
    # Aggregate the results from the files
    results = []
    for rank in range(world_size):
        with open(os.path.join(args.dataset_dir, f"{args.save_generation_to}_{rank}"), "r") as f:
            results += f.readlines()

    print(f"Aggregated {len(results)} results.")

    with open(os.path.join(args.dataset_dir, args.save_generation_to), "w") as f:
        for result in results:
            f.write(result)

    # Remove the temporary files
    for rank in range(world_size):
        os.remove(os.path.join(args.dataset_dir, f"{args.save_generation_to}_{rank}"))

    print(f"Finished aggregating the results.")
