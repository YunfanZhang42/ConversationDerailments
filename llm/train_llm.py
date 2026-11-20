import argparse
import time
import random
import os
import json
import gc

from dotmap import DotMap
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, LlamaForCausalLM, MistralForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from data import ConversationDataset, CGADataset


def setup(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    setup(rank, world_size, args)
    print(f"Initialized process group with rank {rank} and world size {world_size}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Finished setting up random seeds and TF32 on rank {rank}")

    # Set up the tensorboard on rank 0
    if rank == 0:
        log_writer = SummaryWriter(os.path.join(args.tensorboard_log_dir, args.model_name))

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    print(f"Finished initializing tokenizer on rank {rank}")

    if args.dataset_type == "podcast":
        train_dataset = ConversationDataset(
            dataset_path=os.path.join(args.dataset_dir, args.train_filename),
            is_training=True,
            is_generation=False,
            tokenizer=tokenizer,
            context_length=args.max_context_length,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            task_prompt_file_path=args.task_prompt_file_path,
            use_speaker_profile=args.use_speaker_profile,
            use_channel_name=args.use_channel_name,
            use_episode_title=args.use_episode_title,
        )

        val_dataset = ConversationDataset(
            dataset_path=os.path.join(args.dataset_dir, args.val_filename),
            is_training=False,
            is_generation=False,
            tokenizer=tokenizer,
            context_length=args.max_context_length,
            task_prompt_file_path=args.task_prompt_file_path,
            use_speaker_profile=args.use_speaker_profile,
            use_channel_name=args.use_channel_name,
            use_episode_title=args.use_episode_title,
        )
    elif args.dataset_type == "cga_wiki":
        train_dataset = CGADataset(
            dataset_path=os.path.join(args.dataset_dir, args.train_filename),
            is_training=True,
            is_generation=False,
            tokenizer=tokenizer,
            context_length=args.max_context_length,
            predict_label=args.predict_label,
            use_first_n_turns=args.use_first_n_turns,
            use_social_orientation=args.use_social_orientation,
            do_not_count_section_headers=args.do_not_count_section_headers
        )

        val_dataset = CGADataset(
            dataset_path=os.path.join(args.dataset_dir, args.val_filename),
            is_training=True,
            is_generation=False,
            tokenizer=tokenizer,
            context_length=args.max_context_length,
            predict_label=args.predict_label,
            use_first_n_turns=args.use_first_n_turns,
            use_social_orientation=args.use_social_orientation,
            do_not_count_section_headers=args.do_not_count_section_headers
        )
    else:
        raise NotImplementedError(f"Dataset type {args.dataset_type} not implemented")

    # Initialize the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.gradient_accumulation_steps // world_size,
        shuffle=False,
        # Training LLMs will be slow anyway, so we don't need to use multiple workers
        num_workers=1,
        sampler=DistributedSampler(train_dataset, shuffle=True, seed=args.seed, drop_last=True),
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // args.gradient_accumulation_steps // world_size,
        shuffle=False,
        num_workers=1,
        sampler=DistributedSampler(val_dataset, shuffle=False, seed=args.seed, drop_last=True),
    )
    print(f"Finished loading datasets, training samples: {len(train_dataset)}, validation samples: {len(val_dataset)} on rank {rank}")

    # Initialize the model and optimizer
    # Figure out data type
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32

    device = torch.device(rank)
    if "llama" in args.model_type:
        model = LlamaForCausalLM.from_pretrained(args.model_type, device_map=device, torch_dtype=torch_dtype, use_flash_attention_2=args.flash_attn)
    elif "mistral" in args.model_type or "zephyr" in args.model_type:
        model = MistralForCausalLM.from_pretrained(args.model_type, device_map=device, torch_dtype=torch_dtype, use_flash_attention_2=args.flash_attn)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.load_lora:
        model = PeftModel.from_pretrained(model, args.load_lora, is_trainable=True)
        print(f"Finished loading Lora model from {args.load_lora} on rank {rank}")
        model.print_trainable_parameters()
    elif args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )
        model = get_peft_model(model, lora_config)
        print(f"Finished initializing Lora model on rank {rank}")
        model.print_trainable_parameters()

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    print(f"Finished loading model and lora on rank {rank}")

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
    print(f"Initialized optimizer and LR scheduler on rank {rank}")

    best_eval_loss = 1e9

    batch_loss = 0.0
    step_count = 0
    batch_count = 0
    last_batch_time = time.time()

    for epoch in range(args.num_epochs):
        # Set the epoch for the sampler
        train_dataloader.sampler.set_epoch(epoch)

        model.train()
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

        for step_count, batch in enumerate(train_dataloader):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)
                loss = outputs.loss
                loss /= args.gradient_accumulation_steps

            loss.backward()

            reduced_loss = loss.detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
            batch_loss += float(reduced_loss.item())

            if (step_count + 1) % args.gradient_accumulation_steps == 0:
                # Perform gradient clipping and optimizer step
                clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()
                batch_count += 1

                if rank == 0:
                    print(f"Epoch {epoch}, Batch {batch_count}, Loss {batch_loss}, samples/sec {args.batch_size / (time.time() - last_batch_time)}")
                    log_writer.add_scalar("train_loss", batch_loss, batch_count)

                batch_loss = 0.0
                last_batch_time = time.time()

        model.eval()

        dist.barrier()
        if rank == 0:
            model.module.save_pretrained(os.path.join(args.saved_model_path, args.model_name + "_latest"))
            print(f"Saved latest model, batch {batch_count}")

        eval_loss = 0
        eval_start_time = time.time()
        with torch.no_grad():
            for batch in valid_dataloader:
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                    inputs = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    mask = batch["attention_mask"].to(device)

                    outputs = model(input_ids=inputs, attention_mask=mask, labels=labels)

                reduced_loss = outputs.loss.detach()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
                eval_loss += float(reduced_loss.item())

        dist.barrier()
        if rank == 0:
            eval_loss /= len(valid_dataloader)
            log_writer.add_scalar("eval_loss", eval_loss, batch_count)
            print(
                f"Batch count {batch_count}, Eval loss {eval_loss}, "
                + f"samples/sec {len(valid_dataloader) * args.gradient_accumulation_steps * world_size / (time.time() - eval_start_time)}"
            )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model.module.save_pretrained(os.path.join(args.saved_model_path, args.model_name + "_best"))
                print(f"Saved best model, batch {batch_count}")

    if rank == 0:
        log_writer.close()

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune models on natural instructions dataset.")
    parser.add_argument("--config", type=str, default="./llm/config_mistral_full_dataset_test.json", help="Path to config file")
    args = parser.parse_args()

    # Load the config
    with open(args.config, "r") as f:
        args = DotMap(json.load(f))

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
