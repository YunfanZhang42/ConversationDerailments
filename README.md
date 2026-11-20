# ConversationDerailments
Code and dataset for ACL INLG 2025 Paper: Forecasting Conversation Derailments Through Generation

Research code for forecasting conversation derailment on CGA-Wiki and Before
Name Calling (BNC) by sampling future conversations and classifying their
outcomes. The selected source snapshot contains the shared Mistral generator,
BART/Mistral classifiers, social-orientation prompts, and experiment configs.
Datasets, annotations, generated conversations, predictions, and model weights
must be supplied or regenerated separately.

## Core execution path

Run commands from the repository root and always pass an explicit `--config`.
Some historical script defaults refer to experiments outside this selection.

1. Prepare the original train/validation/test JSONL splits under
   `data_collection/final_data/cga_wiki/` or `data_collection/final_data/bnc/`.
   `downstream_task/download_cga_dataset.py` exports CGA from ConvoKit;
   `downstream_task/process_before_name_calling.py` processes the separately
   obtained BNC sampled threads and full CMV posts. Pass its two input-folder
   arguments instead of relying on the author's local Downloads directory.
2. For social-orientation (SO) experiments, prepare annotated versions of the
   same splits. `gpt4_batched_circumplex_tags.py` formats annotation requests and
   `process_gpt_4_batched_circumplex_tags.py` merges returned annotations.
   The `social_orientation_user_prompt_pass_1/2.txt` files use the original
   eight-category scheme; the `*_cga_wiki_new.txt` and
   `bnc_social_orientation_prompt_pass_1/2.txt` files use four dimensions.
   Pass the input/output and prompt paths explicitly for each annotation pass.
   Request submission and returned API outputs are external to this snapshot.
   `match_circumplex_tags.py` is the historical CGA CSV-label import helper.
3. Fine-tune the generator with `llm/train_llm.py` and the appropriate generator
   config below. It launches DDP workers on all visible CUDA devices. For a
   fresh training run, `load_lora` must be empty; nonempty values resume a local
   adapter. Use a visible GPU count compatible with the configured batch size
   and gradient accumulation.
4. Sample continuations with `llm/infer_llm.py`. It loads the generator adapter
   specified by `generation_load_lora`, writes `all_text` records, and supports
   `--generation-dataset-override`, `--save-generation-to-override`, and
   `--generation-epochs-override`. Use two continuations per training example
   and five per validation/test example for the main experimental protocol.
   Match output filenames to the chosen classifier config. Configs ending in
   `_combined.jsonl` training filenames expect original and generated training
   records together; the same `episode_title` identifies one conversation.
5. Train and evaluate with `downstream_task/train.py`. It selects the best
   checkpoint on validation accuracy and prints test accuracy, precision,
   recall, and F1. Generation configs enable majority voting by `episode_title`.
   `--seed-override` selects the classifier seed, and
   `--intermediate-result-path` selects the prediction output. Give each run a
   separate output path. LoRA checkpoint directories use `model_name` without
   the seed, so also use distinct model names for separate seeded runs.

Original JSONL records require `episode_title`, `toxic_label`, and
`conversations` (a list of `role`/`content` turns). Annotated turns additionally
carry `social_orientation`; CGA can carry `is_section_header`. Generated records
require `episode_title`, `toxic_label`, and `all_text`. Keep the original splits
and all continuations of a conversation together. For original conversations,
baseline classifiers exclude the held-out final turn. For `all_text` records,
the loader uses the complete generated text and removes the generated verdict.

## Main classifier configurations

All filenames in this table are relative to `downstream_task/`. G denotes
generated continuations; SO denotes social-orientation annotations. These are
historical experiment settings, not a claim that every file is a final paper run.

| Dataset / model | Baseline | + G | + SO | + SO + G |
| --- | --- | --- | --- | --- |
| CGA-Wiki / BART | `bart_cga_wiki_no_generation.json` | `bart_cga_wiki_generation.json` | `bart_cga_wiki_no_generation_with_social_tags.json` | `bart_cga_wiki_generation_with_social_tags.json` |
| CGA-Wiki / Mistral | `mistral_cga_wiki_baseline_no_social_orientation.json` | `mistral_cga_wiki_generation.json` | `mistral_cga_wiki_social_orientation.json` | `mistral_cga_wiki_generation_social_orientation.json` |
| BNC / BART | `bart_bnc_baseline.json` | `bart_bnc_aug.json` | `bart_bnc_circumplex.json` | `bart_bnc_circumplex_generation.json` |
| BNC / Mistral | `mistral_bnc_aug_baseline.json` | `mistral_bnc_aug.json` | `mistral_bnc_social_orientation.json` | `mistral_bnc_generation_social_orientation.json` |

The selected two- and four-turn CGA comparisons are
`bart_cga_wiki_first_{2,4}_turns_baseline.json`,
`bart_cga_wiki_augmented_first_{2,4}_turns_no_social_orientation.json`, and
`mistral_cga_wiki_first_{2,4}_turns_{baseline,aug}.json`.
The later four-dimensional CGA alternatives are
`bart_cga_wiki_circumplex_new.json` and
`mistral_cga_wiki_new_circumplex_generation_alt.json` (combined training data).

Generator configs under `llm/` are `config_mistral_bnc.json` (without SO),
`config_mistral_bnc_circumplex.json` (with SO),
`config_mistral_cga_wiki_all_but_last_turns.json` (eight-category SO),
`config_mistral_cga_wiki_first_2_turns.json`,
`config_mistral_cga_wiki_first_4_turns.json` (without SO), and
`config_mistral_cga_wiki_new_circumplex.json` (four-dimensional SO).
BNC deliberately uses `dataset_type: "cga_wiki"` to share the same dataset class.

Example baseline, after preparing data and the environment:

```bash
mkdir -p checkpoints logs
python3 downstream_task/train.py \
  --config downstream_task/bart_bnc_baseline.json \
  --intermediate-result-path logs/bart_bnc_baseline_predictions.json
```

Example generation, after training the matching adapter:

```bash
python3 llm/infer_llm.py \
  --config llm/config_mistral_cga_wiki_all_but_last_turns.json \
  --generation-dataset-override test_circumplex_fixed.jsonl \
  --save-generation-to-override test_circumplex_fixed_aug_latest_ckpt.jsonl \
  --generation-epochs-override 5
```

The example config points to the **best** generator checkpoint despite the
historical output filename containing `latest_ckpt`. Confirm checkpoint choice
before treating regenerated files as replacements for historical outputs.

## Reproduction limits to resolve before a full run

The code and numeric settings are preserved during source selection. The repo
does not establish a complete mapping from paper table cells to final seeds,
checkpoints, annotation revisions, and combined datasets. Exact published
numbers have not been reproduced by this commit.

- `requirements.txt` is the historical environment snapshot (PyTorch
  `2.2.1+cu118`, Transformers `4.40.2`, PEFT `0.10.0`, FlashAttention `2.5.8`).
  CUDA wheels need the corresponding PyTorch package index and FlashAttention
  needs a compatible CUDA build environment. ConvoKit and tiktoken are additional
  preprocessing dependencies absent from that snapshot. Validate this
  environment in isolation before a full run; generation uses the
  `stop_strings`/`tokenizer` generation API. The current workspace has newer
  packages and cannot validate the historical environment by itself.
- `config_mistral_bnc.json` has a nonempty training `load_lora` but lacks the
  separate `generation_load_lora` key used by the current inference script.
  Set the former to empty for fresh training and the latter to the resulting
  adapter path for inference. Otherwise inference can run the base model.
  Several other configs use singular `generation_epoch`; pass the explicit
  `--generation-epochs-override` for every generation run.
- Some Mistral classifier configs load a CGA first-two-turn adapter even for
  other settings; the four-turn generator config also references the two-turn
  adapter. Verify these choices against the intended run. Do not silently
  substitute another adapter when comparing to a published result.
- Default generation input names such as `val_generation_test.jsonl` are local
  subsets. Override them with the actual evaluation split. `*_combined` files
  and the historical `*_circumplex_fixed` annotation corrections are separate
  preparation artifacts; their exact construction is not fully recorded here.
  The CGA exporter writes `*_username_2018.jsonl` but does not currently enable
  its commented-out 2018 filter or produce the classifier's plain split names.
  BNC preprocessing seeds shuffling, but uses unsorted directory iteration;
  CGA preprocessing does not seed its training shuffle. Preserve the original
  prepared splits when exact run matching is required.
- Generation's distributed sampler pads when the dataset size is not divisible
  by the number of GPUs, potentially adding repeated examples. Check per-ID
  continuation counts before majority-vote evaluation. The evaluator also
  assumes nonzero positive-label/prediction counts for precision and recall.

Unrelated podcast collection/classification, human-evaluation applications,
presidential debates, PAN attribution, Qwen/T5 comparisons, debugging scripts,
and July 2025 analysis runs are outside this selected core. Legacy files already
tracked by earlier commits remain in repository history; this commit adds no
datasets or model artifacts.
