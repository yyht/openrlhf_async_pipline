import os

from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from openrlhf.utils.logging_utils import init_logger
import json
import os, sys, uuid

logger = init_logger(__name__)

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    # train_dataset = interleave_datasets(
    #     train_data_list,
    #     probabilities=probabilities,
    #     seed=seed,
    #     stopping_strategy=stopping_strategy,
    # )

    # merge and filter dataset
    def my_map(example):
        if 'level' in example:
            level = example['level']
        else:
            level = 6
        try:
            level = int(level)
        except:
            level = 6
        if 'label' in example:
            label = json.loads(example['label'])
            label['uuid'] = str(uuid.uuid4())
            label['gold_ans'] = example['gold_ans']
            label = json.dumps(label)
        else:
            label = {
                'uuid': str(uuid.uuid4()),
                'gold_ans': example['gold_ans']
            }
            label.update(example)
            label = json.dumps(label)
        new_example = {
            'query': example['query'],
            'prompt': example['prompt'],
            'gold_ans': example['gold_ans'],
            'level': level,
            'partial_rollout': example.get('partial_rollout', ''),
            'label': label
        }
        return new_example

    new_data_list = []
    for data in train_data_list:
        data = data.map(lambda example: my_map(example),
                            num_proc=8, remove_columns = data.column_names)
        new_data_list.append(data)

    train_dataset = concatenate_datasets(
        new_data_list
    )

    logger.info({
        'INFO': "BEFOREFILTERING",
        'SIZE': len(train_dataset)
    })

    def filter_fn(example):
        if example.get('partial_rollout', ''):
            return example['level'] >= 5
        else:
            return example['level'] >= 4 and example['level'] <= 9

    if strategy.args.use_data_filter:
        train_dataset = train_dataset.filter(lambda example: filter_fn(example), num_proc=8)

        logger.info({
            'INFO': "AFTERFILTERING",
            'SIZE': len(train_dataset)
        })

    train_dataset = train_dataset.filter(lambda example: len(example['gold_ans'].strip()) >=1, num_proc=32)

    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")
