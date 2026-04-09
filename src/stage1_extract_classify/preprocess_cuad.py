import json
import logging
import os
from pathlib import Path
import random

from datasets import DatasetDict, Dataset, load_from_disk
from transformers import AutoTokenizer
from datasets import concatenate_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/deberta-base"
CACHE_PATH = "./tokenized_cuad"


# =========================
# CACHE VALIDATION
# =========================
def cache_is_valid(path: str) -> bool:
    try:
        load_from_disk(path)
        logger.info(f"✅ Valid cache found at: {path}")
        return True
    except Exception:
        return False

# =========================
# LOAD DATASET
# =========================
def load_cuad_dataset() -> DatasetDict:
    candidates = [
        os.environ.get("CUAD_JSON", ""),
        "./CUAD_v1.json",
    ]

    json_path = next((p for p in candidates if p and Path(p).exists()), None)

    if json_path is None:
        raise FileNotFoundError("CUAD_v1.json not found")

    logger.info(f"Loading CUAD dataset from: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = {"id": [], "question": [], "context": [], "answers": []}

    for doc in raw["data"]:
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                rows["id"].append(qa["id"])
                rows["question"].append(qa["question"].strip())
                rows["context"].append(context)

                if qa.get("is_impossible") or not qa["answers"]:
                    rows["answers"].append({"text": [], "answer_start": []})
                else:
                    rows["answers"].append({
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    })

    flat = Dataset.from_dict(rows)
    logger.info(f"Loaded {len(flat)} total QA pairs")

    # =========================
    # SPLIT raw unbalanced data
    # =========================
    split = flat.train_test_split(test_size=0.10, seed=42)
    raw_train_val = split["train"]
    test = split["test"]  # natural distribution — never touched by balancing

    VAL_RATIO = 0.10 / 0.90
    split2 = raw_train_val.train_test_split(test_size=VAL_RATIO, seed=42)
    raw_train = split2["train"]
    val = split2["test"]  # natural distribution — never touched by balancing

    logger.info(f"Pre-balance sizes → train: {len(raw_train)}, val: {len(val)}, test: {len(test)}")

    # =========================
    # BALANCE TRAIN ONLY, with 1:1 ratio
    # =========================
    logger.info("Balancing training set...")

    positives = raw_train.filter(lambda x: len(x["answers"]["answer_start"]) > 0)
    negatives = raw_train.filter(lambda x: len(x["answers"]["answer_start"]) == 0)

    logger.info(f"Train → Positives: {len(positives)}, Negatives: {len(negatives)}")

    # Sanity check — no rows should be lost or double-counted
    assert len(positives) + len(negatives) == len(raw_train), (
        f"Balancing filter leaked rows! "
        f"{len(positives)} + {len(negatives)} != {len(raw_train)}"
    )

    # 1:1 ratio — better for CUAD's sparse clause distribution
    negatives = negatives.shuffle(seed=42).select(
        range(min(len(negatives), len(positives)))
    )

    train = concatenate_datasets([positives, negatives])
    train = train.shuffle(seed=42)

    pos = sum(1 for x in train if len(x["answers"]["answer_start"]) > 0)
    neg = len(train) - pos
    logger.info(f"Balanced train → Positives: {pos}, Negatives: {neg}, Ratio: 1:{neg/pos:.2f}")

    dataset = DatasetDict({
        "train": train,
        "validation": val,
        "test": test,
    })

    logger.info(f"Final sizes → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return dataset

# =========================
# TOKENIZATION
# =========================
def preprocess_for_qa(examples: dict, tokenizer: AutoTokenizer) -> dict:
    tokenized = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    answers = examples["answers"]

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):

        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        try:
            ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
            ctx_end = len(sequence_ids) - 1 - next(
                j for j, s in enumerate(reversed(sequence_ids)) if s == 1
            )
        except StopIteration:
            start_positions.append(0)
            end_positions.append(0)
            continue

        if not answer["answer_start"]:
            start_positions.append(0)
            end_positions.append(0)
            continue

        char_start = answer["answer_start"][0]
        char_end = char_start + len(answer["text"][0])

        if offsets[ctx_start][0] > char_end or offsets[ctx_end][1] < char_start:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # FIX: Check current token's offset, not next/prev (was off-by-one)
        # Find start token: walk forward until we pass char_start, then step back
        token_start = ctx_start
        while token_start <= ctx_end and offsets[token_start][0] <= char_start:
            token_start += 1
        token_start -= 1  # last token whose start offset <= char_start

        # Find end token: walk backward until we pass char_end, then step forward
        token_end = ctx_end
        while token_end >= ctx_start and offsets[token_end][1] >= char_end:
            token_end -= 1
        token_end += 1  # first token whose end offset >= char_end

        start_positions.append(token_start)
        end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized

# =========================
# MAIN
# =========================
def main():

    if cache_is_valid(CACHE_PATH):
        logger.info("Skipping preprocessing — delete cache to rerun.")
        return

    dataset = load_cuad_dataset()

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Tokenizing dataset...")

    tokenized = dataset.map(
        preprocess_for_qa,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=None,
        remove_columns=dataset["train"].column_names,
    )

    logger.info(f"Saving dataset to: {CACHE_PATH}")
    tokenized.save_to_disk(CACHE_PATH)

    tokenizer.save_pretrained(os.path.join(CACHE_PATH, "tokenizer"))

    logger.info("✅ DONE!")


if __name__ == "__main__":
    main()
