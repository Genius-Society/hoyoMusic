import os
import json
import time
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from modelscope.msdatasets import MsDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Config, get_scheduler
from utils import Patchilizer, TunesFormer, PatchilizedData, download, DEVICE
from config import *


def init():
    random.seed(42)
    batch_size = max(1, torch.cuda.device_count())
    patchilizer = Patchilizer()
    patch_config = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS,
        max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH,
        vocab_size=1,
    )
    char_config = GPT2Config(
        num_hidden_layers=CHAR_NUM_LAYERS,
        max_length=PATCH_SIZE,
        max_position_embeddings=PATCH_SIZE,
        vocab_size=128,
    )
    model = TunesFormer(patch_config, char_config, share_weights=SHARE_WEIGHTS)

    # print parameter number
    print(
        f"Parameter Number: {str(sum(p.numel() for p in model.parameters() if p.requires_grad))}"
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    scaler = GradScaler()
    is_autocast = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    return batch_size, patchilizer, model, scaler, is_autocast, optimizer


def collate_batch(batch):
    input_patches = []
    for input_patch in batch:
        input_patches.append(input_patch.reshape(-1))

    input_patches = torch.nn.utils.rnn.pad_sequence(
        input_patches,
        batch_first=True,
        padding_value=0,
    )

    return input_patches.to(DEVICE)


def split_data(data, eval_ratio=0.1):
    random.shuffle(data)
    split_idx = int(len(data) * eval_ratio)
    eval_set = data[:split_idx]
    train_set = data[split_idx:]
    return train_set, eval_set


def process_one_batch(batch, model):  # call model with a batch of input
    input_patches = batch
    loss = model(input_patches).loss
    return loss.mean()


def train_epoch(
    model,
    optimizer,
    lr_scheduler,
    is_autocast,
    scaler,
    train_set,
):  # do one epoch for training
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()

    for batch in tqdm_train_set:
        try:
            if is_autocast:
                with autocast():
                    loss = process_one_batch(batch, model)

                if loss == None or torch.isnan(loss).item():
                    continue

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss = process_one_batch(batch, model)
                if loss == None or torch.isnan(loss).item():
                    continue

                loss.backward()
                optimizer.step()

        except RuntimeError as exception:
            if "memory" in str(exception):
                print(str(exception))
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

                continue

            else:
                raise exception

        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        tqdm_train_set.set_postfix({"train_loss": total_train_loss / iter_idx})
        iter_idx += 1

    return total_train_loss / (iter_idx - 1)


def eval_epoch(model, eval_set):  # do one epoch for eval
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm_eval_set:
        with torch.no_grad():
            loss = process_one_batch(batch, model)
            if loss == None or torch.isnan(loss).item():
                continue

            total_eval_loss += loss.item()

        tqdm_eval_set.set_postfix({"eval_loss": total_eval_loss / iter_idx})
        iter_idx += 1

    return total_eval_loss / (iter_idx - 1)


def train(weights_url=WEIGHT_URL):
    # load data
    dataset = MsDataset.load(
        f"Genius-Society/{DATASET}",
        subset_name="default",
        cache_dir="./__pycache__",
        trust_remote_code=True,
    )
    trainset, evalset = [], []
    classes = dataset["test"].features["label"].names
    for song in dataset["train"]:
        label = classes[song["label"]]
        if label == "Teyvat":
            label = ""
        else:
            label = f"A:{label}\n"

        trainset.append(
            {
                "control code": label + song["prompt"],
                "abc notation": song["data"],
            }
        )

    for song in dataset["test"]:
        label = classes[song["label"]]
        if label == "Teyvat":
            label = ""
        else:
            label = f"A:{label}\n"

        evalset.append(
            {
                "control code": label + song["prompt"],
                "abc notation": song["data"],
            }
        )

    batch_size, patchilizer, model, scaler, is_autocast, optimizer = init()

    trainset = DataLoader(
        PatchilizedData(trainset, patchilizer),
        batch_size=batch_size,
        collate_fn=collate_batch,
        shuffle=True,
    )

    evalset = DataLoader(
        PatchilizedData(evalset, patchilizer),
        batch_size=batch_size,
        collate_fn=collate_batch,
        shuffle=True,
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=NUM_EPOCHS * len(trainset) / 10,
        num_training_steps=NUM_EPOCHS * len(trainset),
    )

    if LOAD_FROM_CHECKPOINT:
        if not os.path.exists(WEIGHT_PATH):
            download(url=weights_url)

        checkpoint = torch.load(WEIGHT_PATH, weights_only=False)
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint["model"], strict=False)

        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_sched"])
        pre_epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        min_eval_loss = checkpoint["min_eval_loss"]
        print(f"Successfully Loaded Checkpoint from Epoch {pre_epoch}")

    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = 100

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for epoch in range(1, NUM_EPOCHS + 1 - pre_epoch):
        epoch += pre_epoch
        print(f"{'-' * 21}Epoch {str(epoch)}{'-' * 21}")
        train_loss = train_epoch(
            model,
            optimizer,
            lr_scheduler,
            is_autocast,
            scaler,
            trainset,
        )
        eval_loss = eval_epoch(model, evalset)
        with open(LOG_PATH, "a", encoding="utf-8") as jsonl_file:
            json_str = json.dumps(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "eval_loss": float(eval_loss),
                    "time": f"{time.asctime(time.localtime(time.time()))}",
                }
            )
            jsonl_file.write(json_str + "\n")

        if eval_loss < min_eval_loss:
            best_epoch = epoch
            min_eval_loss = eval_loss
            checkpoint = {
                "optimizer": optimizer.state_dict(),
                "lr_sched": lr_scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "min_eval_loss": min_eval_loss,
                "time_stamp": time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime()),
            }

            if torch.cuda.device_count() > 1:
                checkpoint["model"] = model.module.state_dict()
            else:
                checkpoint["model"] = model.state_dict()

            torch.save(checkpoint, WEIGHT_PATH)
            break

    print(f"Best Eval Epoch : {best_epoch}\nMin Eval Loss : {min_eval_loss}")


if __name__ == "__main__":
    train()
