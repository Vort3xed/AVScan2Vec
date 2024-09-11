import os
import sys
import time
import torch
import pickle
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
import re

sys.path.insert(0, "/home/agneya/AVScan2Vec/avscan2vec/")
from globalvars import *
from utils import pretrain_collate_fn
from dataset import PretrainDataset
from avscan2vec_model import PositionalEmbedding, PretrainEncoder, PretrainLoss

# REMOVE EPOCH SETTING
def train_network(model, optimizer, scheduler, train_loader, epochs, checkpoint_file):
    """Pre-trains AVScan2Vec."""

    # Track model results
    start_time = time.time()
    total_batches = 0

    # Iterate over each epoch
    for epoch in range(epochs):
        # train_loader.sampler.set_epoch(epoch)
        model = model.train()
        print_token_loss = 0.0
        print_label_loss = 0.0
        batches = 0

        # Get epoch start time
        epoch_start_time = time.time()

        # Iterate over each batch
        for X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av, _, _ in train_loader:
            X_scan = X_scan.to("cuda", non_blocking=True)
            X_av = X_av.to("cuda", non_blocking=True)
            Y_scan = Y_scan.to("cuda", non_blocking=True)
            Y_idxs = Y_idxs.to("cuda", non_blocking=True)
            Y_label = Y_label.to("cuda", non_blocking=True)
            Y_av = Y_av.to("cuda", non_blocking=True)

            # Train model on batch
            optimizer.zero_grad()
            token_loss, label_loss, _, _ = model(X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av)

            # print(batches)
            # if batches > 3:
            #     print("kill pretrain on line 54")
            #     exit(0)

            # Get batch size
            B_token = Y_scan.shape[0]
            B_label = Y_label.shape[0]

            # Backprop and update weights
            train_loss = token_loss + label_loss
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            # Update loss totals
            batches += 1
            total_batches += 1
            with torch.no_grad():
                batch_token_loss = token_loss.item() * B_token
                batch_label_loss = label_loss.item() * B_label
            print_token_loss += batch_token_loss
            print_label_loss += batch_label_loss

            # Print training info every 100 batches
            sys.stdout.flush()
            if batches % 100 == 0:
                fmt_str = "Batches: {}  Total loss: {}\tToken loss: {}\tLabel loss: {}\t Time: {}"
                print_loss = print_token_loss + print_label_loss
                print(fmt_str.format(batches, print_loss, print_token_loss, print_label_loss, time.time()))
                sys.stdout.flush()
                print_token_loss = 0.0
                print_label_loss = 0.0

            # Checkpoint every 2,500 batches
            if batches % 2500 == 0:

                # Save partial model
                if rank == 0:
                    checkpoint_file_part = checkpoint_file + "part"
                    torch.save({
                        # "model_state_dict": model.module.state_dict(),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, checkpoint_file_part)
                    print("Saved model to {}".format(checkpoint_file_part))

        # Save model statistics at end of epoch
        torch.save({
            # "model_state_dict": model.module.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, checkpoint_file)
        print("Saved model to {}".format(checkpoint_file))

    return

# NEVER CALLING MODEL PARALLELIZATION

# def model_parallel(rank, cmd_args, pretrain_model, train_dataset):
#     """Enable AVScan2Vec to be pre-trained under Distributed Data Parallel."""

#     # Set up DDP environment
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     world_size = len(cmd_args.devices)
#     torch.cuda.set_device(rank)
#     #dist.init_process_group("nccl", rank=rank, world_size=world_size)

#     # Get distributed sampler for train dataset
#     train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
#                                        rank=rank, shuffle=True, drop_last=True)

#     # Get train loader
#     train_loader = DataLoader(train_dataset, batch_size=cmd_args.batch_size,
#                               shuffle=False, pin_memory=True,
#                               num_workers=cmd_args.num_workers,
#                               collate_fn=pretrain_collate_fn,
#                               sampler=train_sampler)

#     # Load model from checkpoint, if checkpoint file exists
#     save_info = None
#     if os.path.isfile(cmd_args.checkpoint_file):
#         save_info = torch.load(cmd_args.checkpoint_file, map_location="cpu")
#         state_dict = OrderedDict()
#         for k, v in save_info["model_state_dict"].items():
#             new_k = re.sub(r"module.", "", k)
#             state_dict[new_k] = v
#         pretrain_model.load_state_dict(state_dict)

#     # Move model to GPU
#     pretrain_model = pretrain_model.to(rank)
#     pretrain_model = DDP(pretrain_model, delay_allreduce=True)

#     # Define optimizer and scheduler
#     optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=0.00025)
#     num_schedule = cmd_args.batch_size * world_size
#     scheduler = CosineAnnealingWarmRestarts(optimizer, len(train_dataset) // num_schedule + 1)

#     # Load optimizer and scheduler if loading from checkpoint
#     if save_info is not None:
#         optimizer.load_state_dict(save_info["optimizer_state_dict"])
#         scheduler.load_state_dict(save_info["scheduler_state_dict"])

#     # Train model
#     train_args = {
#         "model": pretrain_model,
#         "optimizer": optimizer,
#         "scheduler": scheduler,
#         "train_loader": train_loader,
#         "epochs": cmd_args.num_epochs,
#         "checkpoint_file": cmd_args.checkpoint_file,
#         "rank": rank,
#         "world_size": world_size
#     }
#     train_network(**train_args)

#     # Clean up from DDP
#     dist.destroy_process_group()
#     return


if __name__ == "__main__":

    # Parse commnand line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("--temporal-split", default=False, action="store_true",
                        help="Split dataset by date, rather than randomly")
    parser.add_argument("--checkpoint-file", default="checkpoint_pretrain.sav",
                        help="Path to the checkpoint file")
    parser.add_argument("--batch-size", default=100, type=int,
                        help="Batch size")
    parser.add_argument("--num-epochs", default=5, type=int,
                        help="Number of epochs")
    parser.add_argument("--devices", default=["cuda:0", "cuda:1"],
                        help="Devices to use")
    parser.add_argument("--num-workers", default=4, type=int,
                        help="Number of subprocesses per DataLoader")
    parser.add_argument("-L", default=35, type=int,
                        help="The maximum number of tokens in an AV label")
    parser.add_argument("-D", default=768, type=int,
                        help="AVScan2Vec vector dimension")
    parser.add_argument("-H", default=768, type=int,
                        help="Hidden layer dimension")
    parser.add_argument("--tok-layers", default=4, type=int,
                        help="Number of layers in the token encoder")
    args = parser.parse_args()

    # Commandline arguments
    print("Pre-training AVScan2Vec with args: {}".format(args))
    sys.stdout.flush()

    # Set Torch Benchark mode to True
    torch.backends.cudnn.benchmark = True

    # Initialize dataset
    dataset = PretrainDataset(args.data_dir, args.L)
    # print(dataset[0])
    # dataset[0]
    # exit(0)
    # print(dataset[0])
    # dataset[0]
    # exit(0)

    # Get sizes of train / test set
    n_train = int(len(dataset) * 0.9)
    n_train = n_train // args.batch_size * args.batch_size
    print("Size of training set: {}".format(n_train))

    # Allow temporal split of data
    ids = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(ids)
    if args.temporal_split:
        with open(os.path.join(args.data_dir, "id_dates.pkl"), "rb") as f:
            id_dates = pickle.load(f)
        ids = [idx for idx, _ in id_dates]

    # Get subset for train dataset
    train_dataset = Subset(dataset, ids[:n_train])

    # Model arguments
    A = dataset.num_avs
    n_chars = len(dataset.alphabet)
    max_chars = dataset.max_chars
    PAD_idx = dataset.alphabet_rev[PAD]
    # NO_AV_idx = dataset.av_vocab_rev[NO_AV]
    NO_AV_idx = 0
    print("NO_AV_idx: ", NO_AV_idx)
    
    # n_chars is the vocab size
    token_embd = PositionalEmbedding(A, args.L, args.D, n_chars, max_chars,
                                     PAD_idx)
    encoder = PretrainEncoder(A, args.L, args.D, args.H, args.tok_layers,
                              PAD_idx, NO_AV_idx, token_embd)
    pretrain_model = PretrainLoss(A, args.L, args.D, args.H, args.tok_layers,
                                  encoder, dataset)

    """
    # Run model under DDP
    mp.spawn(
        model_parallel,
        args=(args, pretrain_model, train_dataset),
        nprocs=len(args.devices)
    )
    """

    print("done")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True,
                              num_workers=args.num_workers,
                              collate_fn=pretrain_collate_fn)

    # Load model from checkpoint, if checkpoint file exists
    save_info = None
    if os.path.isfile(args.checkpoint_file):
        print("Loading model from {}".format(args.checkpoint_file))
        sys.stdout.flush()
        save_info = torch.load(args.checkpoint_file, map_location="cuda")
        

        # state_dict = OrderedDict()
        # for k, v in save_info["model_state_dict"].items():
        #     new_k = re.sub(r"module.", "", k)
        #     state_dict[new_k] = v
        
        pretrain_model.load_state_dict(state_dict)

    # Move model to GPU
    pretrain_model = pretrain_model.to("cuda")

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=0.00025)
    num_schedule = args.batch_size
    scheduler = CosineAnnealingWarmRestarts(optimizer, len(train_dataset) // num_schedule + 1)

    # Load optimizer and scheduler if loading from checkpoint
    if save_info is not None:
        optimizer.load_state_dict(save_info["optimizer_state_dict"])
        scheduler.load_state_dict(save_info["scheduler_state_dict"])

    # Train model
    train_args = {
        "model": pretrain_model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "epochs": args.num_epochs,
        "checkpoint_file": args.checkpoint_file,
    }
    train_network(**train_args)
    print("Done!")
