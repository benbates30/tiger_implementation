import torch
from torch.utils.data import DataLoader
from embedding_dataset import EmbeddingDataset
from seq_dataset import SeqDataset
from src.rqvae.rqvae import RQVAE
import numpy as np


"""
Generate a train/val/test splits of sequential tokens to be fed to the transformer model.

TODO: Use config
"""



batch_size = 15000
model_path = "models/rqvae.pt"
item_feat_dir = "data/beauty"
item_context_file = "item_feat.json"
item_embed = "item_embeddings.pt"
item_sequences = "data/beauty/sequential_data.txt"
id_seq_file = "data/beauty/semantic_id_sequences.npy"






def generate_id_dict(item_feat_dir, item_context_file, item_embed, model_path):

    """
    Given the path to a trained rqvae model and a dictionary of items + contexts, 
    genertes the semantic IDs for each item, returning a dictionary mapping from item ID to
    semantic ID.
    
    """
    # initalizing RQVAE
    model = RQVAE(
        num_codebooks=3,
        codebook_size=256,
        in_channels=768,
        latent_dim=32,
        hidden_channels=[512, 256, 128]
    )
    model.load_state_dict(torch.load(model_path))

    dataset = EmbeddingDataset(item_feat_dir, item_context_file, item_embed)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    semantic_codes = {} # dict to map from each list[id, id, id] to how many times it has occured
    item_2_codes = {} # dict to map each item to its code

    for batch in data_loader:
        codes_batch = model.get_codes(batch)

        for index, codes in enumerate(codes_batch):
            item_id = dataset.index_to_id[index]
            counter = semantic_codes.get(tuple(codes), -1) + 1
            semantic_codes[tuple(codes)] = counter
            item_2_codes[item_id] = list(codes) + [torch.tensor(counter)]

    return item_2_codes
            

def generate_semantic_id_sequence(item_sequences, item_2_codes):
    """
    Takes as inpuut a text file location that contains a string of item sequences on each line,
    as well as a dictionary mapping item ids to code ids. The output is
    """

    with open(item_sequences, 'r') as file:
        item_sequences = [line.strip().split()[1:] for line in file]
        
    semantic_id_seq = []

    for seq in item_sequences:
        codes = []
        for item_id in seq:
            codes.extend(item_2_codes[item_id])
        semantic_id_seq.append(codes)

    return semantic_id_seq

def pad_truncate_sequences(semantic_seq, seq_length=80, pad_token = 1029):
    """ pad/trunctae sequences as needed and return sequence as tensor, with
    padding mask tensor
    """
    padding_mask = []
    no_pad = [1 for _ in range(seq_length)]
    for index, seq in enumerate(semantic_seq):
        pad_length = seq_length - len(seq)
        if pad_length > 0:
            padding = [torch.tensor(pad_token) for _ in range(pad_length)]
            semantic_seq[index] = padding + seq
            padding_mask.append([0 for _ in range(pad_length)] + [1 for _ in range(seq_length-pad_length)])
        if pad_length < 0:
            # truncate
            semantic_seq[index] = seq[pad_length*-1:]
            padding_mask.append(no_pad)
    return torch.tensor(semantic_seq), torch.tensor(padding_mask)















if __name__ == "__main__":
    item_2_codes = generate_id_dict(item_feat_dir, item_context_file, item_embed, model_path)
    
    semantic_seq = generate_semantic_id_sequence(item_sequences, item_2_codes)

    semantic_seq, padding_mask = pad_truncate_sequences(semantic_seq)
    # print(len(semantic_seq))
    # print(len(semantic_seq[0]))
    # print(semantic_seq[0])
    # print(padding_mask[5])
    # # print(semantic_seq.size())

    SeqDataset("data/beauty/seq", "test", seq=semantic_seq, padding_mask=padding_mask)
    SeqDataset("data/beauty/seq", "val", seq=semantic_seq, padding_mask=padding_mask)
    SeqDataset("data/beauty/seq", "train", seq=semantic_seq, padding_mask=padding_mask)


