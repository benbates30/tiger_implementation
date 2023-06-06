import torch
from torch import Tensor
from torch.utils.data import Dataset
import os
import json

from sentence_transformers import SentenceTransformer


"""
TODO: Use torch.nn.embedding instead to store embeddings
"""

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def compute_embeddings(sentences):
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    sentence_list = list(sentences.values())
    embeddings = model.encode(sentence_list)
    return embeddings

class EmbeddingDataset(Dataset):

    def __init__(self, item_feat_dir, item_context_file, item_embed) -> None:
        self.item_feat_dir = item_feat_dir
        self.item_context_file = item_context_file
        self.item_embed = item_embed

        self.item_context = load_json(os.path.join(self.item_feat_dir, self.item_context_file))

        try:
            self.item_embeddings = torch.load(os.path.join(self.item_feat_dir, self.item_embed))
            print("Precomputed embeddings found. Loading from file...")
        except:
            print("Precomputed embeddings not found. Computing embeddings...")
            self.item_embeddings = compute_embeddings(self.item_context)
            torch.save(self.item_embeddings, os.path.join(self.item_feat_dir, self.item_embed))

        self.id_to_index = {item_id: index for index, item_id in enumerate(self.item_context.keys())}
        self.index_to_id = {index: item_id for index, item_id in enumerate(self.item_context.keys())}

        "Done!"
    def __len__(self):
        return len(self.item_embeddings)

    def __getitem__(self, idx):
        return self.item_embeddings[idx]

    def get_embedding_by_id(self, item_id):
        index = self.id_to_index[item_id]
        return self.item_embeddings[index]
    


if __name__ == "__main__":
    from torch.utils.data import DataLoader


    item_feat_dir = "data/beauty"
    item_context_file = "item_feat.json"
    item_embed = "item_embeddings.pt"

    # torch.load(os.path.join(item_feat_dir, item_embed))

    # item_context = load_json(os.path.join(item_feat_dir, item_context_file))
    # print(item_context)
    dataset = EmbeddingDataset(item_feat_dir, item_context_file, item_embed)
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True)


