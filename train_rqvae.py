from sentence_transformers import SentenceTransformer
from src.rqvae.rqvae import RQVAE
import json
import torch
from torch import nn
from torch.optim import adagrad
from torch.utils.data import TensorDataset, DataLoader

"""
Create item embeddings using SentenceT5, then train rqvae on the dataset of item embeddings.
"""


"""
Trained for 20k epochs, achieves >= 80% codebook usage. Adagrad Optimizer with lr=0.4.
Batch size = 1024
"""

########   HYPERPARAMETERS  ##############
EPOCHS=20000
BATCH_SIZE = 1024


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
file_path = "data/beauty/item_feat.json"
item_sentences = load_json(file_path)


model = SentenceTransformer('sentence-transformers/sentence-t5-base')

sentence_list = list(item_sentences.values())

# Modify the list of sentences using the modify_sentences function
modified_sentence_list = model.encode(sentence_list)

# Map the modified sentences back to their original ids in the dictionary
item_embeddings = {}
for id, sentence in item_sentences.items():
    item_embeddings[id] = modified_sentence_list[sentence_list.index(sentence)]






########

dataset = TensorDataset(torch.tensor(item_embeddings.values()))
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initalizing RQVAE
vae = RQVAE(
    num_codebooks=3,
    codebook_size=256,
    in_channels=768,
    latent_dim=32,
    hidden_channels=[512, 256, 128]
)

optimizer = adagrad(vae.parameters(), lr=0.4)

for epoch in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_rqvae_loss = 0
    for data in train_loader:
        data = data[0]
        optimizer.zero_grad()
        out, codes, quant_loss = vae(data)
        loss, recon_loss, rqvae_loss = vae.compute_loss(out, data, quant_loss)

        loss.backward()
        optimizer.step()
        epoch_loss += loss
        epoch_recon_loss += recon_loss
        epoch_rqvae_loss += rqvae_loss
    
    

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss: {epoch_loss} Recon: {epoch_recon_loss}, Rqvae: {epoch_rqvae_loss}")

    
    





