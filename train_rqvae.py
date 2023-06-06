from src.rqvae.rqvae import RQVAE
import json
import torch
from torch import nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from embedding_dataset import EmbeddingDataset

"""
Trained for 20k epochs, achieves >= 80% codebook usage. Adagrad Optimizer with lr=0.4.
Batch size = 1024
"""

########   HYPERPARAMETERS  ##############
EPOCHS=20000
BATCH_SIZE = 1024
PRINT_LOSS_INTERVAL = 100
model_path = "models/rqvae.pt"
item_feat_dir = "data/beauty"
item_context_file = "item_feat.json"
item_embed = "item_embeddings.pt"


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

dataset = EmbeddingDataset(item_feat_dir, item_context_file, item_embed)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initalizing RQVAE
vae = RQVAE(
    num_codebooks=3,
    codebook_size=256,
    in_channels=768,
    latent_dim=32,
    hidden_channels=[512, 256, 128]
)

optimizer = Adagrad(vae.parameters(), lr=0.4)

print("Training...")
for epoch in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_rqvae_loss = 0
    for data in train_loader:
        data = data
        optimizer.zero_grad()
        out, codes, quant_loss = vae(data)
        loss, recon_loss, rqvae_loss = vae.compute_loss(out, data, quant_loss)

        loss.backward()
        optimizer.step()
        epoch_loss += loss
        epoch_recon_loss += recon_loss
        epoch_rqvae_loss += rqvae_loss
    
    

    if epoch % PRINT_LOSS_INTERVAL == 0:
        print(f"Epoch {epoch}: Loss: {epoch_loss} Recon: {epoch_recon_loss}, Rqvae: {epoch_rqvae_loss}")



torch.save(vae.state_dict(), model_path)
    





