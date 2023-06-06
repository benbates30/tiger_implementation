import torch
from torch import Tensor
from torch.utils.data import Dataset
import os


class SeqDataset(Dataset):
    """
    Seq Dataset is used to store and the train, val, or test dataset for training the t5-based
    transformer model. It also allows the files to be stored and reloaded to avoided re-doing masking.
    """

    files = ["inputs.pt", "targets.pt", "padding_mask.pt", "mask.pt"]
    def __init__(self, data_dir, split, seq, padding_mask) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.dir = os.path.join(data_dir, split)
        self.seq = seq
        good = True
        for file in self.files:
            if not os.path.exists(os.path.join(self.dir, file)):
                good = False
                break
        if all(os.path.exists(os.path.join(self.dir, file)) for file in self.files):
            print(f"Dataset found in {self.dir} !")
            self.inputs = torch.load(os.path.join(self.dir, "inputs.pt"))
            self.targets = torch.load(os.path.join(self.dir, "targets.pt"))
            self.padding_mask = torch.load(os.path.join(self.dir, "padding_mask.pt"))
            self.mask = torch.load(os.path.join(self.dir, "mask.pt"))
        else:
            print("Creating dataset...")
            self.padding_mask = padding_mask
            if split == "test":
                self.targets = seq
                self.inputs = seq
                self.mask = torch.ones_like(self.inputs)
                self.mask[-4:] = 0
                
            elif split == "val":
                self.targets = seq[:-4]
                self.inputs = seq[:-4]
                self.mask = torch.ones_like(self.inputs)
                self.mask[-4:] = 0

            elif split == "train":
                self.targets = seq[:-8]
                self.inputs = seq[:-8]


                self.mask = torch.ones_like(self.inputs)
                mask_index = torch.randint(0, len(self.mask) - 3, (1,)).item()
                mask_index = mask_index - (mask_index % 4)
                self.mask[mask_index:mask_index + 4] = 0

            torch.save(self.inputs, os.path.join(self.dir, "inputs.pt"))
            torch.save(self.targets, os.path.join(self.dir, "targets.pt"))
            torch.save(self.mask, os.path.join(self.dir, "mask.pt"))
            torch.save(self.padding_mask, os.path.join(self.dir, "padding_mask.pt"))
            print("Dataset created and stored!")       
    
    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.targets[index],
            self.padding_mask[index],
            self.mask[index]
        )
    
    def __len__(self):
        return len(self.inputs)




