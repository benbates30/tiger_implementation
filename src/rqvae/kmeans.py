from sklearn.cluster import KMeans
import numpy as np
import torch

def kmeans(data, n_clusters, max_iter):
    km = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init="auto")
    np_data = data.numpy()
    km.fit(np_data)
    return torch.tensor(km.cluster_centers_)

if __name__ == "__main__":
    input = torch.tensor([[0.04, 0.67, 0.56], [0.45, 0.69, -0.34], [-0.3, -0.5, .97], [0.43, -0.4, 0.99]])
    centers = kmeans(input, 2, max_iter=300)

    print(centers)
