from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import joblib

MEM = joblib.Memory(".cache")

@MEM.cache()
def compute_embedding(X):
    return UMAP().fit_transform(X)

X = load_digits().data

labels = KMeans(n_clusters=10).fit_predict(X)
embedding = compute_embedding(X)

print(embedding)

plt.scatter(
    x = embedding[:,0],
    y = embedding[:,1],
    c = labels,
    cmap="tab10",
    s=5
)
plt.gca().axis("off")
plt.gca().set_aspect("equal", "datalim")
plt.savefig("imgs/example-digits.png")

