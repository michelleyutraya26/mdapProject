from pathlib import Path
from itertools import islice
from text_processing import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data_dir = Path("data/determinations")
n_docs = None

files = sorted(data_dir.glob("*.txt"))
if n_docs is not None:
    files = files[:n_docs]


file_names = []
raw_texts = []

for p in files:
    text = p.read_text(encoding="utf-8", errors="ignore")
    cleaned = preprocess(text)          
    if cleaned.strip():
        file_names.append(p.name)
        raw_texts.append(cleaned)  # " ".join

print(f"Preprocessed {len(raw_texts)} files.")


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None, # already removed stopwords in preprocess
    ngram_range=(1, 2),      # try (1,2) later
    min_df=10,
    max_df=0.5,
    max_features=30000,
    dtype=np.float32,
    sublinear_tf=True,
    norm="l2",
)
X = vectorizer.fit_transform(raw_texts)
print("TF-IDF shape:", X.shape)

# high dimensionality, so use SVD
USE_SVD = True
if USE_SVD:
    n_components = min(100, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_clust = svd.fit_transform(X)           # dense (n_docs, n_components)
    print(f"SVD -> shape: {X_clust.shape}")
    print(svd.explained_variance_ratio_)
    print("Total explained variance:", svd.explained_variance_ratio_.sum())

else:
    X_clust = X




Ks = range(4, 16)
inertias = []
sils = []

for k in Ks:
    km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42)
    labels = km.fit_predict(X_clust)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_clust, labels))


best_k = Ks[int(np.argmax(sils))]
print("Best k by silhouette:", best_k)


plt.figure()
plt.plot(list(Ks), inertias, marker="o")
plt.xlabel("k (clusters)")
plt.ylabel("SSE (inertia)")
plt.title("Elbow Method (TF-IDF + SVD+ KMeans)")
plt.savefig("elbow_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved elbow plot to elbow_plot.png")

plt.figure()
plt.plot(list(Ks), sils, marker="o")
plt.xlabel("k (clusters)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method (TF-IDF + SVD + KMeans)")
plt.savefig("silhouette_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved silhouette plot to silhouette_plot.png")
