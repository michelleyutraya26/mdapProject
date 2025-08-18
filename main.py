from pathlib import Path
from itertools import islice
from text_processing import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

data_dir = Path("data/determinations")
n_docs = 50  # bump to None for all docs after testing

file_names = []
raw_texts = []

for p in islice(sorted(data_dir.glob("*.txt")), n_docs):
    text = p.read_text(encoding="utf-8", errors="ignore")
    tokens = preprocess(text)             # now fast (tokenizer-only)
    if tokens:
        file_names.append(p.name)
        raw_texts.append(" ".join(tokens))  # join tokens to string

print(f"Preprocessed {len(raw_texts)} files.")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 1),      # try (1,2) later
    min_df=10,
    max_df=0.5,
    max_features=30000,
    dtype=np.float32,
)
X = vectorizer.fit_transform(raw_texts)
print("TF-IDF shape:", X.shape)

Ks = range(5, 31, 5)
inertias = []
for k in Ks:
    km = MiniBatchKMeans(n_clusters=k, n_init=5, batch_size=2048, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(list(Ks), inertias, marker="o")
plt.xlabel("k (clusters)")
plt.ylabel("SSE (inertia)")
plt.title("Elbow Method (TF-IDF + MiniBatchKMeans)")
plt.savefig("elbow_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved elbow plot to elbow_plot.png")
