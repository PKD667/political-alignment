from transformers import AutoTokenizer, AutoModel
import torch

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

class EmbeddingAnalyzer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(device)
    
    def get_embeddings(self, words):
        encoded = self.tokenizer(words, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            model_output = self.model(**encoded)
        return self.mean_pooling(model_output, encoded['attention_mask'])
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def visualize_embeddings(self, words):
        embeddings = self.get_embeddings(words)
        embeddings = torch.tensor(embeddings).squeeze(1) 

        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(normalize(embeddings.cpu().numpy()))

        # Plot the results
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
        for i, word in enumerate(words):
            plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
        plt.title("2D Visualization of Word Embeddings (PCA)")
        plt.show()

    def compute_distance(self, w1, w2):
        embeddings = self.get_embeddings([w1, w2])
        return torch.dist(embeddings[0], embeddings[1]).item()