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

    def __del__(self):
        del self.model
        torch.cuda.empty_cache()

    def debug_embeddings(self, w1, w2):
        emb = self.get_embeddings([w1, w2])
        print(f"Embedding shapes: {emb.shape}")
        print(f"Norms: {torch.norm(emb[0])}, {torch.norm(emb[1])}")
        print(f"Sample values: {emb[0,:5]}, {emb[1,:5]}")  # first 5 values

    def debug_tokens(self, w1, w2):
        # Show raw tokens
        t1 = self.tokenizer.encode(w1)
        t2 = self.tokenizer.encode(w2)
        print(f"Token ids for {w1}: {t1}")
        print(f"Token ids for {w2}: {t2}")

        # Show actual tokens
        print(f"Tokens for {w1}: {self.tokenizer.convert_ids_to_tokens(t1)}")
        print(f"Tokens for {w2}: {self.tokenizer.convert_ids_to_tokens(t2)}")

    def weighted_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        seq_length = token_embeddings.size(1)

        # Create exponential decay weights
        position = torch.arange(seq_length, device=token_embeddings.device)
        decay_rate = 0.5  # we can tune this
        weights = torch.exp(decay_rate * position)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # shape: [1, seq_length, 1]

        print(f"Weights for {seq_length} tokens: {weights.squeeze()}")

        # Apply attention mask and weights
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        weighted_embeddings = token_embeddings * input_mask_expanded * weights

        # Sum and normalize
        sum_embeddings = torch.sum(weighted_embeddings, 1)
        sum_weights = torch.sum(input_mask_expanded * weights, 1)
        return sum_embeddings / torch.clamp(sum_weights, min=1e-9)

    def analyze_direction(self,v1, v2):
        diff = v2 - v1  # direction vector
        # Get top components (most significant dimensions)
        values, indices = torch.sort(torch.abs(diff), descending=True)
        print(f"Top contributing dimensions: {indices[:10]}")
        print(f"With values: {values[:10]}")
        return diff
    
    def get_embeddings(self, words):
        all_embeddings = []
        for word in words:
            encoded = self.tokenizer([word], return_tensors='pt')
            print(f"Processing '{word}':")
            print(f"Input id: {encoded['input_ids']}")

            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                model_output = self.model(**encoded).last_hidden_state
                print(f"Raw output first values: {model_output[0,0,:5]}\n")

            embedding = self.weighted_pooling(model_output, encoded['attention_mask'])
            all_embeddings.append(embedding)

        return torch.cat(all_embeddings, dim=0)

    def compute_distance(self, w1, w2):

        self.debug_tokens(w1, w2)

        emb = self.get_embeddings([w1, w2])

        self.debug_embeddings(w1, w2)

        # Normalize
        emb = emb / emb.norm(dim=1, keepdim=True)
        
        return torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0).item()
    
    def compute_direction_diff(self, w1, w2):

        self.debug_tokens(w1, w2)

        emb = self.get_embeddings([w1, w2])

        self.debug_embeddings(w1, w2)

        # Normalize
        emb = emb / emb.norm(dim=1, keepdim=True)

        d1 = self.analyze_direction(emb[0], emb[1])
        
        return d1
    
    def show_tokenization(self, word):
        tokens = self.tokenizer.encode(word)
        return self.tokenizer.convert_ids_to_tokens(tokens)
    
    def run_sanity_checks(self):
        # Test known relationships
        pairs = [
            ("king", "queen"),
            ("man", "woman"),
            ("happy", "sad"),
        ]
        for w1, w2 in pairs:

            dist = self.compute_distance(w1, w2)
            print(f"{w1} - {w2}: {dist}")
    
    def test_analogies(self):
        # king:queen :: man:woman
        king = self.get_embeddings(["king"])[0]
        queen = self.get_embeddings(["queen"])[0]
        man = self.get_embeddings(["man"])[0]
        woman = self.get_embeddings(["woman"])[0]

        # The vectors should be roughly parallel
        delta1 = king - queen
        delta2 = man - woman
        similarity = torch.nn.functional.cosine_similarity(delta1, delta2, dim=0)
        return similarity.item()