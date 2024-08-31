from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("pkd/MarxGPT-2-aligned")

def get_embedding(word):
    encoded = tokenizer(word, return_tensors='pt')
    model_output = model(**encoded)
    #print(model_output.shape)
    return mean_pooling(model_output, encoded['attention_mask'])


# This visualization is very bad, seems close to random.
def visualize_embeddings(words):
    embeddings = [get_embedding(word).detach().numpy() for word in words]
    embeddings = torch.tensor(embeddings).squeeze(1)

    # Reduce to 2D using t-SNE
    if len(words) < 3:
        raise ValueError("At least 3 words are required for t-SNE visualization")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words) - 1))
    reduced_embeddings = tsne.fit_transform(embeddings.numpy())

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.title("2D Visualization of Word Embeddings")
    plt.show()

def compute_distance(w1, w2):
    emb1 = get_embedding(w1)
    emb2 = get_embedding(w2)
    return torch.dist(emb1, emb2).item()

