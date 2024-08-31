# Political Alignement

Tools to study LLM political alignement

## Political Evaluation
You can evaluate an AI political bias automatically and plot it in a nice political compass.
Inspired by PoliTune [[1]](#1).

## Embedding analysis

Embedding space conveys a lot of information about political biases in AI models. `In the embedding.py` you can find tools to analyse it.

**Example**

GPT-2 base :
```
>>> from embedding import compute_distance
>>> compute_distance("marx","stalin")
2249.33056640625
```

MarxGPT-2
```
>>> from embedding import compute_distance
>>> compute_distance("marx","stalin")
1984.6358642578125
```

### References

<a id="1">[1]</a> 
Ahmed Agiza1, Mohamed Mostagir2, Sherief Reda (2024).
[PoliTune: Analyzing the Impact of Data Selection and Fine-Tuning on Economic
and Political Biases in Large Language Models](https://arxiv.org/abs/2404.08699)
