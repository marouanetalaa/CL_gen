# CL_gen

Ce dépôt a pour objectif de construire un agent capable de générer des requêtes factuelles. 
Donné une longueur de séquence `T`, un sujet `s`, une relation `r`, un mot de liaison `ℓ` et une stratégie de bruit `n`, 
il produit une liste de séquences `Q(i,j)` où `i` et `j` représentent les positions aléatoires pour insérer `s` et `r`.

## Structure des requêtes

On définit une requête `Q(s, r, n, ℓ)` comme une séquence de longueur fixe `T` :

```
Q = [n1, n2, …, s, ni, ni+1, …, r, …, nj, ℓ]
```

avec :

1. `s` – tokens sujet (par exemple « Paris »)
2. `r` – tokens relation (par exemple « est la capitale »)
3. `n` – tokens de bruit ou de remplissage de longueur `T - |s| - |r| - 1`
4. `ℓ` – mot de liaison (par exemple « est », « de », « le »)

Le notebook `notebooks/en/faiss_with_hf_datasets_and_clip.ipynb` présente un exemple d'utilisation autour de l'indexation de données multimodales.

## Utilisation de l'agent

Le module `cl_gen.augment` fournit une implémentation minimaliste d'un agent de
génération de requêtes basée sur `pydantic`. Si l'SDK `openai` est disponible,
il peut être utilisé pour générer dynamiquement les tokens de bruit.

```python
from cl_gen.augment import QueryConfig, QueryGeneratorAgent

cfg = QueryConfig(T=10, s="Paris", r="est la capitale", ell="de", mode="supportive")
agent = QueryGeneratorAgent(config=cfg)
print(agent.generate_queries(2))
```
