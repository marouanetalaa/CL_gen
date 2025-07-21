from __future__ import annotations

import random
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


class NoiseMode(str, Enum):
    """Available strategies for noise generation."""

    NEUTRAL = "neutral"
    SUPPORTIVE = "supportive"
    CONTRADICTORY = "contradictory"


class QueryConfig(BaseModel):
    """Configuration for query generation."""

    T: int = Field(..., description="Longueur totale de la séquence")
    subject: str = Field(..., alias="s", description="Sujet de la requête")
    relation: str = Field(..., alias="r", description="Relation de la requête")
    linking_word: str = Field(..., alias="ell", description="Mot de liaison")
    noise_token: str = Field("n", alias="n", description="Token de bruit")
    mode: NoiseMode = Field(NoiseMode.NEUTRAL, description="Stratégie de bruit")
    false_target: Optional[str] = Field(
        None,
        alias="target",
        description="Cible fausse pour le mode contradictoire",
    )

    @validator("T")
    def check_length(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("T must be positive")
        return v


class QueryGeneratorAgent(BaseModel):
    """Agent capable of générer des requêtes factuelles.

    L'agent repose sur `pydantic` pour la validation de la configuration et peut
    optionnellement utiliser l'SDK OpenAI pour générer dynamiquement des tokens de
    bruit si un client est fourni.
    """

    config: QueryConfig
    client: Optional["OpenAI"] = None

    class Config:
        arbitrary_types_allowed = True

    def _generate_noise(self, length: int) -> List[str]:
        """Generate noise tokens depending on the chosen mode."""
        if length <= 0:
            return []
        if self.client is None:
            return [self.config.noise_token] * length

        s = self.config.subject
        r = self.config.relation
        if self.config.mode is NoiseMode.SUPPORTIVE:
            prompt = (
                f"Rédige {length} tokens en français soutenant l'idée que {s} {r}."
            )
        elif self.config.mode is NoiseMode.CONTRADICTORY:
            target = self.config.false_target or "quelque chose d'autre"
            prompt = (
                f"Rédige {length} tokens en français laissant entendre que {target} "
                f"plutôt que {s} {r}."
            )
        else:
            prompt = (
                f"Rédige {length} tokens français cohérents sans mentionner que {s} {r}."
            )

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        tokens = response.choices[0].message.content.split()
        if len(tokens) < length:
            tokens.extend([self.config.noise_token] * (length - len(tokens)))
        return tokens[:length]

    def generate_query(self) -> List[str]:
        """Génère une seule séquence selon la description du README."""
        T = self.config.T
        s_tokens = self.config.subject.split()
        r_tokens = self.config.relation.split()
        ell_token = self.config.linking_word

        required = len(s_tokens) + len(r_tokens) + 1
        if T < required:
            raise ValueError("T is too small for the given tokens")

        max_start = T - len(s_tokens) - len(r_tokens) - 1
        i = random.randint(0, max_start)
        j = random.randint(i + len(s_tokens), T - len(r_tokens) - 1)

        n1_len = i
        n2_len = j - i - len(s_tokens)
        n3_len = T - j - len(r_tokens) - 1

        sequence = (
            self._generate_noise(n1_len)
            + s_tokens
            + self._generate_noise(n2_len)
            + r_tokens
            + self._generate_noise(n3_len)
            + [ell_token]
        )
        return sequence

    def generate_queries(self, count: int = 1) -> List[List[str]]:
        """Generate plusieurs séquences Q(i,j)."""
        return [self.generate_query() for _ in range(count)]
