from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.models import RuleDocument


@dataclass(slots=True)
class LexicalHit:
    document_id: str
    score: float
    snippet: str


class LexicalIndex:
    """
    Lightweight BM25-like lexical index for deterministic legal phrase matching.
    """

    def __init__(self) -> None:
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_text: dict[str, str] = {}
        self.df: dict[str, int] = defaultdict(int)
        self.avg_doc_len = 0.0
        algorithm = os.getenv("PLOTMAGIC_LEXICAL_ALGORITHM", "bm25").strip().lower() or "bm25"
        self.algorithm = "tfidf" if algorithm == "tfidf" else "bm25"

    def set_algorithm(self, algorithm: str) -> None:
        algo = algorithm.strip().lower()
        self.algorithm = "tfidf" if algo == "tfidf" else "bm25"

    def build(self, docs: list[RuleDocument]) -> None:
        self.doc_tokens.clear()
        self.doc_text.clear()
        self.df.clear()
        for doc in docs:
            text = f"{doc.rule_title}. {doc.full_text}"
            tokens = self._tokenize(text)
            self.doc_tokens[doc.document_id] = tokens
            self.doc_text[doc.document_id] = text
            for token in set(tokens):
                self.df[token] += 1
        total_len = sum(len(tokens) for tokens in self.doc_tokens.values())
        self.avg_doc_len = total_len / max(1, len(self.doc_tokens))

    def search(self, query: str, limit: int = 10) -> list[LexicalHit]:
        if self.algorithm == "tfidf":
            return self._search_tfidf(query, limit=limit)
        return self._search_bm25(query, limit=limit)

    def _search_bm25(self, query: str, limit: int = 10) -> list[LexicalHit]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        scores: dict[str, float] = {}
        n_docs = max(1, len(self.doc_tokens))
        k1 = 1.5
        b = 0.75
        for doc_id, tokens in self.doc_tokens.items():
            tf = Counter(tokens)
            doc_len = len(tokens) or 1
            score = 0.0
            for token in q_tokens:
                if token not in tf:
                    continue
                df = self.df.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                term_tf = tf[token]
                score += idf * ((term_tf * (k1 + 1)) / (term_tf + k1 * (1 - b + b * doc_len / self.avg_doc_len)))
            if score > 0:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        hits: list[LexicalHit] = []
        for doc_id, score in ranked:
            text = self.doc_text.get(doc_id, "")
            hits.append(LexicalHit(document_id=doc_id, score=score, snippet=text[:300]))
        return hits

    def _search_tfidf(self, query: str, limit: int = 10) -> list[LexicalHit]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        n_docs = max(1, len(self.doc_tokens))
        q_tf = Counter(q_tokens)
        q_weights: dict[str, float] = {}
        for token, freq in q_tf.items():
            df = self.df.get(token, 0)
            idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
            q_weights[token] = float(freq) * idf
        q_norm = math.sqrt(sum(weight * weight for weight in q_weights.values()))
        if q_norm <= 0.0:
            return []

        scores: dict[str, float] = {}
        for doc_id, tokens in self.doc_tokens.items():
            tf = Counter(tokens)
            dot = 0.0
            d_norm_sq = 0.0
            for token, freq in tf.items():
                df = self.df.get(token, 0)
                idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
                d_weight = float(freq) * idf
                d_norm_sq += d_weight * d_weight
                if token in q_weights:
                    dot += d_weight * q_weights[token]
            d_norm = math.sqrt(d_norm_sq)
            if d_norm <= 0.0:
                continue
            score = dot / (q_norm * d_norm)
            if score > 0.0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        hits: list[LexicalHit] = []
        for doc_id, score in ranked:
            text = self.doc_text.get(doc_id, "")
            hits.append(LexicalHit(document_id=doc_id, score=score, snippet=text[:300]))
        return hits

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower()
        normalized = re.sub(r"[^a-z0-9()\s]", " ", normalized)
        return [self._normalize_token(tok) for tok in normalized.split() if tok]

    def _normalize_token(self, token: str) -> str:
        # Light stemming to improve plural/singular matches in legal text (e.g., objection vs objections).
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token
