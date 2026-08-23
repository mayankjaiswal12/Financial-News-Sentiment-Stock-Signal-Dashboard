"""
Stage 2 -- transformer sentiment scoring.

Model: ProsusAI/finbert. A BERT-base encoder further pre-trained on a financial
corpus (Reuters TRC2) and fine-tuned on the Financial PhraseBank for 3-way
sentiment {positive, negative, neutral}.

WHY FinBERT rather than a general-purpose sentiment model (e.g. SST-2 DistilBERT):
financial language inverts everyday polarity. "Shares plunge on shortfall" is
unambiguously negative, but so is "Company beats estimates, stock falls on guidance."
Words like *liability*, *aggressive*, *exposure*, *short* are neutral-to-positive
in general English and loaded in finance. A model trained on movie reviews gets
these systematically wrong. FinBERT was fine-tuned on analyst-written financial
sentences, so it has the right priors.

WHY not just a lexicon (Loughran-McDonald):
a lexicon is bag-of-words -- it cannot represent negation or contrast. "Not as bad
as feared" scores negative under LM and positive under FinBERT. Attention handles
the scope of negation; a word list cannot.

--------------------------------------------------------------------------
PERFORMANCE: the three things that make this ~15x faster than the naive loop
--------------------------------------------------------------------------
1. Caching in SQL, keyed by (headline_uid, model). Inference is the only
   expensive step in the pipeline; we pay it exactly once per headline, ever.
2. Length-sorted batching. Padding is per-batch, so mixing a 6-token and a
   60-token headline in one batch pads the short one to 60 and wastes ~90% of
   the FLOPs. Sorting by token length first makes every batch nearly uniform,
   then we restore the original order. Typical speedup here: 2-3x.
3. `torch.inference_mode()` + fp16 on GPU. inference_mode is strictly stronger
   than no_grad (it also skips version counters / view tracking).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..config import SETTINGS
from ..db import read_sql, sentiment as sentiment_tbl, upsert


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple Silicon (MPS), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FinBertScorer:
    """Thin, batched wrapper around the HF sequence-classification head."""

    def __init__(self, model_name: str | None = None, device: torch.device | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name or SETTINGS.hf_model
        self.device = device or pick_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        # eval() disables dropout. Forgetting this is a classic bug: the model
        # would return slightly different scores on every run.
        self.model.eval()

        # fp16 halves memory traffic and roughly doubles throughput on GPU.
        # We deliberately do NOT do this on CPU, where fp16 is emulated and slower.
        if self.device.type == "cuda":
            self.model.half()

        # Never hardcode label order. ProsusAI/finbert ships id2label as
        # {0: 'positive', 1: 'negative', 2: 'neutral'} -- which is NOT the
        # alphabetical order most people assume. Reading it from the config is
        # the difference between a working signal and a sign-flipped one.
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.idx_pos = next(i for i, l in id2label.items() if l.startswith("pos"))
        self.idx_neg = next(i for i, l in id2label.items() if l.startswith("neg"))
        self.idx_neu = next(i for i, l in id2label.items() if l.startswith("neu"))

    @torch.inference_mode()
    def score(self, texts: list[str], batch_size: int | None = None,
              verbose: bool = True) -> pd.DataFrame:
        """Return a DataFrame of [p_neg, p_neu, p_pos, score, label], row-aligned to `texts`."""
        bs = batch_size or SETTINGS.batch_size
        n = len(texts)
        if n == 0:
            return pd.DataFrame(columns=["p_neg", "p_neu", "p_pos", "score", "label"])

        # --- length-sorted batching (optimisation 2) ------------------------
        # Character length is a cheap, monotone proxy for token length; using it
        # avoids tokenising the whole corpus twice just to plan the batches.
        order = np.argsort([len(t) for t in texts], kind="stable")
        probs = np.empty((n, 3), dtype=np.float32)

        for start in range(0, n, bs):
            idx = order[start:start + bs]
            batch = [texts[i] for i in idx]

            enc = self.tokenizer(
                batch,
                padding=True,            # pad to the longest IN THIS BATCH only
                truncation=True,
                max_length=SETTINGS.max_seq_len,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits
            # softmax in float32 for numerical stability even when the model is fp16
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probs[idx] = p

            if verbose and (start // bs) % 20 == 0:
                print(f"  [finbert] {min(start + bs, n):,}/{n:,}", flush=True)

        out = pd.DataFrame({
            "p_neg": probs[:, self.idx_neg],
            "p_neu": probs[:, self.idx_neu],
            "p_pos": probs[:, self.idx_pos],
        })
        # A single scalar signal in [-1, 1]. Using the probability DIFFERENCE
        # rather than argmax keeps the model's confidence: a 0.95-positive
        # headline should move the daily average more than a 0.40-positive one.
        # Neutral mass cancels out automatically since it appears in neither term.
        out["score"] = out["p_pos"] - out["p_neg"]
        out["label"] = np.select(
            [out["score"] > 0.15, out["score"] < -0.15],
            ["pos", "neg"], default="neu",
        )
        return out


def score_headlines(limit: int | None = None, batch_size: int | None = None) -> int:
    """Score every headline that does not yet have a cached score for this model.

    The LEFT JOIN ... WHERE s.headline_uid IS NULL is the cache lookup: it selects
    exactly the un-scored rows, so re-running this is free and interrupting it
    mid-way loses nothing.
    """
    model_name = SETTINGS.hf_model
    q = """
        SELECT h.headline_uid, h.title
        FROM headlines h
        LEFT JOIN sentiment s
          ON s.headline_uid = h.headline_uid AND s.model = :model
        WHERE s.headline_uid IS NULL
        ORDER BY h.headline_uid
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    todo = read_sql(q, {"model": model_name})

    if todo.empty:
        print("[finbert] nothing to score -- cache is warm")
        return 0

    scorer = FinBertScorer()
    print(f"[finbert] scoring {len(todo):,} headlines on {scorer.device.type.upper()}")
    import time
    t0 = time.time()
    scores = scorer.score(todo["title"].tolist(), batch_size=batch_size)
    dt = time.time() - t0

    scores.insert(0, "model", model_name)
    scores.insert(0, "headline_uid", todo["headline_uid"].values)
    n = upsert(sentiment_tbl, scores)
    print(f"[finbert] {n:,} scored in {dt:.1f}s ({len(todo)/max(dt,1e-9):.0f} headlines/s)")
    return n


if __name__ == "__main__":
    score_headlines()
