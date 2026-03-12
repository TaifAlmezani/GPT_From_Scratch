"""
Byte-Pair Encoding (BPE) Tokenizer — built from scratch for Arabic text.

Following Chapter 2 of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

Pipeline:
  1. Normalize Arabic text (optional diacritics removal, unicode normalization)
  2. Pre-tokenize into words using regex (handles Arabic + numbers + punctuation)
  3. Initialize vocabulary with individual UTF-8 characters
  4. Iteratively merge the most frequent adjacent pair → BPE
  5. Encode text → list[int],  Decode list[int] → text
"""

import re
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# Arabic text utilities
# ──────────────────────────────────────────────

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")

def normalize_arabic(text: str, remove_diacritics: bool = True) -> str:
    """Normalize Arabic text: NFKC + optional diacritic stripping."""
    text = unicodedata.normalize("NFKC", text)
    if remove_diacritics:
        text = ARABIC_DIACRITICS.sub("", text)
    # Normalize alef variants → ا
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalize taa marbuta → ه
    text = text.replace("ة", "ه")
    return text


# Pre-tokenization pattern: Arabic word | number | non-whitespace punctuation
PRE_TOK_PATTERN = re.compile(
    r"""[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+"""   # Arabic
    r"""|[0-9]+"""                                                      # digits
    r"""|[^\s\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF0-9]+""",  # other
    re.VERBOSE,
)


def pre_tokenize(text: str) -> list[str]:
    """Split text into coarse word-level tokens (keeps Arabic words intact)."""
    return PRE_TOK_PATTERN.findall(text)


# ──────────────────────────────────────────────
# BPE Tokenizer
# ──────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer implemented from scratch.

    Workflow (training):
        1. Pre-tokenize corpus into words
        2. Represent each word as a tuple of characters
        3. Count adjacent-pair frequencies across all words
        4. Merge the most frequent pair, update representations
        5. Repeat until vocab reaches desired size

    Workflow (encoding):
        Apply learned merges in priority order to new text.
    """

    SPECIAL_TOKENS = {
        "<|pad|>": 0,
        "<|unk|>": 1,
        "<|bos|>": 2,
        "<|eos|>": 3,
        "<|im_start|>": 4,
        "<|im_end|>": 5,
    }

    def __init__(self, vocab_size: int = 5000, remove_diacritics: bool = True):
        self.target_vocab_size = vocab_size
        self.remove_diacritics = remove_diacritics

        self.merges: list[tuple[str, str]] = []          # ordered merge rules
        self.vocab: dict[str, int] = {}                   # token → id
        self.inverse_vocab: dict[int, str] = {}           # id → token
        self._trained = False

    # ── Training ──────────────────────────────

    def train(self, text: str, verbose: bool = False):
        """Learn BPE merges from *text*."""
        text = normalize_arabic(text, remove_diacritics=self.remove_diacritics)
        words = pre_tokenize(text)

        # word_freqs:  tuple-of-chars  →  count
        word_freqs: dict[tuple[str, ...], int] = Counter(
            tuple(w) for w in words
        )

        # Build initial char vocab
        chars = set()
        for word_tuple in word_freqs:
            chars.update(word_tuple)

        # Reserve ids for special tokens, then chars
        self.vocab = dict(self.SPECIAL_TOKENS)
        idx = len(self.vocab)
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1

        num_merges = self.target_vocab_size - len(self.vocab)
        if num_merges <= 0:
            self._build_inverse()
            self._trained = True
            return

        for i in range(num_merges):
            pair_counts = self._count_pairs(word_freqs)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            merged_token = best_pair[0] + best_pair[1]

            self.merges.append(best_pair)
            self.vocab[merged_token] = idx
            idx += 1

            word_freqs = self._merge_pair(word_freqs, best_pair)

            if verbose and (i + 1) % 200 == 0:
                print(f"  merge {i+1}/{num_merges}: "
                      f"'{best_pair[0]}' + '{best_pair[1]}' → '{merged_token}'  "
                      f"(freq {pair_counts[best_pair]})")

        self._build_inverse()
        self._trained = True
        if verbose:
            print(f"Training complete. Vocab size: {len(self.vocab)}")

    @staticmethod
    def _count_pairs(word_freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for word_tuple, freq in word_freqs.items():
            for a, b in zip(word_tuple, word_tuple[1:]):
                counts[(a, b)] += freq
        return counts

    @staticmethod
    def _merge_pair(
        word_freqs: dict[tuple[str, ...], int],
        pair: tuple[str, str],
    ) -> dict[tuple[str, ...], int]:
        new_freqs: dict[tuple[str, ...], int] = {}
        merged = pair[0] + pair[1]
        for word_tuple, freq in word_freqs.items():
            new_word: list[str] = []
            i = 0
            while i < len(word_tuple):
                if (i < len(word_tuple) - 1
                        and word_tuple[i] == pair[0]
                        and word_tuple[i + 1] == pair[1]):
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    # ── Encoding / Decoding ───────────────────

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Encode *text* → list of token ids."""
        assert self._trained, "Tokenizer has not been trained yet."
        text = normalize_arabic(text, remove_diacritics=self.remove_diacritics)
        words = pre_tokenize(text)

        ids: list[int] = []
        if add_special:
            ids.append(self.vocab["<|bos|>"])

        for word in words:
            tokens = list(word)  # start at character level
            for pair in self.merges:
                tokens = self._apply_merge(tokens, pair)
            for tok in tokens:
                ids.append(self.vocab.get(tok, self.vocab["<|unk|>"]))

        if add_special:
            ids.append(self.vocab["<|eos|>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode list of token ids → text."""
        tokens = []
        for i in ids:
            tok = self.inverse_vocab.get(i, "<|unk|>")
            if tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return " ".join(self._merge_adjacent_arabic(tokens))

    @staticmethod
    def _merge_adjacent_arabic(tokens: list[str]) -> list[str]:
        """Heuristic: join consecutive Arabic sub-word pieces without space."""
        if not tokens:
            return tokens
        merged = [tokens[0]]
        arabic_re = re.compile(r"[\u0600-\u06FF]")
        for tok in tokens[1:]:
            if arabic_re.search(tok) and arabic_re.search(merged[-1]):
                merged[-1] += tok
            else:
                merged.append(tok)
        return merged

    @staticmethod
    def _apply_merge(tokens: list[str], pair: tuple[str, str]) -> list[str]:
        merged_token = pair[0] + pair[1]
        new_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]):
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    # ── Persistence ───────────────────────────

    def save(self, path: str):
        """Save tokenizer to JSON file."""
        data = {
            "vocab_size": self.target_vocab_size,
            "remove_diacritics": self.remove_diacritics,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"],
                  remove_diacritics=data["remove_diacritics"])
        tok.merges = [tuple(p) for p in data["merges"]]
        tok.vocab = data["vocab"]
        tok._build_inverse()
        tok._trained = True
        return tok

    # ── Helpers ────────────────────────────────

    def _build_inverse(self):
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<|eos|>"]

    def get_token_details(self, text: str) -> list[dict]:
        """Return detailed info for each token (useful for visualization)."""
        text = normalize_arabic(text, remove_diacritics=self.remove_diacritics)
        words = pre_tokenize(text)
        details = []
        for word in words:
            tokens = list(word)
            for pair in self.merges:
                tokens = self._apply_merge(tokens, pair)
            for tok in tokens:
                tid = self.vocab.get(tok, self.vocab["<|unk|>"])
                details.append({"token": tok, "id": tid, "source_word": word})
        return details

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return (f"BPETokenizer(vocab_size={self.vocab_size}, "
                f"merges={len(self.merges)}, trained={self._trained})")
