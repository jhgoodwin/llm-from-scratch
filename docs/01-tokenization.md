# Part 1: Tokenization

LLMs don't see text — they see sequences of integers. A tokenizer converts between the two.

No file to create here — the tokenizer is built directly into `train.py` (Part 3). This part explains how it works so you understand what you're writing later.

## Character-Level Tokenization

We use the simplest possible tokenizer: each unique character gets an ID.

```python
text = open("data/shakespeare.txt").read()
chars = sorted(set(text))
vocab_size = len(chars)  # 65 for Shakespeare

stoi = {c: i for i, c in enumerate(chars)}  # string to int
itos = {i: c for c, i in stoi.items()}      # int to string

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])
```

```python
encode("Hello")  # [20, 43, 50, 50, 53]
decode([20, 43, 50, 50, 53])  # "Hello"
```

That's it. No libraries, no pretrained models. Shakespeare uses 65 unique characters (letters, digits, punctuation, newlines). Each character becomes one token.

This tokenizer is built directly into our data loading — there's no separate tokenizer file to write.

## Why Character-Level?

Character-level tokenization works well for small datasets like Shakespeare (~1M characters) because:

- **Tiny vocabulary** (65 tokens) — the model's embedding table and output layer are small
- **Dense statistics** — with only 65² = 4,225 possible bigrams, every bigram appears many times in the data
- **No dependencies** — no external libraries needed

This matters more than you'd expect. GPT-2's tokenizer (BPE, 50,257 tokens) turns Shakespeare into ~338k tokens with ~11,700 unique token types. Most token bigrams appear only once or twice — the data is too sparse for the model to learn sequential patterns. We tested this: with BPE on Shakespeare, training loss gets stuck at ~6.3 (unigram frequency) and never improves. With character-level, it drops to ~1.5.

## The Tradeoff

Character-level doesn't scale to large datasets:

- **Sequences are ~3x longer** than BPE (more compute per sample)
- **The model has to learn spelling from scratch** — it has no concept of "words"
- **It wastes capacity on predictable patterns** — `t-h-e` almost always appears together, but the model must predict each character separately

For large datasets (TinyStories, OpenWebText), you'd switch to **Byte-Pair Encoding (BPE)** which groups common character sequences into single tokens. GPT-2 uses a BPE tokenizer with 50,257 tokens. You can use it via `tiktoken`:

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, world!")  # [15496, 11, 995, 0]
text = enc.decode(tokens)             # "Hello, world!"
```

But for this workshop on Shakespeare, character-level is the right choice.

## How It Connects to the Model

The vocabulary size directly determines two things in the model architecture:

1. **The embedding table** — `nn.Embedding(vocab_size, n_embd)` maps each token ID to a learned vector. With vocab_size=65, this is tiny (65 × 384 = 24,960 parameters). With GPT-2's vocab of 50,257, it's 50,257 × 384 = 19.3M parameters — nearly half the model.

2. **The output layer** — `nn.Linear(n_embd, vocab_size)` produces a probability over all possible next tokens. With 65 tokens, the model chooses from 65 options. With 50,257 tokens, it must spread its predictions across 50,257 classes.

## Key Takeaways

- Tokenization converts text ↔ integer sequences
- We use character-level: each unique character gets an ID (`stoi`/`itos` mappings)
- Vocabulary size must match the dataset — 50k vocab on a tiny dataset means most token patterns are too sparse to learn
- Character-level works for small experiments; BPE is needed for large datasets
- The vocab size directly affects model size (embedding table) and training difficulty (output distribution)

## Next: [Part 2 — The Transformer →](02-the-transformer.md)
