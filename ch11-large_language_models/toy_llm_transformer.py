from collections import Counter

# Stop merging when top bigram frequency falls to this threshold
FREQ_THRESHOLD = 3

def character_tokenizer(text):
    # Split into character tokens
    return list(text)

def bigram_frequency_table(tokens):
    # Count adjacent-token pairs
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    return Counter(bigrams)

def merge_bigram(tokens, bigram):
    # Merge every adjacent occurrence of bigram into one token.
    # E.g. if bigram=('a','l'), 'a','l' → 'al'
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == bigram:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged

# Step 1: Input text (demo)
# In practice, read a larger corpus from file(s)
text = (
    "alice was beginning to get very tired of sitting by her sister on the bank,\n"
    "and of having nothing to do : \n"
    "once or twice she had peeped into the book her sister was reading , \n"
    "but it had no pictures or conversations in it , \n"
    "\" and what is the use of a book,\" thought alice \"without pictures or conversations ? \""
)
tokens = character_tokenizer(text)

# Step 1: Inspect character tokens
print("Step 1: character tokens")
print("Shows the full list of character tokens parsed from the input text.")
print(" ".join(tokens), "\n")

# Step 2: Bigram frequency table
bigram_table = bigram_frequency_table(tokens)
sorted_bigrams = sorted(bigram_table.items(), key=lambda kv: kv[1], reverse=True)

print("Step 2: initial bigram frequencies")
print("Displays bigram (adjacent token pair) counts sorted by frequency.")
for i, (bg, freq) in enumerate(sorted_bigrams, start=1):
    print(f"  {i}. {bg!r} → {freq}")
print()

# Step 3: Merge frequent bigrams until threshold
print("Step 3: bigram merges until threshold")
print("Repeatedly merge the most frequent adjacent token pair, then show the merged pair and updated bigram frequencies after each merge.")
step = 3
while True:
    bigram_table = bigram_frequency_table(tokens)
    if not bigram_table or bigram_table.most_common(1)[0][1] <= FREQ_THRESHOLD:
        break

    top_bigram, top_count = bigram_table.most_common(1)[0]
    tokens = merge_bigram(tokens, top_bigram)

    print(f"Step {step}: merged {top_bigram!r} (count={top_count})")
    print("  New tokens:")
    print("    " + " ".join(tokens))

    # Print updated bigram frequencies
    updated = bigram_frequency_table(tokens)
    sorted_updated = sorted(updated.items(), key=lambda kv: kv[1], reverse=True)
    print("  Updated bigram frequencies:")
    for i, (bg, freq) in enumerate(sorted_updated, start=1):
        print(f"    {i}. {bg!r} → {freq}")
    print()

    step += 1

# Step 3: Final merged tokens
print("Final tokens:")
print("Final token list after all merges are complete.")
print(" ".join(tokens))

# Step 4: Map tokens to IDs and build vocab
def tokens_to_id_stream(tokens):
    # Map tokens to IDs in first-appearance order; return ID stream and vocab
    vocab, id_stream = {}, []
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab) # Next free ID
        id_stream.append(vocab[tok])
    return id_stream, vocab

id_stream, vocab = tokens_to_id_stream(tokens)
PAD_TOKEN = "<PAD>"
pad_id = vocab.setdefault(PAD_TOKEN, len(vocab))

print("\nStep 4: token → ID mapping (first 20)")
print("Shows the first 20 tokens with their assigned integer IDs, then reports vocabulary size and total stream length.")
for tok, idx in list(vocab.items())[:20]:
    print(f"  id {idx:<3}  token '{tok}'")
print(f"Total vocab size: {len(vocab)}")
print(f"Total IDs in stream (token river): {len(id_stream)}")

# Step 5: Pack ID stream into fixed-length blocks
print("\nStep 5: pack ID stream into fixed-length blocks")
print("Splits the ID stream into equal-sized training blocks and prints their contents.")
BLOCK_SIZE = 32
blocks = [id_stream[i:i+BLOCK_SIZE]
          for i in range(0, len(id_stream), BLOCK_SIZE)]

# Right-pad final block to exactly BLOCK_SIZE
if len(blocks[-1]) < BLOCK_SIZE:
    pad_id = vocab.get("<PAD>", len(vocab)) # Define PAD if missing
    blocks[-1].extend([pad_id] * (BLOCK_SIZE - len(blocks[-1])))

# Show vocabulary
print("\nFull vocabulary")
for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]): # Sort by ID
    print(f"{idx}\t{tok}")

# Save vocabulary to CSV
import csv, pathlib
out_path = pathlib.Path("vocab.csv")
with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "token"])
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        # Human-readable escape for control chars
        printable = tok.encode("unicode_escape").decode()
        writer.writerow([idx, printable])

def pack_id_stream(id_stream, block_size=2048, pad_token="<PAD>", vocab=None):
    # Slice the ID stream into equal-length blocks and pad the final block
    if vocab is None:
        raise ValueError("Need the vocab dict so we know the PAD ID.")

    pad_id = vocab.setdefault(pad_token, len(vocab)) # Add PAD if missing

    # Slice the stream
    blocks = [id_stream[i:i + block_size]
              for i in range(0, len(id_stream), block_size)]

    # Right-pad final block
    if len(blocks[-1]) < block_size:
        blocks[-1].extend([pad_id] * (block_size - len(blocks[-1])))

    return blocks

# Step 5 (cont.): Build blocks for training
BLOCK_SIZE = 32
blocks = pack_id_stream(id_stream, BLOCK_SIZE, "<PAD>", vocab)

print(f"\nStep 5: token river packed into {len(blocks)} blocks")
print("Summary of how many training blocks were created from the ID stream.")
for b in blocks:
    print("  ", b)


# Step 6: Build toy Transformer and run a forward+loss pass
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
VOCAB_SIZE = len(vocab)   # [V]
CTX_LEN    = BLOCK_SIZE   # [T]
D_MODEL    = 64           # [d]
N_HEADS    = 4            # [h]
N_LAYERS   = 2            # [L]
D_FF       = 256          # [d_ff]

device = "cuda" if torch.cuda.is_available() else "cpu"

class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, ctx_len, d_model, n_heads, n_layers, d_ff):  # vocab_size [V], ctx_len [T], d_model [d]
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Parameter(self._init_pos(ctx_len, d_model), requires_grad=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.tblock = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    @staticmethod
    def _init_pos(seq_len, d_model_dim):  # seq_len [T], d_model_dim [d]
        # Fixed sinusoidal positional encodings
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        i   = torch.arange(d_model_dim, dtype=torch.float32).unsqueeze(0)
        angles = pos / (10000 ** (2 * (i // 2) / d_model_dim))
        pe = torch.zeros(seq_len, d_model_dim)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe # Shape (seq_len, d_model_dim)

    def forward(self, token_ids):  # token_ids [idx]
        # token_ids: (B,T) → logits: (B,T,V)
        x = self.embed(token_ids) + self.pos[: token_ids.size(1)]
        x = self.tblock(x)
        return self.lm_head(x)

# Build the model
model = ToyTransformer(
    vocab_size=VOCAB_SIZE,
    ctx_len=CTX_LEN,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
).to(device)

# Step 6: one forward pass and loss
print("\nStep 6: one forward pass and loss")
print("Description: Runs the Transformer on a sample block and reports next-token cross-entropy loss (ignoring PAD).")
# Take one packed block as a mini-batch
batch_ids = torch.tensor(blocks[:1], dtype=torch.long, device=device) # (1, 32)

# Next-token targets (shifted left by 1)
targets = batch_ids.roll(-1, dims=1)

# Forward + loss
logits = model(batch_ids)  # (1, 32, V)
loss   = F.cross_entropy(
    logits.view(-1, VOCAB_SIZE),
    targets.view(-1),
    ignore_index=pad_id
)

print(f"\nStep 6: forward pass done  →  loss = {loss.item():.4f}")

print("\nStep 7: training loop")
print("Optimizes the model over all blocks, printing running average loss each epoch.")
# Step 7: Training loop
import random, math, time

EPOCHS       = 5
BATCH_SIZE   = 4 # blocks per step (1 block = 32 tokens)
LR           = 3e-3
GRAD_CLIP    = 1.0

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def sample_batch(blocks, batch_size):
    # Return (B,T) int64 tensor of token IDs.
    selected_blocks  = random.sample(blocks, batch_size)
    return torch.tensor(selected_blocks, dtype=torch.long, device=device)

start = time.time()
for epoch in range(1, EPOCHS + 1):
    random.shuffle(blocks)
    steps_per_epoch = math.ceil(len(blocks) / BATCH_SIZE)

    running = 0.0
    for step_idx in range(steps_per_epoch):
        # Fetch & shift targets
        batch = sample_batch(blocks, BATCH_SIZE) # (B, T)
        targets = batch.roll(-1, dims=1)

        # Forward / loss
        logits = model(batch)
        loss   = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            targets.view(-1),
        )

        # Backprop + update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running += loss.item()

        # Print every 20 updates
        if (step_idx + 1) % 20 == 0 or step_idx == steps_per_epoch - 1:
            avg = running / (step_idx + 1)
            elapsed = time.time() - start
            print(f"epoch {epoch}/{EPOCHS}  step {step_idx+1:>3}/{steps_per_epoch}"
                  f"  loss {avg:6.3f}  ({elapsed:.1f}s)")

print("\nTraining done!")

print("\nStep 7 (sanity): generate sample")
print("Generates a short continuation using greedy decoding from a single seed token.")
# Step 7 (sanity check): Generate 40 tokens
def greedy_generate(model, start_ids, max_new=40):
    model.eval()
    ids = start_ids.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids[:, -CTX_LEN:]) # Keep ctx window
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    return ids[0].tolist()

# seed with the single token "a"
seed = torch.tensor([[vocab['a']]], dtype=torch.long, device=device)
generated_ids = greedy_generate(model, seed, 40)
generated_tokens = [next(tok for tok, idx in vocab.items() if idx == i)
                    for i in generated_ids]

print("\nSample generation:")
print("".join(generated_tokens))


# Step 8: Validation, checkpoint save/load
print("\nStep 8: validation and checkpoint")
print("Measures validation perplexity, saves a checkpoint, reloads it, and samples again.")

VAL_FRACTION = 0.15 # 15 % of blocks for validation
CHECKPOINT   = pathlib.Path("toy_llm_transformer.pt")

# Split train/valid blocks
split = int(len(blocks) * (1 - VAL_FRACTION))
train_blocks, val_blocks = blocks[:split], blocks[split:]

def batch_loss(model, batch_token_ids):
    targets = batch_token_ids.roll(-1, dims=1)
    logits = model(batch_token_ids)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
    return loss

def evaluate(model, val_blocks, batch_size=8):
    model.eval()
    with torch.no_grad():
        total, seen = 0.0, 0
        for i in range(0, len(val_blocks), batch_size):
            batch = torch.tensor(val_blocks[i:i+batch_size],
                                 dtype=torch.long, device=device)
            total += batch_loss(model, batch).item() * batch.size(0)
            seen  += batch.size(0)
    ppl = math.exp(total / seen)
    return ppl

# Compute validation perplexity
val_ppl = evaluate(model, val_blocks)
print(f"\nValidation perplexity: {val_ppl:.2f}")

# Save checkpoint
torch.save({
    "model_state": model.state_dict(),
    "vocab": vocab,
}, CHECKPOINT)
print(f"Checkpoint written → {CHECKPOINT.resolve()}")

# Reload checkpoint and test
reloaded = ToyTransformer(vocab_size=VOCAB_SIZE, ctx_len=CTX_LEN,
                          d_model=D_MODEL, n_heads=N_HEADS,
                          n_layers=N_LAYERS, d_ff=D_FF).to(device)
state = torch.load(CHECKPOINT, map_location=device)
reloaded.load_state_dict(state["model_state"], strict=True)
reloaded.eval()

# Quick generation with the reloaded model
print("\nQuick generation with the reloaded model")
print("Verifies the checkpoint by sampling a short continuation after reload.")
seed = torch.tensor([[vocab['a']]], dtype=torch.long, device=device)
reloaded_generated_ids = greedy_generate(reloaded, seed, 40)
reloaded_generated_tokens = [tok for i in reloaded_generated_ids
                             for tok, idx in vocab.items() if idx == i]

print("\nReloaded model sample:")
print("".join(reloaded_generated_tokens))

print("\nStep 9: interactive sampling (temperature + top-k)")
print("Samples tokens from the trained model with temperature and top-k; includes an optional interactive prompt.")
# Step 9: Interactive sampling (temperature + top-k)
def sample_next_token(model, current_ids, temperature=1.0, top_k=20):
    # Return 1 new ID (torch.int64) using softmax sampling
    logits = model(current_ids[:, -CTX_LEN:]) # (B, T, V)
    logits = logits[:, -1, :] / temperature # last pos
    if top_k is not None:
        # Keep only the k largest, set rest to -Inf so softmax≈0
        topk_values, topk_indices = torch.topk(logits, top_k)
        mask   = logits < topk_values[:, [-1]]
        logits[mask] = -float('inf')
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1) # (B,1)
    return next_id

def generate(model, prompt: str, steps=60,
             temperature=0.8, top_k=20):
    # Return string continuation given a plaintext prompt
    # Map prompt → IDs (char-level for our toy vocab)
    prompt_ids = [vocab[ch] for ch in prompt if ch in vocab]
    if not prompt_ids:
        prompt_ids = [vocab['a']] # Fallback seed
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(steps):
        next_token_id = sample_next_token(model, current_ids,
                                          temperature=temperature, top_k=top_k)
        current_ids = torch.cat([current_ids, next_token_id], dim=1)
    # Decode back to tokens→text
    txt = "".join(next(tok for tok, j in vocab.items() if j == i) for i in current_ids[0])
    return txt

# Demo
print("\nTemperature demo")
for temp in (1.5, 1.0, 0.5):
    out = generate(reloaded, "alice was beginning ", 40, temperature=temp, top_k=20)
    print(f"[T={temp}] {out!r}")

# Optional interactive loop
print("\nType a prompt and press Enter (blank line to quit):")
while True:
    prompt = input("> ")
    if not prompt.strip():
        break
    reply = generate(reloaded, prompt,steps=60, temperature=0.8, top_k=20)
    print(reply)
