#!/usr/bin/env python3
"""
Produce the data layout that the recursive-nanochat `lib.py` expects.

lib.py (recursive nanochat) requires, under DATA_DIR:
  <out>/shard_00000.parquet ...      train shards, each with a 'text' column
  <out>/shard_06542.parquet          the VAL shard (lib.py: MAX_SHARD == VAL_SHARD == 6542)
  <out>/tokenizer/tokenizer.pkl      a pickled tiktoken.Encoding (loaded via pickle.load)
  <out>/tokenizer/token_bytes.pt     1-D LongTensor[n_vocab]; UTF-8 byte length per token,
                                     0 for special tokens (lib.evaluate_bpb masks byte==0)

Grounded against:
  - recursive lib.py: column name 'text', VAL_SHARD=6542, BOS_TOKEN="<|reserved_0|>",
    Tokenizer uses enc.{encode_single_token, n_vocab, encode_ordinary,
    encode_ordinary_batch(num_threads=...), decode}; token_bytes indexed by token id.
  - tiktoken Encoding API: Encoding(name, *, pat_str, mergeable_ranks, special_tokens);
    attrs _pat_str/_mergeable_ranks; n_vocab = max_token_value+1; a *custom-named*
    Encoding pickles BY VALUE (self-contained, no registry needed at load time).

NOTE: this uses a truncated cl100k_base BPE (vocab 32768) as a stand-in for nanochat's
original tokenizer. It is fully functional; absolute BPB is NOT comparable to the paper,
but P-vs-P' differential testing (same tokenizer+data for both) is unaffected.

Usage:
  pip install tiktoken pyarrow torch datasets
  python prepare.py --out /home/ubuntu/data --train-shards 2 --docs-per-shard 2000
  # or, offline, from a local text file (one doc per blank-line-separated block):
  python prepare.py --out /home/ubuntu/data --local-text mytext.txt
"""
import argparse
import os
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
import torch

# Re-export the runtime harness so the autoresearch / leaderboard programs'
#   `from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb`
# resolves here. Those programs expect prepare.py to be BOTH the data builder
# (run via `python prepare.py`) AND the runtime harness — our harness lives in
# lib.py, so we surface its symbols here. main() stays guarded, so importing
# prepare is side-effect-free (no data build is triggered).
from lib import (  # noqa: E402,F401
    MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb,
)

VAL_SHARD = 6542               # MUST match lib.py MAX_SHARD / VAL_SHARD
BOS_TOKEN = "<|reserved_0|>"   # MUST match lib.py BOS_TOKEN


def build_tokenizer(target_vocab: int):
    """Truncated cl100k_base -> a self-contained tiktoken.Encoding with one BOS special token."""
    base = tiktoken.get_encoding("cl100k_base")
    ranks = base._mergeable_ranks                      # dict[bytes, int], ranks 0..N-1
    if target_vocab - 1 > len(ranks):
        raise SystemExit(f"target vocab {target_vocab} too large for base ({len(ranks)} merges)")

    # Keep the lowest (target_vocab - 1) ranks; BOS takes the top id.
    keep = {tok: r for tok, r in ranks.items() if r < target_vocab - 1}
    n_single = sum(1 for tok in keep if len(tok) == 1)
    assert n_single == 256, f"only {n_single}/256 single-byte tokens survived; raise --vocab"

    # Re-rank contiguously 0..k-1, preserving original merge priority (lower rank = higher priority).
    reranked = {tok: i for i, (tok, _) in enumerate(sorted(keep.items(), key=lambda kv: kv[1]))}
    bos_id = len(reranked)                             # = top id, no overlap with ranks

    enc = tiktoken.Encoding(
        name="nanochat_smoke",                         # custom name -> pickled BY VALUE
        pat_str=base._pat_str,
        mergeable_ranks=reranked,
        special_tokens={BOS_TOKEN: bos_id},
    )
    assert enc.n_vocab == bos_id + 1 == target_vocab
    assert enc.encode_single_token(BOS_TOKEN) == bos_id
    return enc, reranked, bos_id


def build_token_bytes(reranked: dict, bos_id: int, n_vocab: int) -> torch.Tensor:
    tb = torch.zeros(n_vocab, dtype=torch.long)
    for tok_bytes, idx in reranked.items():
        tb[idx] = len(tok_bytes)                       # UTF-8 byte length of this token
    tb[bos_id] = 0                                      # special token excluded from BPB
    return tb


def get_docs(args, n: int) -> list:
    if args.local_text:
        with open(args.local_text, "r", encoding="utf-8") as f:
            blocks = [b.strip() for b in f.read().split("\n\n")]
        docs = [b for b in blocks if b]
        if len(docs) < n:
            docs = (docs * (n // max(1, len(docs)) + 1))[:n]
        return docs[:n]
    from datasets import load_dataset
    ds = load_dataset(args.dataset, name=args.config, split="train", streaming=True)
    out = []
    for ex in ds:
        t = ex.get(args.text_key)
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return out


def write_shard(path: str, docs: list) -> None:
    table = pa.table({"text": pa.array(docs, type=pa.string())})
    pq.write_table(table, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ubuntu/data")
    ap.add_argument("--train-shards", type=int, default=2)
    ap.add_argument("--docs-per-shard", type=int, default=2000)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--local-text", default=None, help="bypass HF; read docs from a local file")
    args = ap.parse_args()

    tok_dir = os.path.join(args.out, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)

    print("[1/3] building tokenizer ...")
    enc, reranked, bos_id = build_tokenizer(args.vocab)
    with open(os.path.join(tok_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(enc, f)
    tb = build_token_bytes(reranked, bos_id, enc.n_vocab)
    torch.save(tb, os.path.join(tok_dir, "token_bytes.pt"))
    print(f"      vocab={enc.n_vocab}  bos_id={bos_id}  token_bytes={tuple(tb.shape)}")

    print("[2/3] fetching documents ...")
    k, ns = args.docs_per_shard, args.train_shards
    need = k * (ns + 1)
    docs = get_docs(args, need)
    if len(docs) < need:
        raise SystemExit(f"got {len(docs)} docs, need {need}")

    print("[3/3] writing shards ...")
    for s in range(ns):                                # train shards: shard_00000.parquet ...
        p = os.path.join(args.out, f"shard_{s:05d}.parquet")
        write_shard(p, docs[s * k:(s + 1) * k])
        print(f"      wrote {p}  ({k} docs)")
    vp = os.path.join(args.out, f"shard_{VAL_SHARD:05d}.parquet")  # VAL shard MUST be 06542
    write_shard(vp, docs[ns * k:(ns + 1) * k])
    print(f"      wrote {vp}  ({k} docs)  [VAL]")

    # ---- self-verify by replaying lib.py's exact load + call sequence ----
    print("verifying (mimicking lib.py) ...")
    with open(os.path.join(tok_dir, "tokenizer.pkl"), "rb") as f:
        e2 = pickle.load(f)                             # lib.Tokenizer.from_directory
    assert e2.encode_single_token(BOS_TOKEN) == bos_id
    assert e2.decode(e2.encode_ordinary("Hello world")) == "Hello world"   # ASCII round-trips
    b = e2.encode_ordinary_batch(["alpha", "beta"], num_threads=8)         # lib.encode(list)
    assert isinstance(b, list) and isinstance(b[0], list)
    assert max(e2.encode_ordinary("the quick brown fox") + [bos_id]) < e2.n_vocab
    tb2 = torch.load(os.path.join(tok_dir, "token_bytes.pt"))
    assert tb2.shape[0] == e2.n_vocab and (tb2 >= 0).all()
    assert "text" in pq.read_table(vp).column_names                        # lib._document_batches
    print(f"OK -> data ready under {args.out}  (set DATA_DIR or symlink /data)")

    # All files are written/flushed above. Skip interpreter finalization to dodge a
    # benign native-threadpool teardown crash (tiktoken/pyarrow rust threads):
    # "PyGILState_Release: thread state must be current". Cosmetic, but it makes the
    # process exit non-zero, so we exit cleanly here instead.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
