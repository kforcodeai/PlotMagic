# KMBR Multi-hop Retrieval Dataset

This folder contains a retrieval-oriented QnA-style benchmark dataset built from Kerala Municipality Building Rules source markdown chapters:

- `data/kerala/kmbr_muncipal_rules_md/chapter*.md`

A combined canonical source file is generated for stable line-range citations:

- `evaluation/kmbr/kmbr_municipality_rules_combined.md`

## Files

- `evaluation/kmbr/generate_kmbr_multihop_dataset.py`
- `evaluation/kmbr/kmbr_multihop_retrieval_dataset.jsonl`
- `evaluation/kmbr/kmbr_multihop_retrieval_dataset.json`
- `evaluation/kmbr/kmbr_multihop_retrieval_benchmark.json`
- `evaluation/kmbr/kmbr_municipality_rules_combined.md`

## Record schema

Each record has:

- `id`: stable query id
- `query`: complex multi-hop query text
- `ground_truth_chunks`: list of exact evidence snippets
  - `citation`: `evaluation/kmbr/kmbr_municipality_rules_combined.md:<start>-<end>`
  - `snippet`: verbatim text from those lines
- `source_document`: combined source file path
- `query_type`: `multi_hop_complex`

## Design intent

- Real-world municipal permitting/compliance workflows (timelines, thresholds, exceptions, authority dependencies)
- Complex multi-hop retrieval stress across chapters (permit, occupancy, fire/lift, telecom, regularization, registration)
- Evidence-first ground truth (no final answers), suitable for end-to-end retrieval pipeline evaluation

## Regeneration

```bash
python evaluation/kmbr/generate_kmbr_multihop_dataset.py
```

## Benchmark command

Use the repository virtual environment python if available:

```bash
./.venv/bin/python scripts/benchmark_retrieval.py \
  --dataset evaluation/kmbr/kmbr_multihop_retrieval_dataset.jsonl \
  --state kerala \
  --jurisdiction municipality \
  --top-k 20 \
  --output evaluation/kmbr/kmbr_multihop_retrieval_benchmark.json
```

The benchmark output includes summary metrics (`hit@k`, `recall@k`, `chunk_recall@k`, `MRR`, latency) and per-query diagnostics.
