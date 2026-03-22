# Evaluation Harness

Kerala baseline evaluation data is stored in `tests/evaluation/gold_queries_kerala.json`.

## Gold Query Taxonomy

- `scope_resolution`
- `occupancy_mapping`
- `numeric_table`
- `mixed_use`
- `cross_reference`

## Metrics

Thresholds are defined in `config/evaluation/metrics.yaml`:

- jurisdiction accuracy
- occupancy/category accuracy
- clause recall@20
- citation precision
- numeric correctness
- unsupported claim rate

## Running Evaluation

```bash
python scripts/evaluate.py --state kerala
```

