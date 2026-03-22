# Citation Contract

Every answer claim must map to a citation object with deterministic provenance.

## Payload Fields

- `claim_id`: stable claim identifier from evidence matrix.
- `ruleset_id`: e.g., `KMBR_1999`, `KPBR_2011`.
- `chapter_number`, `rule_number`, `sub_rule_path`.
- `table_ref`: optional table reference for numeric claims.
- `anchor_id`: source anchor used by frontend deep links.
- `source_file`: canonical file path with hash anchor.
- `display_citation`: human-readable legal citation.
- `quote_excerpt`: direct evidence excerpt used for the claim.

## Source-Click Navigation

- KMBR HTML files use direct source anchors from `<a name="...">`.
  - Example: `data/kerala/kmbr_muncipal_rules/chapter5.html#chapter5-2`
- KPBR markdown uses synthetic deterministic anchors derived from clause IDs.
  - Example: `data/kerala/kpbr_panchayat_rule.md#kpbr-ch5-r35`

## Guardrails

- If a claim lacks citation evidence, the claim must not be included.
- If a sub-question has no evidence, response must include explicit unresolved note.

