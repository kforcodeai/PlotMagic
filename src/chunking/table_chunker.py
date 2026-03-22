from __future__ import annotations

from src.chunking.chunks import Chunk
from src.models import RuleDocument


class TableChunker:
    def chunk(self, doc: RuleDocument, max_rows_per_chunk: int = 30) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, table in enumerate(doc.tables, start=1):
            header_row = " | ".join(table.headers)
            rows = table.rows or []
            if not rows:
                lines = [table.caption or f"Table {index}", header_row]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}-table-{index}",
                        document_id=doc.document_id,
                        level="table",
                        text="\n".join([line for line in lines if line]),
                        metadata={
                            "table_id": table.table_id,
                            "caption": table.caption,
                            "occupancy_groups": doc.occupancy_groups,
                            "row_start": 0,
                            "row_end": 0,
                        },
                    )
                )
                continue
            for part, start in enumerate(range(0, len(rows), max_rows_per_chunk), start=1):
                end = min(len(rows), start + max_rows_per_chunk)
                lines = [table.caption or f"Table {index}", header_row]
                lines.extend([" | ".join(row) for row in rows[start:end]])
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}-table-{index}-p{part}",
                        document_id=doc.document_id,
                        level="table",
                        text="\n".join([line for line in lines if line]),
                        metadata={
                            "table_id": table.table_id,
                            "caption": table.caption,
                            "occupancy_groups": doc.occupancy_groups,
                            "row_start": start + 1,
                            "row_end": end,
                        },
                    )
                )
        return chunks
