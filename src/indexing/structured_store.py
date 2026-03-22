from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

from src.models import ClauseNode, RuleDocument


class StructuredStore:
    """
    Authoritative structured index. SQLite is used here for local development;
    swap connection string for PostgreSQL in production.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(
                """
            CREATE TABLE IF NOT EXISTS rules (
                document_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                jurisdiction_type TEXT NOT NULL,
                ruleset_id TEXT NOT NULL,
                ruleset_version TEXT NOT NULL,
                chapter_number INTEGER,
                chapter_title TEXT,
                rule_number TEXT,
                rule_title TEXT,
                full_text TEXT,
                anchor_id TEXT,
                source_file TEXT,
                occupancy_groups TEXT,
                is_generic INTEGER,
                panchayat_category TEXT,
                conditions_json TEXT,
                numeric_values_json TEXT,
                effective_from TEXT,
                effective_to TEXT
            );

            CREATE TABLE IF NOT EXISTS clauses (
                clause_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                clause_type TEXT NOT NULL,
                display_citation TEXT,
                sub_rule_path TEXT,
                anchor_id TEXT,
                normalized_text TEXT,
                topic_tags TEXT,
                occupancy_groups TEXT,
                is_generic INTEGER,
                parent_clause_id TEXT,
                FOREIGN KEY(document_id) REFERENCES rules(document_id)
            );

            CREATE TABLE IF NOT EXISTS tables_data (
                table_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                caption TEXT,
                headers_json TEXT,
                rows_json TEXT,
                FOREIGN KEY(document_id) REFERENCES rules(document_id)
            );

            CREATE TABLE IF NOT EXISTS table_cells (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                col_index INTEGER NOT NULL,
                header TEXT,
                cell_value TEXT,
                FOREIGN KEY(table_id) REFERENCES tables_data(table_id),
                FOREIGN KEY(document_id) REFERENCES rules(document_id)
            );

            CREATE TABLE IF NOT EXISTS cross_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                source_clause_id TEXT NOT NULL,
                target_ref TEXT NOT NULL,
                target_type TEXT NOT NULL,
                normalized_target_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_rules_scope
            ON rules(state, jurisdiction_type, ruleset_id, panchayat_category);

            CREATE INDEX IF NOT EXISTS idx_rules_rule_no
            ON rules(rule_number);
                """
            )
            cross_ref_columns = {
                str(row["name"])
                for row in cur.execute("PRAGMA table_info(cross_references)").fetchall()
            }
            if "document_id" not in cross_ref_columns:
                cur.execute("ALTER TABLE cross_references ADD COLUMN document_id TEXT")
            self.conn.commit()

    def upsert_documents(self, docs: list[RuleDocument]) -> None:
        with self._lock:
            cur = self.conn.cursor()
            for doc in docs:
                self._delete_document_rows(cur, document_id=doc.document_id)
                cur.execute(
                    """
                INSERT OR REPLACE INTO rules (
                    document_id, state, jurisdiction_type, ruleset_id, ruleset_version,
                    chapter_number, chapter_title, rule_number, rule_title, full_text,
                    anchor_id, source_file, occupancy_groups, is_generic, panchayat_category,
                    conditions_json, numeric_values_json, effective_from, effective_to
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        doc.document_id,
                        doc.state,
                        doc.jurisdiction_type,
                        doc.ruleset_id,
                        doc.ruleset_version,
                        doc.chapter_number,
                        doc.chapter_title,
                        doc.rule_number,
                        doc.rule_title,
                        doc.full_text,
                        doc.anchor_id,
                        doc.source_file,
                        json.dumps(doc.occupancy_groups),
                        1 if doc.is_generic else 0,
                        doc.panchayat_category,
                        json.dumps(doc.conditions),
                        json.dumps(doc.numeric_values),
                        doc.effective_from.isoformat() if doc.effective_from else None,
                        doc.effective_to.isoformat() if doc.effective_to else None,
                    ),
                )

                for clause in doc.clause_nodes:
                    self._upsert_clause(cur, doc.document_id, clause)
                for table in doc.tables:
                    cur.execute(
                        """
                    INSERT OR REPLACE INTO tables_data(table_id, document_id, caption, headers_json, rows_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            table.table_id,
                            doc.document_id,
                            table.caption,
                            json.dumps(table.headers),
                            json.dumps(table.rows),
                        ),
                    )
                    for row_index, row in enumerate(table.rows):
                        for col_index, cell_value in enumerate(row):
                            header = table.headers[col_index] if col_index < len(table.headers) else None
                            cur.execute(
                                """
                            INSERT INTO table_cells(table_id, document_id, row_index, col_index, header, cell_value)
                            VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (table.table_id, doc.document_id, row_index, col_index, header, cell_value),
                            )
                for ref in doc.cross_references:
                    cur.execute(
                        """
                    INSERT INTO cross_references(document_id, source_clause_id, target_ref, target_type, normalized_target_id)
                    VALUES (?, ?, ?, ?, ?)
                        """,
                        (doc.document_id, ref.source_clause_id, ref.target_ref, ref.target_type, ref.normalized_target_id),
                    )

            self.conn.commit()

    def _delete_document_rows(self, cur: sqlite3.Cursor, *, document_id: str) -> None:
        clause_rows = cur.execute(
            "SELECT clause_id FROM clauses WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        clause_ids = [str(row["clause_id"]) for row in clause_rows]
        cur.execute("DELETE FROM cross_references WHERE document_id = ?", (document_id,))
        if clause_ids:
            placeholders = ",".join(["?"] * len(clause_ids))
            cur.execute(
                f"DELETE FROM cross_references WHERE source_clause_id IN ({placeholders})",
                clause_ids,
            )
        cur.execute("DELETE FROM table_cells WHERE document_id = ?", (document_id,))
        cur.execute("DELETE FROM tables_data WHERE document_id = ?", (document_id,))
        cur.execute("DELETE FROM clauses WHERE document_id = ?", (document_id,))
        cur.execute("DELETE FROM rules WHERE document_id = ?", (document_id,))

    def _upsert_clause(self, cur: sqlite3.Cursor, document_id: str, clause: ClauseNode) -> None:
        cur.execute(
            """
            INSERT OR REPLACE INTO clauses(
                clause_id, document_id, clause_type, display_citation, sub_rule_path, anchor_id,
                normalized_text, topic_tags, occupancy_groups, is_generic, parent_clause_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clause.clause_id,
                document_id,
                clause.clause_type.value,
                clause.display_citation,
                clause.sub_rule_path,
                clause.anchor_id,
                clause.normalized_text,
                json.dumps(clause.topic_tags),
                json.dumps(clause.occupancy_groups),
                1 if clause.is_generic else 0,
                clause.parent_clause_id,
            ),
        )

    def search_rules(
        self,
        state: str,
        jurisdiction_type: str,
        ruleset_id: str | None = None,
        occupancy: list[str] | None = None,
        topic_like: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT *
            FROM rules
            WHERE state = ? AND jurisdiction_type = ?
        """
        params: list[object] = [state, jurisdiction_type]
        if ruleset_id:
            query += " AND ruleset_id = ?"
            params.append(ruleset_id)
        if topic_like:
            terms = self._topic_terms(topic_like)
            if terms:
                query += " AND (" + " OR ".join(["lower(full_text) LIKE ?"] * len(terms)) + ")"
                params.extend([f"%{term}%" for term in terms])
        if occupancy:
            occupancy_filters = " OR ".join(["occupancy_groups LIKE ?" for _ in occupancy])
            query += f" AND (is_generic = 1 OR ({occupancy_filters}))"
            params.extend([f'%"{item}"%' for item in occupancy])
        query += " ORDER BY chapter_number, rule_number LIMIT ?"
        params.append(limit)
        with self._lock:
            return self.conn.execute(query, params).fetchall()

    def get_rule_by_number(self, state: str, jurisdiction_type: str, rule_number: str) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                """
            SELECT *
            FROM rules
            WHERE state = ? AND jurisdiction_type = ? AND rule_number = ?
            ORDER BY chapter_number
            """,
                (state, jurisdiction_type, rule_number),
            ).fetchall()

    def get_cross_references(self, source_clause_ids: list[str]) -> list[sqlite3.Row]:
        if not source_clause_ids:
            return []
        placeholders = ",".join(["?"] * len(source_clause_ids))
        with self._lock:
            return self.conn.execute(
                f"SELECT * FROM cross_references WHERE source_clause_id IN ({placeholders})",
                source_clause_ids,
            ).fetchall()

    def search_table_cells(
        self,
        state: str,
        jurisdiction_type: str,
        contains: str,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                """
            SELECT c.*, r.ruleset_id, r.chapter_number, r.rule_number, r.rule_title
            FROM table_cells c
            JOIN rules r ON r.document_id = c.document_id
            WHERE r.state = ? AND r.jurisdiction_type = ? AND lower(c.cell_value) LIKE ?
            LIMIT ?
            """,
                (state, jurisdiction_type, f"%{contains.lower()}%", limit),
            ).fetchall()

    def _topic_terms(self, topic_like: str) -> list[str]:
        raw = topic_like.replace("_", " ").lower().strip()
        if not raw:
            return []
        tokens = [token for token in re.findall(r"[a-z0-9]{3,}", raw) if token]
        seen: set[str] = set()
        ordered: list[str] = []
        for token in [raw, *tokens]:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered[:8]
