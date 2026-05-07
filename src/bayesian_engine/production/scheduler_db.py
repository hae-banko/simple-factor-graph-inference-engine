"""SQLite persistence layer for the Bayesian scheduler."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Module-level constants and helpers
# ---------------------------------------------------------------------------

STATE_DIR = Path.home() / ".hermes" / "bayesian_scheduler"
DEFAULT_DB_NAME = "scheduler.db"


def get_state_dir() -> Path:
    """Return the state directory, creating it if it does not exist."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


# ---------------------------------------------------------------------------
# Dataclasses for row results
# ---------------------------------------------------------------------------

@dataclass
class InferenceRecord:
    """Represents a row from the inferences table."""

    id: int
    run_at: str
    model_name: str
    target: str
    evidence: dict[str, Any]
    action: str
    reason: str
    confidence: float
    trace: dict[str, Any]
    latency_ms: float | None
    cron_job_id: str | None
    session_id: str | None


@dataclass
class OutcomeRecord:
    """Represents a row from the outcomes table."""

    id: int
    inferred_at: str
    observed_at: str
    task_id: str
    action: str
    success: bool | None
    latency_ms: float | None
    error: str | None
    context: dict[str, Any] | None


@dataclass
class PosteriorRecord:
    """Represents a row from the posteriors table."""

    id: int
    snapshot_at: str
    model_name: str
    variable_name: str
    value: str
    probability: float
    evidence_count: int


# ---------------------------------------------------------------------------
# SchedulerDB
# ---------------------------------------------------------------------------

class SchedulerDB:
    """SQLite persistence layer for the Bayesian scheduler.

    The database is stored under ``~/.hermes/bayesian_scheduler/`` by default.
    WAL mode is enabled for better concurrency.

    Args:
        db_path: Optional path to the database file. If ``None``, the default
            ``~/.hermes/bayesian_scheduler/scheduler.db`` is used.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = get_state_dir() / DEFAULT_DB_NAME
        else:
            # Caller is responsible for ensuring parent dirs exist.
            db_path = Path(db_path)

        self._db_path = db_path
        self._connection: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Return a live database connection.

        Returns:
            A ``sqlite3.Connection`` configured with WAL mode.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(str(self._db_path))
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL;")
            self._connection.execute("PRAGMA synchronous=NORMAL;")
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and set schema version if this is a fresh database."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # All tables use IF NOT EXISTS so migrations are additive only.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inferences (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at      TEXT    NOT NULL,
                model_name  TEXT    NOT NULL,
                target      TEXT    NOT NULL,
                evidence    TEXT    NOT NULL,
                action      TEXT    NOT NULL,
                reason      TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                trace       TEXT    NOT NULL,
                latency_ms  REAL,
                cron_job_id TEXT,
                session_id  TEXT
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                inferred_at TEXT    NOT NULL,
                observed_at TEXT    NOT NULL,
                task_id     TEXT    NOT NULL,
                action      TEXT    NOT NULL,
                success     INTEGER,
                latency_ms  REAL,
                error       TEXT,
                context     TEXT
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS posteriors (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_at     TEXT    NOT NULL,
                model_name      TEXT    NOT NULL,
                variable_name   TEXT    NOT NULL,
                value           TEXT    NOT NULL,
                probability     REAL    NOT NULL,
                evidence_count  INTEGER DEFAULT 0
            );
            """
        )

        # Set schema version only if not already present.
        cursor.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('version', '1');")
        conn.commit()

    # ------------------------------------------------------------------
    # Inferences
    # ------------------------------------------------------------------

    def insert_inference(
        self,
        run_at: str,
        model_name: str,
        target: str,
        evidence: dict[str, Any],
        action: str,
        reason: str,
        confidence: float,
        trace: dict[str, Any],
        latency_ms: float | None = None,
        cron_job_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Insert an inference record.

        Args:
            run_at: ISO-8601 timestamp when the inference was run.
            model_name: Name of the model used for inference.
            target: Target variable name.
            evidence: Evidence dictionary (serialised to JSON).
            action: Selected action string.
            reason: Human-readable reason for the decision.
            confidence: Confidence score in [0, 1].
            trace: Trace dictionary (serialised to JSON).
            latency_ms: Optional inference latency in milliseconds.
            cron_job_id: Optional identifier of the originating cron job.
            session_id: Optional session identifier.

        Returns:
            The integer rowid of the inserted record.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO inferences
                (run_at, model_name, target, evidence, action, reason,
                 confidence, trace, latency_ms, cron_job_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run_at,
                model_name,
                target,
                json.dumps(evidence),
                action,
                reason,
                confidence,
                json.dumps(trace),
                latency_ms,
                cron_job_id,
                session_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_inferences(
        self,
        model_name: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[InferenceRecord]:
        """Retrieve inference records.

        Args:
            model_name: If provided, filter by model name.
            since: If provided, only return records with ``run_at >= since``.
            limit: Maximum number of rows to return (default 100).

        Returns:
            A list of ``InferenceRecord`` objects.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM inferences WHERE 1=1"
        params: list[Any] = []

        if model_name is not None:
            query += " AND model_name = ?"
            params.append(model_name)
        if since is not None:
            query += " AND run_at >= ?"
            params.append(since)

        query += " ORDER BY run_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        records: list[InferenceRecord] = []
        for row in rows:
            records.append(
                InferenceRecord(
                    id=row["id"],
                    run_at=row["run_at"],
                    model_name=row["model_name"],
                    target=row["target"],
                    evidence=json.loads(row["evidence"]),
                    action=row["action"],
                    reason=row["reason"],
                    confidence=row["confidence"],
                    trace=json.loads(row["trace"]),
                    latency_ms=row["latency_ms"],
                    cron_job_id=row["cron_job_id"],
                    session_id=row["session_id"],
                )
            )
        return records

    # ------------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------------

    def insert_outcome(
        self,
        inferred_at: str,
        observed_at: str,
        task_id: str,
        action: str,
        success: bool | None,
        latency_ms: float | None = None,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> int:
        """Insert an outcome record.

        Args:
            inferred_at: ISO-8601 timestamp of the corresponding inference.
            observed_at: ISO-8601 timestamp when the outcome was observed.
            task_id: Identifier of the task associated with this outcome.
            action: Action that was executed.
            success: ``True`` if successful, ``False`` if failed,
                ``None`` if unknown.
            latency_ms: Optional execution latency in milliseconds.
            error: Optional error message string.
            context: Optional context dictionary (serialised to JSON).

        Returns:
            The integer rowid of the inserted record.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Convert bool to integer for SQLite storage.
        success_int: int | None = None if success is None else (1 if success else 0)

        cursor.execute(
            """
            INSERT INTO outcomes
                (inferred_at, observed_at, task_id, action, success,
                 latency_ms, error, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                inferred_at,
                observed_at,
                task_id,
                action,
                success_int,
                latency_ms,
                error,
                json.dumps(context) if context is not None else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_outcomes(
        self,
        task_id: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[OutcomeRecord]:
        """Retrieve outcome records.

        Args:
            task_id: If provided, filter by task identifier.
            since: If provided, only return records with ``observed_at >= since``.
            limit: Maximum number of rows to return (default 100).

        Returns:
            A list of ``OutcomeRecord`` objects.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM outcomes WHERE 1=1"
        params: list[Any] = []

        if task_id is not None:
            query += " AND task_id = ?"
            params.append(task_id)
        if since is not None:
            query += " AND observed_at >= ?"
            params.append(since)

        query += " ORDER BY observed_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        records: list[OutcomeRecord] = []
        for row in rows:
            success_int = row["success"]
            success: bool | None = None if success_int is None else bool(success_int)
            records.append(
                OutcomeRecord(
                    id=row["id"],
                    inferred_at=row["inferred_at"],
                    observed_at=row["observed_at"],
                    task_id=row["task_id"],
                    action=row["action"],
                    success=success,
                    latency_ms=row["latency_ms"],
                    error=row["error"],
                    context=json.loads(row["context"]) if row["context"] else None,
                )
            )
        return records

    # ------------------------------------------------------------------
    # Posteriors
    # ------------------------------------------------------------------

    def insert_posterior(
        self,
        snapshot_at: str,
        model_name: str,
        variable_name: str,
        value: str,
        probability: float,
        evidence_count: int = 0,
    ) -> int:
        """Insert a posterior probability snapshot.

        Args:
            snapshot_at: ISO-8601 timestamp of the snapshot.
            model_name: Name of the model this snapshot belongs to.
            variable_name: Name of the variable.
            value: Discrete value of the variable.
            probability: Probability of ``value``.
            evidence_count: Number of evidence points used (default 0).

        Returns:
            The integer rowid of the inserted record.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO posteriors
                (snapshot_at, model_name, variable_name, value, probability, evidence_count)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (snapshot_at, model_name, variable_name, value, probability, evidence_count),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_posteriors(
        self,
        model_name: str,
        variable_name: str | None = None,
        limit: int = 50,
    ) -> list[PosteriorRecord]:
        """Retrieve posterior probability snapshots.

        Args:
            model_name: Filter by model name (required).
            variable_name: If provided, further filter by variable name.
            limit: Maximum number of rows to return (default 50).

        Returns:
            A list of ``PosteriorRecord`` objects.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM posteriors WHERE model_name = ?"
        params: list[Any] = [model_name]

        if variable_name is not None:
            query += " AND variable_name = ?"
            params.append(variable_name)

        query += " ORDER BY snapshot_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        records: list[PosteriorRecord] = []
        for row in rows:
            records.append(
                PosteriorRecord(
                    id=row["id"],
                    snapshot_at=row["snapshot_at"],
                    model_name=row["model_name"],
                    variable_name=row["variable_name"],
                    value=row["value"],
                    probability=row["probability"],
                    evidence_count=row["evidence_count"],
                )
            )
        return records