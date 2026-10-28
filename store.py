"""Storage layer. Every state transition is one SQLite transaction, so a retry,
a crash, or a cancel/receipt race can never leave a half-written Task."""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time

WORKING = "TASK_STATE_WORKING"
INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
COMPLETED = "TASK_STATE_COMPLETED"
CANCELED = "TASK_STATE_CANCELED"
FAILED = "TASK_STATE_FAILED"
TERMINAL = {COMPLETED, CANCELED, FAILED, "TASK_STATE_REJECTED"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY,
  principal TEXT NOT NULL,
  batch_id TEXT NOT NULL,
  state TEXT NOT NULL,
  task_json TEXT NOT NULL,
  created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_by_principal ON tasks(principal, created);
CREATE TABLE IF NOT EXISTS messages(
  principal TEXT NOT NULL,
  message_id TEXT NOT NULL,
  hash TEXT NOT NULL,
  task_id TEXT NOT NULL,
  PRIMARY KEY(principal, message_id)
);
CREATE TABLE IF NOT EXISTS executions(
  task_id TEXT NOT NULL,
  action_id TEXT NOT NULL,
  package_id TEXT NOT NULL,
  action TEXT NOT NULL,
  receipt_nonce TEXT NOT NULL,
  created REAL NOT NULL,
  PRIMARY KEY(task_id, action_id)
);
CREATE TABLE IF NOT EXISTS decisions(
  key TEXT PRIMARY KEY,
  json TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str = "a2a_state.db"):
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)

    @contextlib.contextmanager
    def tx(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    # ---- reads -------------------------------------------------------------
    def get_message(self, principal: str, message_id: str):
        r = self.db.execute(
            "SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?",
            (principal, message_id),
        ).fetchone()
        return {"hash": r[0], "task_id": r[1]} if r else None

    def get_task(self, principal: str, task_id: str):
        """Owner-scoped: a task id that belongs to someone else looks like a missing task."""
        r = self.db.execute(
            "SELECT task_json FROM tasks WHERE id=? AND principal=?", (task_id, principal)
        ).fetchone()
        return json.loads(r[0]) if r else None

    def get_batch_id(self, principal: str, task_id: str):
        r = self.db.execute(
            "SELECT batch_id FROM tasks WHERE id=? AND principal=?", (task_id, principal)
        ).fetchone()
        return r[0] if r else None

    def list_tasks(self, principal: str):
        rows = self.db.execute(
            "SELECT task_json FROM tasks WHERE principal=? ORDER BY created DESC, rowid DESC",
            (principal,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    # ---- decision cache (keyed by canonical package content, not by ids) ---
    def get_decision(self, key: str):
        r = self.db.execute("SELECT json FROM decisions WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def put_decision(self, key: str, decision: dict):
        with self.tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO decisions(key, json) VALUES(?,?)",
                (key, json.dumps(decision, separators=(",", ":"))),
            )

    # ---- transitions -------------------------------------------------------
    def create_task(self, principal, message_id, msg_hash, batch_id, task) -> bool:
        """Message record + WORKING task are written together, before any model work."""
        try:
            with self.tx() as db:
                db.execute(
                    "INSERT INTO messages(principal, message_id, hash, task_id) VALUES(?,?,?,?)",
                    (principal, message_id, msg_hash, task["id"]),
                )
                db.execute(
                    "INSERT INTO tasks(id, principal, batch_id, state, task_json, created) VALUES(?,?,?,?,?,?)",
                    (task["id"], principal, batch_id, task["status"]["state"], json.dumps(task), time.time()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def finish_proposals(self, principal, task_id, artifact, timestamp):
        """WORKING -> INPUT_REQUIRED, only if nobody canceled meanwhile."""
        with self.tx() as db:
            r = db.execute(
                "SELECT task_json FROM tasks WHERE id=? AND principal=?", (task_id, principal)
            ).fetchone()
            if not r:
                return None
            task = json.loads(r[0])
            if task["status"]["state"] != WORKING:
                return task
            task["artifacts"] = [artifact]
            task["status"] = {"state": INPUT_REQUIRED, "timestamp": timestamp}
            db.execute(
                "UPDATE tasks SET state=?, task_json=? WHERE id=?",
                (INPUT_REQUIRED, json.dumps(task), task_id),
            )
            return task

    def fail_task(self, principal, task_id, timestamp):
        with self.tx() as db:
            r = db.execute(
                "SELECT task_json FROM tasks WHERE id=? AND principal=?", (task_id, principal)
            ).fetchone()
            if not r:
                return
            task = json.loads(r[0])
            if task["status"]["state"] != WORKING:
                return
            task["status"] = {"state": FAILED, "timestamp": timestamp}
            db.execute("UPDATE tasks SET state=?, task_json=? WHERE id=?", (FAILED, json.dumps(task), task_id))

    def complete(self, principal, task_id, message, message_id, msg_hash, receipts_artifact, executions, timestamp):
        """INPUT_REQUIRED -> COMPLETED. Message record, execution ledger and Task update
        commit together or not at all. Returns (status, task)."""
        with self.tx() as db:
            r = db.execute(
                "SELECT task_json FROM tasks WHERE id=? AND principal=?", (task_id, principal)
            ).fetchone()
            if not r:
                return "notfound", None
            task = json.loads(r[0])
            state = task["status"]["state"]
            if state in TERMINAL:
                return "terminal", task
            if state != INPUT_REQUIRED:
                return "notready", task
            db.execute(
                "INSERT INTO messages(principal, message_id, hash, task_id) VALUES(?,?,?,?)",
                (principal, message_id, msg_hash, task_id),
            )
            for ex in executions:  # PRIMARY KEY(task_id, action_id): an action can never execute twice
                db.execute(
                    "INSERT INTO executions(task_id, action_id, package_id, action, receipt_nonce, created) VALUES(?,?,?,?,?,?)",
                    (task_id, ex["actionId"], ex["packageId"], ex["action"], ex["receiptNonce"], time.time()),
                )
            task["artifacts"].append(receipts_artifact)
            task["history"].append(message)
            task["status"] = {"state": COMPLETED, "timestamp": timestamp}
            db.execute("UPDATE tasks SET state=?, task_json=? WHERE id=?", (COMPLETED, json.dumps(task), task_id))
            return "ok", task

    def cancel(self, principal, task_id, timestamp):
        with self.tx() as db:
            r = db.execute(
                "SELECT task_json FROM tasks WHERE id=? AND principal=?", (task_id, principal)
            ).fetchone()
            if not r:
                return "notfound", None
            task = json.loads(r[0])
            if task["status"]["state"] in TERMINAL:
                return "terminal", task
            task["status"] = {"state": CANCELED, "timestamp": timestamp}
            db.execute("UPDATE tasks SET state=?, task_json=? WHERE id=?", (CANCELED, json.dumps(task), task_id))
            return "ok", task
