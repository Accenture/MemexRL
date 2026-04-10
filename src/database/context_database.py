"""
Database for storing compressed context with lossless retrieval.

Supports multiple backends:
- memory: In-memory dict (for development/testing)
- sqlite: SQLite database (for single-machine persistence)
- redis: Redis (for distributed systems)
"""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ContextDatabase(ABC):
    """Abstract base class for context storage."""

    @abstractmethod
    def store(self, key: str, value: dict) -> None:
        """Store a value with the given key."""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> dict:
        """Retrieve a value by key. Raises KeyError if not found."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key-value pair."""
        pass

    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        """List all keys, optionally filtered by prefix."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass


class MemoryContextDatabase(ContextDatabase):
    """In-memory implementation for development."""

    def __init__(self):
        self._store = {}
        logger.info("[ContextDB] Using in-memory storage")

    def store(self, key: str, value: dict) -> None:
        self._store[key] = value
        logger.debug(f"[ContextDB] Stored key: {key}")

    def retrieve(self, key: str) -> dict:
        if key not in self._store:
            raise KeyError(f"Key not found: {key}")
        return self._store[key]

    def delete(self, key: str) -> None:
        if key in self._store:
            del self._store[key]
            logger.debug(f"[ContextDB] Deleted key: {key}")

    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            return list(self._store.keys())
        return [k for k in self._store.keys() if k.startswith(prefix)]

    def clear(self) -> None:
        count = len(self._store)
        self._store.clear()
        logger.info(f"[ContextDB] Cleared {count} entries")

    def get_stats(self) -> dict:
        """Get storage statistics."""
        total_size = sum(len(json.dumps(v)) for v in self._store.values())
        return {
            "backend": "memory",
            "entry_count": len(self._store),
            "total_size_bytes": total_size,
        }


class SQLiteContextDatabase(ContextDatabase):
    """SQLite implementation for single-machine persistence."""

    def __init__(self, db_path: str = "context_db.sqlite"):
        import sqlite3

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_table()
        logger.info(f"[ContextDB] Using SQLite storage at {self.db_path}")

    def _create_table(self):
        """Create the context table if it doesn't exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_store (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                created_at REAL NOT NULL
            )
        """
        )
        self.conn.commit()

    def store(self, key: str, value: dict) -> None:
        import time

        serialized = pickle.dumps(value)
        self.conn.execute(
            "INSERT OR REPLACE INTO context_store (key, value, created_at) VALUES (?, ?, ?)",
            (key, serialized, time.time()),
        )
        self.conn.commit()
        logger.debug(f"[ContextDB] Stored key: {key}")

    def retrieve(self, key: str) -> dict:
        cursor = self.conn.execute("SELECT value FROM context_store WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Key not found: {key}")
        return pickle.loads(row[0])

    def delete(self, key: str) -> None:
        self.conn.execute("DELETE FROM context_store WHERE key = ?", (key,))
        self.conn.commit()
        logger.debug(f"[ContextDB] Deleted key: {key}")

    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            cursor = self.conn.execute("SELECT key FROM context_store")
        else:
            cursor = self.conn.execute("SELECT key FROM context_store WHERE key LIKE ?", (f"{prefix}%",))
        return [row[0] for row in cursor.fetchall()]

    def clear(self) -> None:
        cursor = self.conn.execute("SELECT COUNT(*) FROM context_store")
        count = cursor.fetchone()[0]
        self.conn.execute("DELETE FROM context_store")
        self.conn.commit()
        logger.info(f"[ContextDB] Cleared {count} entries")

    def get_stats(self) -> dict:
        """Get storage statistics."""
        cursor = self.conn.execute("SELECT COUNT(*), SUM(LENGTH(value)) FROM context_store")
        count, total_size = cursor.fetchone()
        return {
            "backend": "sqlite",
            "db_path": str(self.db_path),
            "entry_count": count or 0,
            "total_size_bytes": total_size or 0,
        }

    def __del__(self):
        if hasattr(self, "conn"):
            self.conn.close()


class RedisContextDatabase(ContextDatabase):
    """Redis implementation for distributed systems."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, prefix: str = "ctx:"):
        try:
            import redis
        except ImportError:
            raise ImportError("redis package required. Install with: pip install redis")

        self.prefix = prefix
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)

        # Test connection
        self.client.ping()
        logger.info(f"[ContextDB] Using Redis storage at {host}:{port}")

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    def store(self, key: str, value: dict) -> None:
        full_key = self._make_key(key)
        serialized = pickle.dumps(value)
        self.client.set(full_key, serialized)
        logger.debug(f"[ContextDB] Stored key: {key}")

    def retrieve(self, key: str) -> dict:
        full_key = self._make_key(key)
        serialized = self.client.get(full_key)
        if serialized is None:
            raise KeyError(f"Key not found: {key}")
        return pickle.loads(serialized)

    def delete(self, key: str) -> None:
        full_key = self._make_key(key)
        self.client.delete(full_key)
        logger.debug(f"[ContextDB] Deleted key: {key}")

    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            pattern = f"{self.prefix}*"
        else:
            pattern = f"{self.prefix}{prefix}*"

        keys = self.client.keys(pattern)
        # Remove prefix from keys
        return [k.decode().replace(self.prefix, "", 1) for k in keys]

    def clear(self) -> None:
        keys = self.client.keys(f"{self.prefix}*")
        if keys:
            self.client.delete(*keys)
            logger.info(f"[ContextDB] Cleared {len(keys)} entries")

    def get_stats(self) -> dict:
        """Get storage statistics."""
        keys = self.client.keys(f"{self.prefix}*")
        total_size = sum(self.client.memory_usage(k) or 0 for k in keys)
        return {
            "backend": "redis",
            "entry_count": len(keys),
            "total_size_bytes": total_size,
        }


def create_context_database(backend: str = "memory", **kwargs) -> ContextDatabase:
    """
    Factory function to create a context database.

    Args:
        backend: "memory", "sqlite", or "redis"
        **kwargs: Backend-specific arguments
            - For sqlite: db_path
            - For redis: host, port, db, prefix

    Returns:
        ContextDatabase instance
    """
    if backend == "memory":
        return MemoryContextDatabase()
    elif backend == "sqlite":
        db_path = kwargs.get("db_path", "context_db.sqlite")
        return SQLiteContextDatabase(db_path=db_path)
    elif backend == "redis":
        return RedisContextDatabase(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 6379),
            db=kwargs.get("db", 0),
            prefix=kwargs.get("prefix", "ctx:"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: memory, sqlite, redis")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test in-memory
    print("\n=== Testing Memory Backend ===")
    db = create_context_database("memory")
    db.store("test1", {"content": "Hello world", "count": 42})
    print(f"Retrieved: {db.retrieve('test1')}")
    print(f"Keys: {db.list_keys()}")
    print(f"Stats: {db.get_stats()}")

    # Test SQLite
    print("\n=== Testing SQLite Backend ===")
    db = create_context_database("sqlite", db_path="test_context.db")
    db.store("test2", {"content": "Hello SQLite", "data": [1, 2, 3]})
    print(f"Retrieved: {db.retrieve('test2')}")
    print(f"Stats: {db.get_stats()}")
    db.clear()

    print("\n✅ All tests passed!")
