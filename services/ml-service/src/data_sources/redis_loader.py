import json
from typing import Any

import pandas as pd
import redis


class RedisLoader:
    """
    Universal Redis Loader.
    Supports reading Strings, Lists, Sets, Hashes, and Sorted Sets.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        decode_responses: bool = True,
    ):
        """
        Initialize Redis connection.
        """
        self.client = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=decode_responses)

    def test_connection(self) -> bool:
        """Verify connection is alive."""
        try:
            return self.client.ping()
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def scan_keys(self, pattern: str = "*", count: int = 1000) -> list[str]:
        """
        Scan keys matching a pattern.
        Using SCAN instead of KEYS for performance safety.
        """
        keys = []
        cursor = "0"
        while cursor != 0:
            cursor, batch = self.client.scan(cursor=cursor, match=pattern, count=count)
            keys.extend(batch)
            if cursor == 0:
                break
        return keys

    def get_key_type(self, key: str) -> str:
        """Get the type of a key."""
        return self.client.type(key)

    def load_key(self, key: str) -> Any:
        """
        Load value of a single key based on its type.
        """
        k_type = self.get_key_type(key)

        if k_type == "string":
            val = self.client.get(key)
            # Try to parse JSON if possible
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return val

        elif k_type == "list":
            return self.client.lrange(key, 0, -1)

        elif k_type == "set":
            return list(self.client.smembers(key))

        elif k_type == "zset":
            return self.client.zrange(key, 0, -1, withscores=True)

        elif k_type == "hash":
            return self.client.hgetall(key)

        return None

    def load_keys_to_df(self, pattern: str = "*") -> pd.DataFrame:
        """
        Load multiple keys matching a pattern into a DataFrame.
        Best for keys that share a structure (e.g., user:1, user:2).
        """
        keys = self.scan_keys(pattern)
        data = []

        for key in keys:
            val = self.load_key(key)

            # If value is a dict (Hash or JSON), flatten it
            if isinstance(val, dict):
                row = {"_key": key, **val}
            # If list/set, store as object
            elif isinstance(val, (list, tuple)):
                row = {"_key": key, "value": val}
            # Primitive
            else:
                row = {"_key": key, "value": val}

            data.append(row)

        return pd.DataFrame(data)

    def estimate_count(self) -> int:
        """Get total number of keys."""
        return self.client.dbsize()
