"""DeepSeek Proof of Work challenge solver.

Based on the reverse-engineered implementation from deepseek4free.
Requires wasmtime Python package to execute the DeepSeek WASM hasher.
"""

import json
import base64
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

WASM_PATH = os.path.join(os.path.dirname(__file__), "wasm", "sha3_wasm_bg.7b9ca65ddd.wasm")


class PoWSolver:
    """Solves DeepSeek's WebAssembly-based Proof-of-Work challenges."""

    def __init__(self):
        self.hasher = None

    def _init_hasher(self):
        if self.hasher is not None:
            return
        try:
            import wasmtime
            import numpy as np

            engine = wasmtime.Engine()
            with open(WASM_PATH, "rb") as f:
                wasm_bytes = f.read()
            module = wasmtime.Module(engine, wasm_bytes)
            store = wasmtime.Store(engine)
            linker = wasmtime.Linker(engine)
            linker.define_wasi()
            instance = linker.instantiate(store, module)
            memory = instance.exports(store)["memory"]

            self.hasher = {
                "store": store,
                "instance": instance,
                "memory": memory,
                "wasmtime": wasmtime,
                "np": np,
            }
            logger.info("WASM PoW hasher initialized")
        except ImportError as e:
            raise RuntimeError(
                f"PoW dependencies not installed: {e}. "
                "Install with: pip install wasmtime numpy"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WASM hasher: {e}")

    def solve_challenge(self, config: dict[str, Any]) -> str:
        """Solve a PoW challenge and return the base64-encoded response header value.

        Args:
            config: Challenge config from /api/v0/chat/create_pow_challenge response.
                    Contains: algorithm, challenge, salt, difficulty, expire_at, signature, target_path

        Returns:
            Base64-encoded JSON string to use as x-ds-pow-response header value.
        """
        self._init_hasher()
        h = self.hasher
        store = h["store"]
        instance = h["instance"]
        memory = h["memory"]
        np = h["np"]

        algorithm = config["algorithm"]
        challenge = config["challenge"]
        salt = config["salt"]
        difficulty = config["difficulty"]
        expire_at = config["expire_at"]
        signature = config["signature"]

        # Compute answer via WASM
        prefix = f"{salt}_{expire_at}_"
        answer = self._calculate_hash(
            store, instance, memory, np,
            algorithm, challenge, prefix, difficulty,
        )

        result = {
            "algorithm": algorithm,
            "challenge": challenge,
            "salt": salt,
            "answer": answer,
            "signature": signature,
            "target_path": config.get("target_path", "/api/v0/chat/completion"),
        }

        return base64.b64encode(json.dumps(result).encode()).decode()

    def _calculate_hash(
        self,
        store,
        instance,
        memory,
        np,
        algorithm: str,
        challenge: str,
        prefix: str,
        difficulty: int,
    ) -> int | None:
        """Run the WASM hash calculation."""
        try:
            challenge_ptr, challenge_len = self._write_to_memory(store, instance, memory, challenge)
            prefix_ptr, prefix_len = self._write_to_memory(store, instance, memory, prefix)

            retptr = instance.exports(store)["__wbindgen_add_to_stack_pointer"](store, -16)

            instance.exports(store)["wasm_solve"](
                store,
                retptr,
                challenge_ptr,
                challenge_len,
                prefix_ptr,
                prefix_len,
                float(difficulty),
            )

            memory_view = memory.data_ptr(store)
            status = int.from_bytes(
                bytes(memory_view[retptr : retptr + 4]), byteorder="little", signed=True
            )

            if status == 0:
                return None

            value_bytes = bytes(memory_view[retptr + 8 : retptr + 16])
            value = np.frombuffer(value_bytes, dtype=np.float64)[0]
            return int(value)

        finally:
            instance.exports(store)["__wbindgen_add_to_stack_pointer"](store, 16)

    @staticmethod
    def _write_to_memory(store, instance, memory, text: str) -> tuple[int, int]:
        """Write a string to WASM linear memory."""
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = instance.exports(store)["__wbindgen_export_0"](store, length, 1)

        memory_view = memory.data_ptr(store)
        for i, byte in enumerate(encoded):
            memory_view[ptr + i] = byte

        return ptr, length
