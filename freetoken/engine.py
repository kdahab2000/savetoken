"""MLX-LM engine wrapper: load, tokenize, stream, cancel.

Prefill is always the chunked mlx-lm path (prefill_step_size=2048); single-pass
prefill is never used because it hard-fails above ~37k tokens on Metal
(M1_REPORT.md, "Failure behavior").
"""

import threading
import time
from typing import Iterator, List, Optional

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate

from .capacity import CapacityConfig, PREFILL_STEP_SIZE


class GenerationCancelled(Exception):
    pass


class Engine:
    def __init__(self, model_path: str, capacity: Optional[CapacityConfig] = None):
        self.model_path = model_path
        self.capacity = capacity or CapacityConfig()
        self.capacity.validate()
        self.prefill_step_size = PREFILL_STEP_SIZE
        self.model = None
        self.tokenizer = None
        self.stats = {
            "generations_started": 0,
            "generations_cancelled": 0,
            "tokens_generated": 0,
        }
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.model_path.rstrip("/").split("/")[-1]

    def load(self, measure: str = "absolute") -> None:
        """measure='absolute' reads total active Metal memory (fresh process);
        measure='delta' attributes only this load's increase (used when another
        model is still resident during an explicit switch)."""
        before = mx.get_active_memory() if measure == "delta" else 0
        t0 = time.time()
        self.model, self.tokenizer = mlx_load(self.model_path)
        mx.eval(self.model.parameters())
        # Measure true resident weight memory instead of trusting the default.
        resident = int(mx.get_active_memory())
        if measure == "delta":
            resident = max(0, resident - int(before))
        self.capacity.weights_resident_bytes = resident
        self.load_seconds = round(time.time() - t0, 2)

    def _require_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("engine not loaded; call load() first")

    def tokenize_chat(self, messages: List[dict]) -> List[int]:
        self._require_loaded()
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )

    def count_chat_tokens(self, messages: List[dict]) -> int:
        return len(self.tokenize_chat(messages))

    def stream_chat(
        self,
        prompt_ids: List[int],
        max_tokens: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield incremental text segments (one per generated token; a segment
        may be an empty string while the detokenizer buffers).

        Raises GenerationCancelled if cancel_event is set; the underlying
        generator is closed immediately so no further model work happens.
        """
        self._require_loaded()
        with self._lock:
            self.stats["generations_started"] += 1
        prompt = mx.array(prompt_ids)
        gen = stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            prefill_step_size=self.prefill_step_size,
        )
        try:
            for resp in gen:
                if cancel_event is not None and cancel_event.is_set():
                    with self._lock:
                        self.stats["generations_cancelled"] += 1
                    raise GenerationCancelled()
                with self._lock:
                    self.stats["tokens_generated"] += 1
                yield resp.text
        except GeneratorExit:
            # consumer abandoned the stream (e.g. HTTP client disconnect)
            with self._lock:
                self.stats["generations_cancelled"] += 1
            raise
        finally:
            gen.close()
