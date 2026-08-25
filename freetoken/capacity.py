"""Token-budget admission and memory/prefill estimation.

All constants were measured in m1_calibration (see M1_REPORT.md) and are
configurable here — none may be hardcoded elsewhere.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- Validated M1 constants (bf16 reference model, Qwen3.5-2B hybrid) -------
KV_BYTES_PER_TOKEN = 12288          # 2 * 6 full-attn layers * 2 KV heads * 256 head_dim * 2 B
CACHE_BLOCK_TOKENS = 256            # mlx-lm KVCache allocation granularity
FIXED_STATE_BYTES = 19537920        # recurrent state of the 18 linear-attention layers
SAFETY_MARGIN_BYTES = 2 * 2**30     # +2 GB estimator safety margin
MODEL_CONTEXT_LIMIT = 262144        # architectural ceiling (max_position_embeddings)
DEFAULT_CAP = 65536                 # M1-recommended default operational cap
EXTENDED_CAP = 131072               # opt-in extended cap
PREFILL_STEP_SIZE = 2048            # mlx_lm.generate chunked-prefill step (mandatory)

# Measured bf16 prefill wall time (seconds) vs prompt tokens on M4 Max, 36 GB.
BF16_PREFILL_CURVE: Tuple[Tuple[int, float], ...] = (
    (8192, 4.55),
    (32768, 23.53),
    (65536, 59.32),
    (131072, 190.47),
    (262144, 757.8),
)

# Measured 4-bit prefill wall time (M1): ~2x slower than bf16 on this hybrid
# architecture because prefill is compute-heavy and pays dequant overhead.
Q4_PREFILL_CURVE: Tuple[Tuple[int, float], ...] = (
    (8192, 9.62),
    (32768, 50.83),
    (65536, 118.3),
    (131072, 330.55),
    (262144, 804.47),
)

PREFILL_CURVES = {"bf16-m1": BF16_PREFILL_CURVE, "q4-m1": Q4_PREFILL_CURVE}


@dataclass
class CapacityConfig:
    model_context_limit: int = MODEL_CONTEXT_LIMIT
    default_cap: int = DEFAULT_CAP
    extended_cap: int = EXTENDED_CAP
    allow_extended: bool = False
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN
    cache_block_tokens: int = CACHE_BLOCK_TOKENS
    fixed_state_bytes: int = FIXED_STATE_BYTES
    safety_margin_bytes: int = SAFETY_MARGIN_BYTES
    # Measured at engine load; this is the M1 bf16 reference value.
    weights_resident_bytes: int = 3_763_700_000
    prefill_curve: Tuple[Tuple[int, float], ...] = field(default=BF16_PREFILL_CURVE)

    def validate(self) -> None:
        if not (0 < self.default_cap <= self.extended_cap <= self.model_context_limit):
            raise ValueError(
                "caps must satisfy 0 < default_cap <= extended_cap <= model_context_limit"
            )
        if self.cache_block_tokens <= 0 or self.kv_bytes_per_token <= 0:
            raise ValueError("cache constants must be positive")


def active_cap(cfg: CapacityConfig) -> int:
    return cfg.extended_cap if cfg.allow_extended else cfg.default_cap


def estimate_prefill_seconds(cfg: CapacityConfig, prompt_tokens: int) -> float:
    """Piecewise-linear interpolation of the measured prefill curve."""
    curve = cfg.prefill_curve
    if prompt_tokens <= 0:
        return 0.0
    if prompt_tokens <= curve[0][0]:
        return prompt_tokens * curve[0][1] / curve[0][0]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if prompt_tokens <= x1:
            return y0 + (y1 - y0) * (prompt_tokens - x0) / (x1 - x0)
    return curve[-1][1]  # clamp at the ceiling (admission never exceeds it)


def estimate_peak_memory_bytes(cfg: CapacityConfig, context_tokens: int) -> int:
    """Conservative peak estimate: weights + block-rounded KV + fixed state + margin.

    The margin also covers prefill activation scratch (measured overshoot was
    0.9–1.8 GB across all M1 runs).
    """
    blocks = (context_tokens + cfg.cache_block_tokens - 1) // cfg.cache_block_tokens
    kv = cfg.kv_bytes_per_token * blocks * cfg.cache_block_tokens
    return cfg.weights_resident_bytes + kv + cfg.fixed_state_bytes + cfg.safety_margin_bytes


@dataclass
class Admission:
    ok: bool
    total_tokens: int
    cap_applied: int
    mode: str = "default"               # default | extended | maximum
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warning: Optional[str] = None
    estimated_prefill_seconds: Optional[float] = None


def admit(
    cfg: CapacityConfig,
    input_tokens: int,
    max_new_tokens: int,
    allow_maximum_context: bool = False,
) -> Admission:
    """Admit or reject a request budget (input + max_new_tokens)."""
    total = input_tokens + max_new_tokens

    if total > cfg.model_context_limit:
        return Admission(
            ok=False,
            total_tokens=total,
            cap_applied=cfg.model_context_limit,
            error_code="context_limit_exceeded",
            error_message=(
                f"Request budget (input {input_tokens} + max_new_tokens {max_new_tokens} "
                f"= {total}) exceeds the model context limit of {cfg.model_context_limit}. "
                "This limit is architectural and cannot be raised."
            ),
        )

    if total <= cfg.default_cap:
        return Admission(ok=True, total_tokens=total, cap_applied=cfg.default_cap)

    est = estimate_prefill_seconds(cfg, input_tokens)
    if total <= cfg.extended_cap:
        if not cfg.allow_extended:
            return Admission(
                ok=False,
                total_tokens=total,
                cap_applied=cfg.default_cap,
                error_code="default_cap_exceeded",
                error_message=(
                    f"Request budget {total} exceeds the default cap of {cfg.default_cap}. "
                    f"Extended mode (up to {cfg.extended_cap}) is disabled; start the server "
                    "with --allow-extended to enable it."
                ),
            )
        return Admission(
            ok=True,
            total_tokens=total,
            cap_applied=cfg.extended_cap,
            mode="extended",
            warning=(
                f"Extended context mode: request budget {total} is above the default cap "
                f"{cfg.default_cap}; estimated prefill time ~{est:.0f}s on the reference "
                "M4 Max before the first token."
            ),
            estimated_prefill_seconds=est,
        )

    # extended_cap < total <= model_context_limit → maximum mode
    if not cfg.allow_extended:
        return Admission(
            ok=False,
            total_tokens=total,
            cap_applied=cfg.default_cap,
            error_code="default_cap_exceeded",
            error_message=(
                f"Request budget {total} exceeds the default cap of {cfg.default_cap}. "
                "Extended/maximum modes are disabled; start the server with --allow-extended."
            ),
        )
    if not allow_maximum_context:
        return Admission(
            ok=False,
            total_tokens=total,
            cap_applied=cfg.extended_cap,
            error_code="extended_cap_exceeded",
            error_message=(
                f"Request budget {total} exceeds the extended cap of {cfg.extended_cap}. "
                f'Maximum mode (up to {cfg.model_context_limit}) requires the request field '
                '"allow_maximum_context": true and a long prefill.'
            ),
        )
    return Admission(
        ok=True,
        total_tokens=total,
        cap_applied=cfg.model_context_limit,
        mode="maximum",
        warning=(
            f"Maximum context mode: request budget {total} is above the extended cap "
            f"{cfg.extended_cap}; estimated prefill time ~{est:.0f}s on the reference "
            "M4 Max. Generation at this depth is slow and memory-heavy."
        ),
        estimated_prefill_seconds=est,
    )
