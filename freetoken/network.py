"""Network-boundary policy shared by the optional MLX server and tests.

This module deliberately has no MLX imports so loopback policy can be audited
on clean machines that do not have the optional inference runtime installed.
"""

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_host(host: str) -> str:
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to bind non-loopback host {host!r}; this server is "
            "localhost-only by design"
        )
    return host
