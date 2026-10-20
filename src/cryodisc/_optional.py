"""Turn a missing optional dependency into a message that says how to get it.

`mrcfile` is an extra, not a requirement: the modeling half of this package never opens a file. Call
sites import it inside the function and translate the `ImportError` through here, so the install hint
stays in one place, in step with `pyproject.toml`'s extras name.

Imports nothing from this package, so the cheap modules do not pay to pull the expensive one in.
"""

from types import ModuleType

EXTRA = "cryodisc[io]"


def require_mrcfile(purpose: str) -> ModuleType:
    """`mrcfile`, or an `ImportError` naming both what failed and the extra that fixes it.

    Args:
        purpose: What the caller was about to do, as a gerund phrase -- "reading a map". It reads as
            the subject of the message, so it is the caller's job to say which file it wanted.

    Returns:
        The `mrcfile` module.

    Raises:
        ImportError: When the extra is not installed.
    """
    try:
        import mrcfile
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(f"{purpose} needs `mrcfile`; install {EXTRA}") from exc
    return mrcfile
