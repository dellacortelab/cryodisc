"""Read the optics a RELION 3.0-era `.star` implies, rather than carrying them as constants.

A star of that vintage has no optics table, so `import_particles` must be told the pixel size,
voltage, spherical aberration and amplitude contrast. A wrong pixel size is silent: it rescales
every resolution downstream and nothing complains, so the derivation belongs in checkable code.

The pixel size is `rlnDetectorPixelSize / rlnMagnification * 1e4` in Angstrom, the detector pitch
being in microns.
"""

from pathlib import Path

WANTED = (
    "rlnVoltage",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
    "rlnMagnification",
    "rlnDetectorPixelSize",
)


def star_columns(path: str | Path) -> dict[str, int]:
    """The zero-based column index of every `_rln` label in a star's first loop.

    Args:
        path: The star file.

    Returns:
        `{label: index}`.
    """
    columns: dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("_rln"):
                parts = stripped.split()
                columns[parts[0].lstrip("_")] = int(parts[1].lstrip("#")) - 1
            elif columns and stripped and not stripped.startswith(("data_", "loop_", "#")):
                break
    return columns


def star_optics(path: str | Path, sample: int = 2000) -> dict[str, float]:
    """The optics `import_particles` needs, read off the star and checked for consistency.

    Args:
        path: The star file.
        sample: How many data rows to check. One row would be enough for a well-formed file; the
            point of reading more is to REFUSE a file whose optics vary by row, which is two
            acquisitions concatenated and cannot be described by one set of numbers.

    Returns:
        `{"psize_A", "accel_kv", "cs_mm", "amp_contrast"}`.

    Raises:
        SystemExit: If a required column is absent, or any of them varies across the sampled rows.
    """
    columns = star_columns(path)
    missing = [name for name in WANTED if name not in columns]
    if missing:
        raise SystemExit(f"{path} has no {missing}; its optics cannot be derived and must not be guessed")

    seen: dict[str, set[str]] = {name: set() for name in WANTED}
    taken = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("_rln", "data_", "loop_", "#")):
                continue
            parts = stripped.split()
            for name in WANTED:
                index = columns[name]
                if index < len(parts):
                    seen[name].add(parts[index])
            taken += 1
            if taken >= sample:
                break
    if not taken:
        raise SystemExit(f"{path} has a header but no data rows")

    varying = {name: sorted(values)[:4] for name, values in seen.items() if len(values) != 1}
    if varying:
        raise SystemExit(
            f"{path}: these optics columns are not constant over {taken} rows: {varying}. "
            "One set of numbers cannot describe two acquisitions; split the star instead."
        )
    value = {name: float(next(iter(values))) for name, values in seen.items()}
    magnification = value["rlnMagnification"]
    if magnification <= 0:
        raise SystemExit(f"{path}: rlnMagnification is {magnification}, so the pixel size is undefined")
    return {
        # microns per detector pixel, over magnification, into Angstrom
        "psize_A": value["rlnDetectorPixelSize"] / magnification * 1e4,
        "accel_kv": value["rlnVoltage"],
        "cs_mm": value["rlnSphericalAberration"],
        "amp_contrast": value["rlnAmplitudeContrast"],
    }
