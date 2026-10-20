"""Fetch a two-state slice of CryoBench and hand it to `realdata/score_pool.py`.
CryoBench (Jeon et al., arXiv:2408.05526) ships simulated particle sets with per-particle ground truth,
including the state each was generated from, so a recovered fraction can be tested against the physics.
    .venv/bin/python reproduction/experiments/preprocess/cryobench.py list --dataset IgG-1D
    .venv/bin/python .../cryobench.py fetch --dataset IgG-1D --states 000 050 --dest ~/scratch/cryobench --out results/cb_fetch.json
Only the wanted members download: a zip's directory is at the end, so HTTP range requests stream one
member without the archive landing on disk (`urllib` honors `http_proxy`). The state label IS the file
name, so `p_true` is exact. `list` reports the route: conformational datasets ship maps and structures,
compositional sets differ by whole subunits and need the map route.
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

from reproduction.record import write_result

DRIVER = "reproduction/experiments/preprocess/cryobench.py"

RECORDS = {
    "IgG-1D": ("11629428", "IgG-1D.zip"),
    "IgG-RL": ("11629428", "IgG-RL.zip"),
    "Ribosembly": ("12528292", "Ribosembly.zip"),
    "Tomotwin-100": ("12528292", "Tomotwin-100.zip"),
    "Spike-MD": ("12528784", "Spike-MD.zip"),
}
"""Zenodo record and archive per dataset, from the paper's data-availability section."""

CONFORMATIONAL = {"IgG-1D", "IgG-RL", "Spike-MD"}
"""Datasets whose states are one molecule in different shapes, and which ship volumes only."""


class HttpFile:
    """A read-only seekable file over HTTP range requests, so `zipfile` can stream one member.

    Zenodo serves byte ranges, and a zip's central directory is at the END of the archive, so
    `zipfile` reading through this makes exactly the requests it needs: the directory, then the
    member. Nothing else is transferred.
    """

    def __init__(self, url: str, timeout: float = 300.0):
        self._timeout = timeout
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as response:
            self._url = response.geturl()  # follow the redirect ONCE, then range against the target
            self._size = int(response.headers["Content-Length"])
        self._offset = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        self._offset = {0: offset, 1: self._offset + offset, 2: self._size + offset}[whence]
        return self._offset

    def tell(self) -> int:
        return self._offset

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._size - self._offset
        if size <= 0:
            return b""
        end = min(self._offset + size, self._size) - 1
        wanted = end - self._offset + 1
        request = urllib.request.Request(self._url, headers={"Range": f"bytes={self._offset}-{end}"})
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            status = response.status
            data = response.read()
        # a server or proxy that IGNORES Range answers 200 with the whole body -- wrong bytes returned
        # as right and the archive buffered in memory; checked because the required `http_proxy` can do this
        if status != 206 or len(data) != wanted:
            raise OSError(
                f"a range request for {wanted} bytes at offset {self._offset} of {self._url} came back "
                f"with status {status} and {len(data)} bytes. 206 and an exact length are what make this "
                "file seekable; a 200 with the whole body is a server or proxy ignoring Range, which "
                "cannot be worked around here -- download the archive whole, or fetch through something "
                "that passes Range untouched."
            )
        self._offset += len(data)
        return data


def archive_url(dataset: str) -> str:
    """The Zenodo content URL for a dataset's archive.

    Args:
        dataset: One of `RECORDS`.

    Returns:
        The URL.

    Raises:
        KeyError: If the dataset is not one CryoBench publishes.
    """
    if dataset not in RECORDS:
        raise KeyError(f"{dataset!r} is not a CryoBench dataset; it has {sorted(RECORDS)}")
    record, archive = RECORDS[dataset]
    return f"https://zenodo.org/api/records/{record}/files/{archive}/content"


def wanted_members(names: list[str], states: list[str], snr: str | None) -> list[str]:
    """The archive members needed to score the given states, and nothing else.

    Args:
        names: Every member name in the archive.
        states: State prefixes, e.g. `["000", "050"]`.
        snr: Which noise level's image directory to take, for datasets that ship several. `None`
            takes whichever single one exists.

    Returns:
        Member names: each state's stack, each state's volume, and the star that indexes them.

    Raises:
        ValueError: If a state matches no stack, or if the noise level is ambiguous.
    """
    stacks = [n for n in names if n.endswith(".mrcs")]
    if snr is not None:
        stacks = [n for n in stacks if f"snr{snr}/" in n]
    levels = {part for n in stacks for part in n.split("/") if part.startswith("snr")}
    if len(levels) > 1:
        raise ValueError(f"several noise levels {sorted(levels)}; name one with --snr")

    chosen: list[str] = []
    for state in states:
        matching = [n for n in stacks if Path(n).name.startswith(f"{state}_")]
        if not matching:
            raise ValueError(f"state {state!r} matches no stack; the archive has {len(stacks)} of them")
        chosen.extend(matching)

    directories = {str(Path(n).parent) for n in chosen}
    chosen += [n for n in names if n.endswith(".star") and str(Path(n).parent) in directories]
    # the ground-truth volume for each state, which is what a conformational set scores against
    chosen += [n for n in names if "/vols/" in n and Path(n).stem.split("_")[0] in states]
    chosen += [n for n in names if "/pdbs/" in n and Path(n).stem.split("_")[0] in states]
    return sorted(set(chosen))


def run_list(args: argparse.Namespace) -> int:
    """Print a dataset's states and what route it can be scored through."""
    with zipfile.ZipFile(HttpFile(archive_url(args.dataset))) as archive:
        names = archive.namelist()
        sizes = {info.filename: info.file_size for info in archive.infolist()}

    stacks = sorted(n for n in names if n.endswith(".mrcs"))
    has_pdbs = any("/pdbs/" in n for n in names)
    has_vols = any("/vols/" in n for n in names)
    levels = sorted({p for n in stacks for p in n.split("/") if p.startswith("snr")})

    print(f"{args.dataset}: {len(stacks)} stacks, volumes={has_vols}, pdbs={has_pdbs}")
    print(f"  noise levels: {levels or ['(one, unnamed)']}")
    print(f"  route: {'--map-a/--map-b (no atoms shipped)' if not has_pdbs else '--structure-a/-b or maps'}")
    if args.dataset not in CONFORMATIONAL:
        print("  NOTE compositional: states differ by whole subunits, so align_conformers will refuse them")
    for name in stacks[:20]:
        print(f"    {sizes.get(name, 0) / 1e6:9.1f} MB  {name}")
    if len(stacks) > 20:
        print(f"    ... and {len(stacks) - 20} more")
    return 0


def run_fetch(args: argparse.Namespace) -> int:
    """Download the members for the requested states."""
    # expanduser BEFORE anything is created: a quoted or job-submitted `--dest ~/...` arrives literal
    # and would otherwise fill a directory named `~` under the working directory, silently
    args.dest = args.dest.expanduser()
    args.dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(HttpFile(archive_url(args.dataset))) as archive:
        members = wanted_members(archive.namelist(), args.states, args.snr)
        written = []
        for name in members:
            target = args.dest / name
            if target.exists() and not args.overwrite:
                print(f"have  {name}")
                written.append({"member": name, "path": str(target), "bytes": target.stat().st_size})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"fetch {name}")
            with archive.open(name) as source, open(target, "wb") as sink:
                while chunk := source.read(1 << 22):
                    sink.write(chunk)
            written.append({"member": name, "path": str(target), "bytes": target.stat().st_size})

    total = sum(item["bytes"] for item in written)
    print(f"\n{len(written)} members, {total / 1e6:.1f} MB -> {args.dest}")
    write_result(
        args.out,
        {"dataset": args.dataset, "states": args.states, "members": written, "total_bytes": total},
        driver=DRIVER,
        arguments=vars(args),
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    lister = sub.add_parser("list", help="print a dataset's states without downloading it")
    lister.add_argument("--dataset", required=True, choices=sorted(RECORDS))

    fetch = sub.add_parser("fetch", help="download only the members two states need")
    fetch.add_argument("--dataset", required=True, choices=sorted(RECORDS))
    fetch.add_argument("--states", nargs="+", required=True, metavar="STATE", help="state prefixes, e.g. 000 050")
    fetch.add_argument("--dest", type=Path, required=True, help="extraction root; keeps the archive's own layout")
    fetch.add_argument("--snr", default=None, help="noise level for datasets shipping several, e.g. 0.01")
    fetch.add_argument("--overwrite", action="store_true", help="re-download members already on disk")
    fetch.add_argument("--out", type=Path, default=Path("results/cryobench_fetch.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {"list": run_list, "fetch": run_fetch}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
