"""Published population splits, and what the counting floor says about their precision.
EMDB records `number_images_used` for essentially every entry, so a publication depositing several maps
of one specimen has published an occupancy split -- almost always with no error bar.
    .venv/bin/python .../emdb_occupancies.py --emdb EMD-XXXX EMD-YYYY --out results/emdb_occupancies.json
By `sigma_p^2 = [p(1-p) + v/Delta^2]/N`, a published split of N particles is pinned to at least
`sqrt(p(1-p)/N)` by counting alone -- usually tiny, so the real question is whether the classes are two
states of one molecule, which no particle count addresses. `required_discriminability` inverts the
relation to say how bad the per-particle separation must be for counting to matter. Maps sharing a
particle count are collapsed as one state (`n_collapsed`), and a group collapsing to one is dropped.
"""

import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from reproduction.record import write_result

_TARGET = 0.05

DRIVER = "reproduction/experiments/preprocess/emdb_occupancies.py"
API = "https://www.ebi.ac.uk/emdb/api"


def fetch(url: str, timeout: float = 45.0):
    """One JSON GET against EMDB.

    Args:
        url: Absolute URL.
        timeout: Seconds.

    Returns:
        The decoded body.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def entry_summary(entry: dict) -> dict:
    """The four fields E8 needs out of an EMDB entry document.

    The DOI is under `citation_type.external_references`, as a list of typed strings prefixed `doi:` --
    NOT a `doi` field, which reads as `None` and silently groups every entry under one key.

    Args:
        entry: A full entry document from the search or entry endpoint.

    Returns:
        `emdb_id`, `title`, `n_images` and `doi`, any of which may be `None`.
    """
    determination = (entry.get("structure_determination_list") or {}).get("structure_determination") or [{}]
    processing = (determination[0].get("image_processing") or [{}])[0]
    reconstruction = processing.get("final_reconstruction") or {}
    resolution = (reconstruction.get("resolution") or {}).get("valueOf_")

    citation = ((entry.get("crossreferences") or {}).get("citation_list") or {}).get("primary_citation") or {}
    doi = None
    for reference in (citation.get("citation_type") or {}).get("external_references") or []:
        if reference.get("type_") == "DOI":
            doi = str(reference.get("valueOf_", "")).removeprefix("doi:") or None
    return {
        "emdb_id": entry.get("emdb_id"),
        "title": (entry.get("admin") or {}).get("title"),
        "n_images": reconstruction.get("number_images_used"),
        "resolution": float(resolution) if resolution else None,
        "doi": doi,
    }


def group_by_doi(summaries: list[dict]) -> dict[str, list[dict]]:
    """Entries grouped by primary-citation DOI, dropping those with no DOI or no particle count.

    Args:
        summaries: From `entry_summary`.

    Returns:
        DOI -> entries, each group sorted by descending particle count.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in summaries:
        if item["doi"] and item["n_images"]:
            grouped[item["doi"]].append(item)
    return {doi: sorted(rows, key=lambda r: -r["n_images"]) for doi, rows in grouped.items()}


def occupancies(entries: list[dict]) -> dict | None:
    """The occupancy split a group of maps implies, with the counting floor on each fraction.

    Args:
        entries: Maps from one publication, each carrying `n_images`.

    Returns:
        `None` when fewer than two DISTINCT particle counts survive -- that is one state refined several
        ways, not a split. Otherwise the fractions, the counting floor `sqrt(p(1-p)/N)` on each, the
        maps each state was built from, the total, and how many maps were collapsed as duplicates of
        a count.
    """
    # each state carries the maps it was built from, not lined up by position: `counts` is de-duplicated
    # and `entries` is not, so a positional pairing would misattribute accessions past the first duplicate
    by_count: dict[int, list[str]] = defaultdict(list)
    for entry in entries:
        by_count[int(entry["n_images"])].append(str(entry.get("emdb_id") or "?"))
    counts = sorted(by_count, reverse=True)
    collapsed = len(entries) - len(counts)
    if len(counts) < 2:
        return None
    total = sum(counts)
    rows = []
    for count in counts:
        fraction = count / total
        rows.append(
            {
                "n_images": count,
                "emdb_ids": by_count[count],
                "fraction": fraction,
                "counting_floor": (fraction * (1.0 - fraction) / total) ** 0.5,
            }
        )
    return {"n_total": total, "n_states": len(counts), "n_collapsed": collapsed, "states": rows}


def required_discriminability(fraction: float, n_total: int, target: float) -> float:
    """How bad `v/Delta^2` must be before counting precision stops dominating.

    Inverts `sigma_p^2 = [p(1-p) + v/Delta^2]/N`. A value far above what any real pair measures means
    the published fraction is limited by classification correctness and not by particle count.

    Args:
        fraction: The published `p`.
        n_total: Particles across the group.
        target: The `sigma_p` that would be considered a problem, e.g. 0.05.

    Returns:
        The `v/Delta^2` needed, or 0.0 when the floor alone already exceeds `target`.
    """
    return max(0.0, target**2 * n_total - fraction * (1.0 - fraction))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every path is an argument; nothing is read from a constant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--emdb",
        nargs="*",
        default=(),
        help="entry ids to fetch directly, given by their EMD- accession. Fetched by id and not "
        "searched for: the search endpoint does not reliably match a quoted DOI, and returns zero "
        "hits without saying so",
    )
    parser.add_argument("--out", type=Path, default=Path("results/emdb_occupancies.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries: list[dict] = []

    for identifier in args.emdb:
        item = entry_summary(fetch(f"{API}/entry/{identifier}"))
        print(f"entry {identifier}: n={item['n_images']} doi={item['doi']}", flush=True)
        summaries.append(item)

    groups = group_by_doi(summaries)
    reported = []
    for doi, entries in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        split = occupancies(entries)
        if split is None:
            print(f"\n{doi}: {len(entries)} map(s), one distinct particle count -- not a split, dropped")
            continue
        largest = split["states"][0]
        need = required_discriminability(largest["fraction"], split["n_total"], _TARGET)
        print(f"\n{doi}")
        print(f"  {len(entries)} maps, {split['n_states']} distinct counts, {split['n_collapsed']} collapsed")
        print(f"  {split['n_total']:,} particles total")
        for state in split["states"]:
            # every map sharing the count, not one of them: a collapsed count is several maps and
            # naming only the first would hide which ones the fraction was pooled over
            print(
                f"    {','.join(state['emdb_ids']):<10} n={state['n_images']:>9,}  f={state['fraction']:.4f} "
                f"+/- {state['counting_floor']:.4f} (counting floor)"
            )
        print(f"  v/Delta^2 needed for sigma_p = {_TARGET:g}: {need:,.1f}")
        reported.append({"doi": doi, **split, "required_discriminability": need, "entries": entries})

    write_result(
        args.out,
        {"n_entries": len(summaries), "n_groups": len(groups), "target": _TARGET, "groups": reported},
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
