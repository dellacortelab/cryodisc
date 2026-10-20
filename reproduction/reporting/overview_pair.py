"""The example pair the overview schematic draws, shared by its figure driver and its render script.

Here rather than in either driver because drivers may not import one another
(`tests/test_experiments_layout.py`); both need the same entries, chain and render paths, and two copies
would let the cartoons and the templates silently disagree about which pair they show.
"""

from pathlib import Path

PAIR = (Path("data/pdb/7Z4C.pdb"), Path("data/pdb/7Z4G.pdb"))
"""SpCas9 with the guide RNA displaced six and twelve bases along the target strand."""
PROTEIN = "B"
"""The Cas9 chain: the traces and the cartoon superposition use it, the templates the whole assembly."""
CARTOONS = tuple(Path(f"docs/manuscript/figures/assets/overview_{path.stem}.png") for path in PAIR)
"""ChimeraX cartoon renders of the two conformers, named by PDB entry so a previous pair's renders can
never be drawn beside this pair's templates."""
