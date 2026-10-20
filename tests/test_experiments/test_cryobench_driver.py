"""`reproduction/experiments/preprocess/cryobench.py` and the map/state route it feeds in `score_pool`.

Nothing here touches the network. What is tested is the two decisions the driver makes that a
download cannot check for you: WHICH members a pair of states needs, and whether a combined table
splits into pools by the only ground truth CryoBench provides -- the stack file name.
"""

import json
from dataclasses import replace

import mrcfile
import numpy as np
import pytest
from conftest import ALPHA_CARBON, atom_records, load_driver, write_pdb, write_relion_star

NAMES = [
    "IgG-1D/vols/128_org/000.mrc",
    "IgG-1D/vols/128_org/050.mrc",
    "IgG-1D/vols/128_org/077.mrc",
    "IgG-1D/images/snr0.01/000_particles_128.mrcs",
    "IgG-1D/images/snr0.01/050_particles_128.mrcs",
    "IgG-1D/images/snr0.01/077_particles_128.mrcs",
    "IgG-1D/images/snr0.01/snr0.01.star",
    "IgG-1D/images/snr0.005/000_particles_128.mrcs",
    "IgG-1D/images/snr0.005/050_particles_128.mrcs",
    "IgG-1D/images/snr0.005/snr0.005.star",
    "IgG-1D/combined_poses.pkl",
    "IgG-1D/init_mask/mask.mrc",
]
"""The real IgG-1D layout, trimmed to three states and two noise levels."""


def test_only_the_members_two_states_need_are_selected():
    """The archive is 18 GB and two states are ~200 MB of it, so what this really pins is what is
    NOT taken: the third state, the other noise level, the mask and the pose pickle."""
    driver = load_driver("preprocess/cryobench.py")

    members = driver.wanted_members(NAMES, ["000", "050"], "0.01")

    assert set(members) == {
        "IgG-1D/vols/128_org/000.mrc",
        "IgG-1D/vols/128_org/050.mrc",
        "IgG-1D/images/snr0.01/000_particles_128.mrcs",
        "IgG-1D/images/snr0.01/050_particles_128.mrcs",
        "IgG-1D/images/snr0.01/snr0.01.star",
    }


def test_an_ambiguous_noise_level_is_refused_rather_than_picked():
    """IgG-1D ships the same 100 states at three SNRs. Choosing one silently would make two runs
    that differ only in noise look identical in every record either of them writes."""
    driver = load_driver("preprocess/cryobench.py")

    with pytest.raises(ValueError, match="several noise levels"):
        driver.wanted_members(NAMES, ["000"], None)


def test_a_state_that_matches_no_stack_is_named():
    driver = load_driver("preprocess/cryobench.py")

    with pytest.raises(ValueError, match="matches no stack"):
        driver.wanted_members(NAMES, ["999"], "0.01")


def test_a_compositional_dataset_is_not_quietly_offered_the_structure_route():
    """Ribosembly ships PDBs, so it LOOKS like the structure route works. Its states differ by whole
    subunits, `align_conformers` refuses them on atom correspondence, and it is right to -- they are
    not two conformations of one molecule. The driver has to say so before a download, not after."""
    driver = load_driver("preprocess/cryobench.py")

    assert "Ribosembly" in driver.RECORDS
    assert "Ribosembly" not in driver.CONFORMATIONAL
    assert "IgG-1D" in driver.CONFORMATIONAL


@pytest.fixture(scope="module")
def two_state_pool(tmp_path_factory):
    """A CryoBench-shaped pool: one combined star indexing two per-state stacks, plus two maps."""
    directory = tmp_path_factory.mktemp("cryobench")
    box, per_state = 32, 6
    rng = np.random.default_rng(0)

    names = []
    for state in ("000", "050"):
        stack = directory / f"{state}_particles_128.mrcs"
        with mrcfile.new(stack, overwrite=True) as handle:
            handle.set_data(rng.normal(0.0, 1.0, (per_state, box, box)).astype(np.float32))
        names += [f"{i + 1:06d}@{stack.name}" for i in range(per_state)]

        volume = np.zeros((box, box, box), np.float32)
        center = box // 2
        offset = 3 if state == "050" else -3
        volume[center - 4 : center + 4, center - 4 : center + 4, center - 4 + offset : center + 4 + offset] = 1.0
        with mrcfile.new(directory / f"{state}.mrc", overwrite=True) as handle:
            handle.set_data(volume)
            handle.voxel_size = 1.5

    star = write_relion_star(
        directory / "combined.star",
        names,
        np.full(len(names), 12000.0),
        pixel_size=1.5,
        box_size=box,
        euler=rng.uniform(-180.0, 180.0, (len(names), 3)),
    )
    return star, directory


def test_a_combined_table_splits_into_pools_by_the_stack_name(two_state_pool, tmp_path):
    """The only ground truth CryoBench gives is which file a particle came out of, so `--state` is
    the whole labeling mechanism. Both halves are asserted: the count kept AND that every kept row
    names the right stack, since a filter that silently kept everything would pass a count alone."""
    star, directory = two_state_pool
    driver = load_driver("realdata/score_pool.py")

    argv = [
        "--meta",
        str(star),
        "--root",
        str(directory),
        "--map-a",
        str(directory / "000.mrc"),
        "--map-b",
        str(directory / "050.mrc"),
        "--out",
        str(tmp_path / "o.json"),
    ]
    args = driver.parse_args([*argv, "--state", "050"])
    meta, images, pixel_size, _, labels = driver.read_pool(args)

    assert len(meta) == 6
    assert images.shape[0] == 6
    assert all(name.startswith("050_") for name in meta.stack_path)
    assert pixel_size == pytest.approx(1.5)
    # ONE state is not a labeled pool: there is nothing to separate, so no labels are handed back
    # and the caller cannot accidentally read a separation off a pool of one composition
    assert labels is None


def test_two_state_prefixes_label_the_pool_by_the_file_each_particle_came_from(two_state_pool, tmp_path):
    """The labeled pool, which is the only way CryoBench's own truth reaches the fraction fit.

    Order matters and is asserted: `--state 000 050` must make 000 state A, because the label has to
    agree with which template is `--map-a` or the sign of every Lambda flips and the pool reads as
    anti-separating rather than separating.
    """
    star, directory = two_state_pool
    driver = load_driver("realdata/score_pool.py")
    argv = [
        "--meta", str(star), "--root", str(directory),
        "--map-a", str(directory / "000.mrc"), "--map-b", str(directory / "050.mrc"),
        "--out", str(tmp_path / "o.json"),
    ]  # fmt: skip

    meta, images, _, _, labels = driver.read_pool(driver.parse_args([*argv, "--state", "000", "050"]))

    assert len(meta) == images.shape[0] == 12
    assert labels is not None
    names = [name for name in meta.stack_path]
    assert all(names[i].startswith("000_") for i in np.flatnonzero(labels == 0))
    assert all(names[i].startswith("050_") for i in np.flatnonzero(labels == 1))
    assert int((labels == 0).sum()) == 6 and int((labels == 1).sum()) == 6

    # and reversing the order reverses which state is A, rather than being ignored
    _, _, _, _, flipped = driver.read_pool(driver.parse_args([*argv, "--state", "050", "000"]))
    assert int((flipped == 0).sum()) == 6
    assert all(names[i].startswith("050_") for i in np.flatnonzero(flipped == 0))


def test_limit_on_a_two_state_pool_takes_from_each_state_and_not_the_first(two_state_pool, tmp_path):
    """A benchmark's star orders its stacks one after another, so a flat `--limit` keeps only the
    first state's particles while the pool still claims two. The limit has to be per state."""
    star, directory = two_state_pool
    driver = load_driver("realdata/score_pool.py")

    argv = [
        "--meta", str(star), "--root", str(directory),
        "--map-a", str(directory / "000.mrc"), "--map-b", str(directory / "050.mrc"),
        "--out", str(tmp_path / "o.json"),
        "--state", "000", "050", "--limit", "2",
    ]  # fmt: skip
    _, _, _, _, labels = driver.read_pool(driver.parse_args(argv))

    # four particles, BOTH states represented -- not the two that a flat truncation would give
    assert labels is not None and labels.size == 4
    assert int((labels == 0).sum()) == 2
    assert int((labels == 1).sum()) == 2


def test_a_state_prefix_that_matches_nothing_says_what_the_pool_holds(two_state_pool, tmp_path):
    star, directory = two_state_pool
    driver = load_driver("realdata/score_pool.py")

    args = driver.parse_args(
        [
            "--meta",
            str(star),
            "--root",
            str(directory),
            "--map-a",
            str(directory / "000.mrc"),
            "--map-b",
            str(directory / "050.mrc"),
            "--out",
            str(tmp_path / "o.json"),
            "--state",
            "999",
        ]
    )
    with pytest.raises(ValueError, match="matches none of this pool"):
        driver.read_pool(args)


def test_known_pose_refuses_a_table_with_no_poses(two_state_pool, tmp_path, monkeypatch):
    """The oracle rung needs a pose per particle. Falling back to the grid would report a marginal
    number under the name of a ceiling."""
    star, directory = two_state_pool
    driver = load_driver("realdata/score_pool.py")

    # patched on `pool`, which is where the read now lives: the driver hands it paths and flags and
    # no longer owns `load` itself
    from cryodisc import pools as pool

    real_load = pool.load
    monkeypatch.setattr(pool, "load", lambda p, **kw: replace(real_load(p, **kw), euler=None, has_euler=False))
    with pytest.raises(ValueError, match="--known-pose needs a pose per particle"):
        driver.main(
            [
                "--meta",
                str(star),
                "--root",
                str(directory),
                "--map-a",
                str(directory / "000.mrc"),
                "--map-b",
                str(directory / "050.mrc"),
                "--out",
                str(tmp_path / "o.json"),
                "--known-pose",
                "--nodes",
                "128",
            ]
        )


def test_the_calibration_default_follows_whether_poses_are_recorded(tmp_path):
    """Neither choice is right in both cases, so the default is resolved rather than fixed.

    Fitting at the recorded poses removes the bank from the calibration, but only a pool that HAS
    recorded poses can do it. So `--known-pose` selects `poses` and its absence selects `bank`, and
    stating `--nodes` is required only for a caller who asks for the bank back.
    """
    driver = load_driver("realdata/score_pool.py")
    common = [
        "--meta",
        "m.star",
        "--root",
        ".",
        "--out",
        str(tmp_path / "o.json"),
        "--map-a",
        "a.mrc",
        "--map-b",
        "b.mrc",
    ]

    # the pairing the bias measurement argues for, without having to be asked for
    assert driver.parse_args([*common, "--known-pose"]).calibrate == "poses"
    # and marginalizing cannot fit at a pose it was not given, so it keeps the bank
    assert driver.parse_args(common).calibrate == "bank"

    # asking for the bank back under --known-pose is the one case that still needs --nodes stated,
    # because that is the combination whose amplitude depends on a grid it does not score with
    with pytest.raises(SystemExit):
        driver.parse_args([*common, "--known-pose", "--calibrate", "bank"])
    stated = driver.parse_args([*common, "--known-pose", "--calibrate", "bank", "--nodes", "27648"])
    assert stated.nodes == 27648

    # the resolved default does NOT inherit that requirement: no bank enters, so no bank size matters
    assert driver.parse_args([*common, "--known-pose"]).nodes == driver._DEFAULT_NODES
    # --calibrate poses without --known-pose is the valid marginal-arm mode: it fits bank-free at the
    # recorded poses the .star carries, then the score still marginalizes. So it parses; the
    # recorded-poses requirement is enforced at runtime against the meta (meta.has_euler), not here.
    assert driver.parse_args([*common, "--calibrate", "poses"]).calibrate == "poses"


def test_the_two_state_routes_are_exclusive(tmp_path):
    """Both given is a caller who thinks one of them is being used, and neither given used to be an
    argparse `required=True`; now that both are optional, the refusal has to be explicit."""
    driver = load_driver("realdata/score_pool.py")
    common = ["--meta", "m.star", "--root", ".", "--out", str(tmp_path / "o.json")]

    with pytest.raises(SystemExit):
        driver.parse_args(common)
    with pytest.raises(SystemExit):
        driver.parse_args([*common, "--structure-a", "a.pdb", "--structure-b", "b.pdb", "--map-a", "a.mrc"])
    with pytest.raises(SystemExit):
        driver.parse_args([*common, "--map-a", "a.mrc"])


def test_the_structure_route_still_works_and_records_which_it_used(tmp_path):
    """The map route is an addition, not a replacement, and the result has to say which ran: a
    log-ratio scored against a superposed pair and one scored against two maps as given are
    different measurements with the same shape."""
    from cryodisc import CollectionPlan

    driver = load_driver("realdata/score_pool.py")
    rng = np.random.default_rng(1)
    coords = rng.uniform(-8.0, 8.0, (40, 3))
    for name, xyz in (("a.pdb", coords), ("b.pdb", coords + np.array([[1.5, 0.0, 0.0]]))):
        write_pdb(tmp_path / name, atom_records(xyz, atoms=ALPHA_CARBON))

    args = driver.parse_args(
        [
            "--meta",
            "m.star",
            "--root",
            ".",
            "--out",
            str(tmp_path / "o.json"),
            "--assembly",
            "none",
            "--structure-a",
            str(tmp_path / "a.pdb"),
            "--structure-b",
            str(tmp_path / "b.pdb"),
        ]
    )
    models, states = driver.build_states(args, CollectionPlan(dose=40.0, pixel_size=1.5), 48, 1.5)

    assert states["from"] == "structures"
    assert states["n_matched"] == 40
    assert len(models) == 2


def test_the_fetch_plan_is_recorded_with_the_member_list(tmp_path, monkeypatch):
    """A fetch is the provenance of every number downstream of it, so the record names the exact
    archive members taken -- not just the dataset, which does not distinguish two noise levels."""
    driver = load_driver("preprocess/cryobench.py")

    class FakeArchive:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def namelist(self):
            return NAMES

        def open(self, name):
            import io

            return io.BytesIO(name.encode())

    monkeypatch.setattr(driver.zipfile, "ZipFile", lambda _f: FakeArchive())
    monkeypatch.setattr(driver, "HttpFile", lambda url, timeout=300.0: None)

    out = tmp_path / "fetch.json"
    assert (
        driver.main(
            [
                "fetch",
                "--dataset",
                "IgG-1D",
                "--states",
                "000",
                "050",
                "--snr",
                "0.01",
                "--dest",
                str(tmp_path / "data"),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    result = json.loads(out.read_text())["result"]
    assert result["dataset"] == "IgG-1D"
    assert result["states"] == ["000", "050"]
    taken = {item["member"] for item in result["members"]}
    assert "IgG-1D/images/snr0.01/000_particles_128.mrcs" in taken
    assert "IgG-1D/images/snr0.005/000_particles_128.mrcs" not in taken
    assert (tmp_path / "data" / "IgG-1D/images/snr0.01/snr0.01.star").exists()


def test_a_server_that_ignores_range_is_refused_rather_than_trusted(monkeypatch):
    """`HttpFile` is only correct if the server honors Range, and nothing in the protocol guarantees it.

    A proxy that ignores it answers 200 with the WHOLE body: taken on trust, that is the wrong bytes
    returned as the right ones and a first read that buffers the entire archive. Compute nodes here
    reach the internet only through `http_proxy`, so such an intermediary is already on the path.
    """
    import io
    import urllib.request

    driver = load_driver("preprocess/cryobench.py")
    body = bytes(range(256)) * 8

    class Response(io.BytesIO):
        def __init__(self, payload, status):
            super().__init__(payload)
            self.status = status
            self.headers = {"Content-Length": str(len(body))}

        def geturl(self):
            return "http://host/x.zip"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def honors(request, timeout=None):
        if request.get_method() == "HEAD":
            return Response(b"", 200)
        first, last = (int(part) for part in request.headers["Range"].removeprefix("bytes=").split("-"))
        return Response(body[first : last + 1], 206)

    def ignores(request, timeout=None):
        return Response(b"" if request.get_method() == "HEAD" else body, 200)

    monkeypatch.setattr(urllib.request, "urlopen", honors)
    handle = driver.HttpFile("http://host/x.zip")
    handle.seek(-8, 2)
    assert handle.read(8) == body[-8:]

    monkeypatch.setattr(urllib.request, "urlopen", ignores)
    handle = driver.HttpFile("http://host/x.zip")
    handle.seek(-8, 2)
    with pytest.raises(OSError, match="ignoring Range"):
        handle.read(8)


@pytest.fixture(scope="module")
def separating_two_state_pool(tmp_path_factory):
    """A labeled pool whose particles really do come from their own state's volume.

    `two_state_pool` is pure noise, which is right for testing the label plumbing and useless for
    testing whether a separation is measured: it has none. Here each state's particles are that
    state's own projection plus noise, so mean Lambda must come out with opposite signs on the two
    halves and the shuffle null has something to be a null OF.
    """
    directory = tmp_path_factory.mktemp("separating")
    box, per_state = 32, 40
    rng = np.random.default_rng(0)

    names = []
    for state in ("000", "050"):
        volume = np.zeros((box, box, box), np.float32)
        center, offset = box // 2, (3 if state == "050" else -3)
        volume[center - 4 : center + 4, center - 4 : center + 4, center - 4 + offset : center + 4 + offset] = 1.0
        with mrcfile.new(directory / f"{state}.mrc", overwrite=True) as handle:
            handle.set_data(volume)
            handle.voxel_size = 1.5
        stack = (volume.sum(0)[None] * 0.6 + rng.normal(0.0, 1.0, (per_state, box, box))).astype(np.float32)
        with mrcfile.new(directory / f"{state}_particles_128.mrcs", overwrite=True) as handle:
            handle.set_data(stack)
        names += [f"{i + 1:06d}@{state}_particles_128.mrcs" for i in range(per_state)]

    star = write_relion_star(
        directory / "combined.star",
        names,
        np.full(len(names), 12000.0),
        pixel_size=1.5,
        box_size=box,
        euler=np.zeros((len(names), 3)),
    )
    return star, directory


def test_a_labeled_pool_separates_and_carries_its_own_null(separating_two_state_pool, tmp_path):
    """The measurement a two-state pool exists to make, and the null that makes it quotable.

    A pool can read a large separation because its labels are informative or because the estimator
    manufactures one, and only permuting the labels tells the two apart.
    """
    star, directory = separating_two_state_pool
    driver = load_driver("realdata/score_pool.py")
    out = tmp_path / "sep.json"

    argv = [
        "--meta", str(star), "--root", str(directory),
        "--map-a", str(directory / "000.mrc"), "--map-b", str(directory / "050.mrc"),
        "--state", "000", "050", "--known-pose", "--nodes", "54",
        # protein BRIGHT, as this fixture writes it and as CryoBench ships it. Without this the
        # calibration fits a negative amplitude and the driver refuses, which is the guard working
        "--invert-contrast",
        "--out", str(out), "--n-shuffles", "200",
    ]  # fmt: skip
    assert driver.main(argv) == 0

    separation = json.loads(out.read_text())["result"]["separation"]
    assert separation["n_a"] == separation["n_b"] == 40

    # the states pull Lambda in OPPOSITE directions, which is the instrument working at all
    assert separation["mean_lambda_given_a"] > 0.0
    assert separation["mean_lambda_given_b"] < 0.0

    # and the separation is outside the null obtained by permuting only the labels
    assert separation["null"]["sd"] > 0.0
    assert separation["null"]["sigmas"] > 3.0


def test_amplitude_shells_needs_the_pose_calibration(tmp_path):
    """a(k) off the bank would read the bank's coarseness as an envelope.

    The bank biases the amplitude low by however badly it matches, and it matches worst where the
    templates carry the most detail -- high frequency. So a bank-calibrated a(k) would slope for a
    reason that has nothing to do with the renderer, and the two causes are not separable after the
    fact. The flag refuses rather than producing a sloped spectrum nobody can attribute.
    """
    driver = load_driver("realdata/score_pool.py")
    common = [
        "--meta", "m.star", "--root", ".", "--out", str(tmp_path / "o.json"),
        "--map-a", "a.mrc", "--map-b", "b.mrc",
    ]  # fmt: skip

    with pytest.raises(SystemExit):
        driver.parse_args([*common, "--amplitude-shells", "4"])
    with pytest.raises(SystemExit):
        driver.parse_args([*common, "--known-pose", "--calibrate", "bank", "--nodes", "54", "--amplitude-shells", "4"])
    with pytest.raises(SystemExit):
        driver.parse_args([*common, "--known-pose", "--amplitude-shells", "0"])

    ok = driver.parse_args([*common, "--known-pose", "--amplitude-shells", "4"])
    assert ok.amplitude_shells == 4 and ok.calibrate == "poses"
    # and the scalar remains the default, a(k) being opt-in
    assert driver.parse_args([*common, "--known-pose"]).amplitude_shells is None


def test_the_recorded_amplitude_spectrum_covers_the_grid_and_says_which_bin_is_the_corner(
    separating_two_state_pool, tmp_path
):
    """The SHAPE of a(k) is the result, so the record has to carry it per shell rather than summarized.

    Every fft mode must fall in exactly one reported bin, or the envelope is read off part of the
    band while d^2 is computed over all of it. And the corner bin -- the modes outside every shell --
    gets its own entry with a null center frequency, because folding it onto the outermost shell
    would attribute power from beyond Nyquist to a frequency inside it.
    """
    star, directory = separating_two_state_pool
    driver = load_driver("realdata/score_pool.py")
    out, plain = tmp_path / "ak.json", tmp_path / "plain.json"

    common = [
        "--meta", str(star), "--root", str(directory),
        "--map-a", str(directory / "000.mrc"), "--map-b", str(directory / "050.mrc"),
        "--state", "000", "050", "--known-pose", "--invert-contrast", "--n-shuffles", "8",
    ]  # fmt: skip
    assert driver.main([*common, "--amplitude-shells", "4", "--out", str(out)]) == 0

    result = json.loads(out.read_text())["result"]
    spectrum = result["amplitude_spectrum"]
    assert spectrum["n_shells"] == 4
    # four shells plus the corner bin, and the corner is the one with no center frequency
    assert len(spectrum["a"]) == len(spectrum["k_cyc_px"]) == len(spectrum["n_modes"]) == 5
    assert spectrum["k_cyc_px"][-1] is None and spectrum["k_inverse_angstrom"][-1] is None
    assert all(k is not None for k in spectrum["k_cyc_px"][:-1])
    # every mode in exactly one bin: the envelope is read off the same band d^2 is computed over
    assert sum(spectrum["n_modes"]) == result["box_size"] ** 2
    assert spectrum["n_modes"][-1] > 0, "the corner bin is empty, so this box has no out-of-band modes"
    # the shells rise in frequency, which is what makes a slope in `a` readable as one
    assert spectrum["k_cyc_px"] == sorted(spectrum["k_cyc_px"][:-1]) + [None]

    # the reported scalar stays the power-weighted summary of these, so the sign guard still applies
    assert result["amplitude"] > 0.0
    assert result["calibrated_at"] == "poses"
    # ...and a scalar run records no spectrum at all, keeping the two distinguishable in the record
    assert driver.main([*common, "--out", str(plain)]) == 0
    assert json.loads(plain.read_text())["result"]["amplitude_spectrum"] is None


def test_shells_a_pool_cannot_support_are_called_out_rather_than_reported_as_an_envelope(
    separating_two_state_pool, tmp_path, capsys
):
    """The failure mode of giving `a(k)` more shells than there is data for, said out loud.

    Over-splitting fits oscillating shells, negative in some of them. A negative shell means the
    templates anti-correlate with the data at that frequency, which no envelope error can produce.
    Nothing downstream catches it -- `d^2` squares the amplitude and the reported scalar is a
    power-weighted mean that stays positive -- so the run has to say so itself.
    """
    star, directory = separating_two_state_pool
    driver = load_driver("realdata/score_pool.py")
    out = tmp_path / "oversplit.json"

    argv = [
        "--meta", str(star), "--root", str(directory),
        "--map-a", str(directory / "000.mrc"), "--map-b", str(directory / "050.mrc"),
        "--state", "000", "050", "--known-pose", "--invert-contrast",
        "--amplitude-shells", "4", "--out", str(out), "--n-shuffles", "8",
    ]  # fmt: skip
    assert driver.main(argv) == 0

    printed = capsys.readouterr().out
    fitted = json.loads(out.read_text())["result"]["amplitude_spectrum"]["a"]
    assert min(a for a in fitted if a is not None) < 0.0, "this fixture is meant to be over-split"
    assert "came out NEGATIVE" in printed
    assert "shell count this pool does not support" in printed
    # and the scalar summary is positive THROUGHOUT, which is why it cannot serve as the guard
    assert json.loads(out.read_text())["result"]["amplitude"] > 0.0


def test_the_offset_maps_are_measured_at_their_own_voxel_size(tmp_path):
    """`--structure-offset-from-maps` reads a center of mass in VOXELS and converts it to angstroms.
    Doing that with the PARTICLES' pixel size rather than the maps' scales the offset by the ratio of
    the two, so a half-corrected offset reads as a smaller signal rather than failing."""
    from cryodisc import CollectionPlan
    from cryodisc.specimen.density import write_map

    driver = load_driver("realdata/score_pool.py")
    rng = np.random.default_rng(1)
    coords = rng.uniform(-8.0, 8.0, (40, 3))
    for name, xyz in (("a.pdb", coords), ("b.pdb", coords + np.array([[1.5, 0.0, 0.0]]))):
        write_pdb(tmp_path / name, atom_records(xyz, atoms=ALPHA_CARBON))

    # one voxel each, 2 and 4 voxels right of the box center, on maps 2x COARSER than the particles
    map_voxel = 3.0
    for name, x in (("ma.mrc", 10), ("mb.mrc", 12)):
        volume = np.zeros((16, 16, 16), np.float32)
        volume[8, 8, x] = 1.0
        write_map(tmp_path / name, volume, map_voxel)

    argv = [
        "--meta", "m.star", "--root", ".", "--out", str(tmp_path / "o.json"), "--assembly", "none",
        "--structure-a", str(tmp_path / "a.pdb"), "--structure-b", str(tmp_path / "b.pdb"),
        "--structure-offset-from-maps", str(tmp_path / "ma.mrc"), str(tmp_path / "mb.mrc"),
    ]  # fmt: skip
    _, states = driver.build_states(driver.parse_args(argv), CollectionPlan(dose=40.0, pixel_size=1.5), 48, 1.5)

    # the midpoint of +2 and +4 voxels is +3, which at the MAPS' 3.0 A/voxel is +9 A. Read at the
    # particles' 1.5 A/px it would be +4.5, which is a real displacement reported as half of itself
    np.testing.assert_allclose(states["offset_in_angstroms"], [9.0, 0.0, 0.0], atol=1e-6)


def test_compare_rungs_scores_the_bank_at_the_same_amplitude_as_the_oracle(
    separating_two_state_pool, tmp_path, monkeypatch
):
    """`--compare-rungs` exists so the two rungs differ in ONE thing: which description of the pose
    the score integrates. Under `--amplitude-shells` the fitted a(k) is folded into the oracle's
    templates while the bank rung must carry the same envelope; a bank rung scored on uncalibrated
    templates inflates `known_pose_over_bank` and makes `chain_holds` true by construction."""
    star, directory = separating_two_state_pool
    driver = load_driver("realdata/score_pool.py")

    seen = []
    real = driver.pose_log_ratio
    monkeypatch.setattr(
        driver,
        "pose_log_ratio",
        lambda images, a, b, *args, **kwargs: (seen.append(np.asarray(a)), real(images, a, b, *args, **kwargs))[1],
    )

    common = [
        "--meta", str(star), "--root", str(directory),
        "--map-a", str(directory / "000.mrc"), "--map-b", str(directory / "050.mrc"),
        "--state", "000", "050", "--known-pose", "--invert-contrast", "--n-shuffles", "8",
        "--compare-rungs",
    ]  # fmt: skip
    assert driver.main([*common, "--out", str(tmp_path / "plain.json")]) == 0
    assert driver.main([*common, "--amplitude-shells", "4", "--out", str(tmp_path / "shells.json")]) == 0

    plain_bank, shells_bank = seen
    assert plain_bank.shape == shells_bank.shape
    # the bank the second run scored must carry the envelope the first one had none of
    assert not np.allclose(plain_bank, shells_bank), "the bank rung was scored on uncalibrated templates"
