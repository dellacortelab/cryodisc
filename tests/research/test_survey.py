"""The survey's parsers and joins, which are where a population turns back into a hand-picked panel.

Every fixture here is a literal, and none of these tests touch the network -- the point of `survey`
taking its fetcher as an argument. The joins are the part that can be wrong, and a join exercised
only by a live run against four public services is a join exercised once and never again.
"""

import json

import pytest
from conftest import load_driver

from reproduction.survey import (
    MOTION_UNCLASSIFIED,
    Conformer,
    Pair,
    Protein,
    apply_dyndom_classes,
    assign_accessions,
    attach_dyndom_details,
    cached,
    is_accession,
    merge_by_accession,
    panel_candidate,
    parse_bioemu_references,
    parse_dyndom_classes,
    parse_dyndom_pairs,
    parse_dyndom_run,
    parse_dyndom_translation,
    parse_gpcrdb,
    parse_kincore,
    parse_uniprot_tsv,
    scoreable_pairs,
    sifts_chain_accessions,
    stratify_by_angle,
    summarize,
    uniprot_batches,
)

KINCORE = (
    # AAK1: both DFG labels and two dihedral clusters, so it yields both kinds of pair. The DFGin
    # state has two chains and the longer one must win.
    ">5TE0A\tAAK1\tAAK1_HUMAN\tNAK_AAK1_HUMAN\tDFGin\tBLAminus\tANP:1\tATPlike\n"
    "MKMKMKMKMK\n"
    ">5TE0B\tAAK1\tAAK1_HUMAN\tNAK_AAK1_HUMAN\tDFGin\tBLAminus\tANP:1\tATPlike\n"
    "MKMKMKMKMKMKMK\n"
    ">7LVIA\tAAK1\tAAK1_HUMAN\tNAK_AAK1_HUMAN\tDFGout\tABAminus\tNone\tNone\n"
    "MKMKMKMKMKMK\n"
    # only ever seen DFG-in, and in one cluster: the commonest row in the real file
    ">1ATPE\tPRKACA\tKAPCA_MOUSE\tAGC_PRKACA_MOUSE\tDFGin\tBLAminus\tNone\tNone\n"
    "GTGTGTGT\n"
    # a predicted model, which must not become a state
    ">AF-P31749-K1A\tAKT1\tAKT1_HUMAN\tAGC_AKT1_HUMAN\tDFGout\tBLBplus\tNone\tNone\n"
    "AAAAAAAA\n"
    ">AF-P31749-K3A\tAKT1\tAKT1_HUMAN\tAGC_AKT1_HUMAN\tDFGin\tBLAminus\tNone\tNone\n"
    "AAAAAAAA\n"
)

GPCRDB = [
    # the inactive state has an X-ray structure only; the active state has both, and the cryo-EM one
    # is the WORSE resolution of the two, so a resolution-first rule would pick the crystal
    {"pdb_code": "4IAR", "protein": "5ht1b_human", "preferred_chain": "A", "state": "Inactive", "resolution": 2.7, "type": "X-ray diffraction"},
    {"pdb_code": "6G79", "protein": "5ht1b_human", "preferred_chain": "S", "state": "Active", "resolution": 3.78, "type": "Electron microscopy", "signalling_protein": {"type": "G protein"}},
    {"pdb_code": "5V54", "protein": "5ht1b_human", "preferred_chain": "A", "state": "Active", "resolution": 2.9, "type": "X-ray diffraction"},
    {"pdb_code": "7XYZ", "protein": "drd2_human", "preferred_chain": "R", "state": "Active", "resolution": 3.1, "type": "Electron microscopy"},
    {"pdb_code": "8ABC", "protein": "drd2_human", "preferred_chain": "R", "state": "Intermediate", "resolution": 3.4, "type": "Electron microscopy"},
]  # fmt: skip

DYNDOM_NR = """
<table>
<tr><th>Protein Name</th><th>Conformers</th><th>Family</th><th>Movement</th></tr>
<tr>
  <td class="ddpagecol1"> 10-Formyltetrahydrofolate Dehydrogenase </td>
  <td class="ddpagecolumn"> 1s3i
      (A)
      <br /> 2bw0
      (A)
  </td>
  <td class="ddpagecol1"><a href="/dyndom/Subgroup.do?subgroupid=3404_m">3404_m_1</a></td>
  <td class="ddpagecol1"><a href="/dyndom/dyndomRun.do?ddid=FORMY1R1&amp;action=view">View</a></td>
</tr>
<tr>
  <td class="ddpagecol1"> Adenylate Kinase </td>
  <td class="ddpagecolumn"> 1aky <br /> 1dvr (B) </td>
  <td class="ddpagecol1"><a href="/dyndom/Subgroup.do?subgroupid=5591_m">5591_m_1</a></td>
  <td class="ddpagecol1"><a href="/dyndom/dyndomRun.do?ddid=ADENY1R1&amp;action=view">View</a></td>
</tr>
<tr>
  <td class="ddpagecol1"> Something Unparseable </td>
  <td class="ddpagecolumn"> pending </td>
  <td class="ddpagecol1"></td>
  <td class="ddpagecol1"><a href="/dyndom/dyndomRun.do?ddid=BROKE1R1&amp;action=view">View</a></td>
</tr>
</table>
"""

DYNDOM_CLASSES = """
<h3><font color="white">Classification of Domain Movements in Proteins</font></h3>
<h3><font color="white"> No Contact (no contacts in either conformation) </font></h3>
<tr><td>Aart</td><td>2I13 (A) 2I13 (B)</td><td><a href="translationAART_1R1.html">View</a></td></tr>
<h3><font color="white">Interface-Creating Movement</font></h3>
<tr><td>Formyl</td><td>1s3i (A) 2bw0 (A)</td><td><a href="translationFORMY1R1.html">View</a></td></tr>
<h3><font color="white">Interface-Preserving Movement</font></h3>
<tr><td>Adenylate</td><td>1aky 1dvr (B)</td><td><a href="translationADENY1R1.html">View</a></td></tr>
<h3><font color="white">Some Class Invented Next Year</font></h3>
<tr><td>Future</td><td>9xyz (A) 9xyw (A)</td><td><a href="translationFUTUR1R1.html">View</a></td></tr>
"""

DYNDOM_RUN = """
<h3>Domain Pairs</h3>
<table>
<tr><td>Rotation Angle (deg)</td><td>12.1</td></tr>
<tr><td>Translation (A)</td><td>0.3</td></tr>
<tr><td>Closure (%)</td><td>100.0</td></tr>
<tr><td>Rotation Angle (deg)</td><td>44.5</td></tr>
<tr><td>Translation (A)</td><td>2.7</td></tr>
<tr><td>Closure (%)</td><td>61.0</td></tr>
</table>
"""

SIFTS = json.dumps(
    {
        "1s3i": {
            "UniProt": {
                # the accession covering most of the chain, and a short fusion tag that must not win
                "O75891": {"mappings": [{"chain_id": "A", "struct_asym_id": "A", "unp_start": 1, "unp_end": 902}]},
                "P0CG48": {"mappings": [{"chain_id": "A", "struct_asym_id": "A", "unp_start": 1, "unp_end": 20}]},
                "Q9Y6K1": {"mappings": [{"chain_id": "B", "struct_asym_id": "B", "unp_start": 1, "unp_end": 300}]},
            }
        }
    }
)

UNIPROT_TSV = (
    "Entry\tEntry Name\tProtein names\tMass\tOrganism\n"
    "Q2M2I8\tAAK1_HUMAN\tAP2-associated protein kinase 1\t103885\tHomo sapiens (Human)\n"
    "P28222\t5HT1B_HUMAN\t5-hydroxytryptamine receptor 1B\t43568\tHomo sapiens (Human)\n"
    "P0DP23\tCALM1_HUMAN\tCalmodulin-1\t16838\tHomo sapiens (Human)\n"
    # a mass UniProt wrote with a separator, and one it left blank
    "O75891\tAL1L1_HUMAN\t10-formyltetrahydrofolate dehydrogenase\t98,175\tHomo sapiens (Human)\n"
    "Q9Y6K1\tDNM3A_HUMAN\tDNA methyltransferase\t\tHomo sapiens (Human)\n"
)


def kincore_pairs():
    """The parsed Kincore fixture, keyed by motion, which most Kincore tests want."""
    pairs, _ = parse_kincore(KINCORE)
    return {pair.motion: pair for pair in pairs}


def test_kincore_yields_one_pair_per_gene_per_axis():
    """DFG spatial label and dihedral cluster are two different motions of the same protein, so a
    gene that has both is two pairs and not a duplicate."""
    pairs, _ = parse_kincore(KINCORE)
    assert sorted((pair.name, pair.motion) for pair in pairs) == [
        ("AAK1", "dfg_flip"),
        ("AAK1", "dihedral_cluster"),
    ]


def test_kincore_drops_alphafold_models_and_says_how_many():
    """AKT1 has both DFG labels in the fixture, and BOTH are AlphaFold models. They must be counted
    and dropped, because a predicted conformation cannot validate a measured claim."""
    pairs, rejections = parse_kincore(KINCORE)
    assert not [pair for pair in pairs if pair.name == "AKT1"]
    assert any("2 AlphaFold model chains" in line for line in rejections)


def test_kincore_representative_is_the_longest_resolved_chain():
    """The FASTA is the modeled kinase domain, so length is how much of the domain the deposition
    resolved. First-in-file would pick by deposition date instead."""
    pair = kincore_pairs()["dfg_flip"]
    assert (pair.a.pdb, pair.a.chain, pair.a.state) == ("5te0", "B", "DFGin")
    assert pair.extra["domain_residues_a"] == 14
    assert pair.extra["n_chains_a"] == 2


def test_kincore_reports_the_single_state_genes_rather_than_dropping_them():
    """A gene with 40 DFG-in chains and nothing else is the commonest case in the real file, and a
    population whose refusals are invisible is a hand-picked panel with extra steps."""
    _, rejections = parse_kincore(KINCORE)
    assert any(line.startswith("kincore PRKACA: one state only") for line in rejections)


def test_kincore_can_demand_a_dihedral_cluster_be_seen_more_than_once():
    """AAK1's second cluster rests on one chain. Whether that is a state is the caller's judgment,
    which is why it is an argument."""
    pairs, _ = parse_kincore(KINCORE, min_cluster_chains=2)
    assert [pair.motion for pair in pairs] == ["dfg_flip"]


def test_gpcrdb_prefers_a_cryoem_structure_over_a_sharper_crystal():
    """The active state in the fixture has a 2.9 A crystal structure and a 3.78 A cryo-EM one. The
    paper is about what a cryo-EM particle can be told apart, and an active conformation seen only in
    a lattice may be a lattice artifact."""
    pairs, _ = parse_gpcrdb(GPCRDB)
    (pair,) = pairs
    assert (pair.b.pdb, pair.b.state) == ("6g79", "Active")
    assert pair.extra["cryoem_b"] is True
    assert pair.extra["cryoem_a"] is False


def test_gpcrdb_ignores_intermediate_states():
    """drd2 has Active and Intermediate only. Folding the intermediate onto the inactive side would
    put a partially activated receptor on one end of a two-state comparison."""
    pairs, rejections = parse_gpcrdb(GPCRDB)
    assert [pair.name for pair in pairs] == ["5ht1b_human"]
    assert any("drd2_human" in line for line in rejections)


def test_gpcrdb_records_the_signaling_partner_that_the_mass_join_cannot_see():
    """The accession's mass is the receptor. The imaged particle is the receptor plus a
    heterotrimeric G protein, some 85 kDa more, and this field is the only warning of that in the
    row."""
    pairs, _ = parse_gpcrdb(GPCRDB)
    assert pairs[0].extra["signalling_protein_b"] == "G protein"
    assert pairs[0].extra["signalling_protein_a"] is None


def test_dyndom_keeps_a_pair_whose_second_conformer_names_no_chain():
    """Some NR rows give a chain for one conformer and not the other. Requiring both would discard
    them for a reason unrelated to the motion; the empty chain is resolved later against the entry's
    dominant accession."""
    pairs, rejections = parse_dyndom_pairs(DYNDOM_NR)
    adenylate = next(pair for pair in pairs if pair.name == "Adenylate Kinase")
    assert (adenylate.a.pdb, adenylate.a.chain) == ("1aky", "")
    assert (adenylate.b.pdb, adenylate.b.chain) == ("1dvr", "B")
    assert any("Something Unparseable" in line for line in rejections)


def test_dyndom_pairs_carry_the_run_id_the_angle_is_fetched_with():
    """The listing has no rotation angle on it, so the id is the only route to the one quantity
    DynDom was included for."""
    pairs, _ = parse_dyndom_pairs(DYNDOM_NR)
    assert {pair.extra["ddid"] for pair in pairs} == {"FORMY1R1", "ADENY1R1"}
    assert all(pair.motion == MOTION_UNCLASSIFIED for pair in pairs)


def test_dyndom_classes_come_from_the_heading_a_run_sits_under():
    classes = parse_dyndom_classes(DYNDOM_CLASSES)
    assert classes["FORMY1R1"] == "hinge"
    assert classes["ADENY1R1"] == "shear"
    assert classes["AART_1R1"] == "no_contact"


def test_an_unknown_dyndom_heading_leaves_its_runs_unclassified_rather_than_guessed():
    """A class this module does not know is a class it must not translate. Unclassified reads as
    unknown; `mixed` would read as a measurement."""
    classes = parse_dyndom_classes(DYNDOM_CLASSES)
    assert "FUTUR1R1" not in classes


def test_dyndom_classes_survive_onto_the_pairs_and_the_unlisted_ones_stay_unclassified():
    pairs, _ = parse_dyndom_pairs(DYNDOM_NR)
    labeled = {pair.extra["ddid"]: pair.motion for pair in apply_dyndom_classes(pairs, {"FORMY1R1": "hinge"})}
    assert labeled == {"FORMY1R1": "hinge", "ADENY1R1": MOTION_UNCLASSIFIED}


def test_a_dyndom_run_reports_its_largest_rotation_with_that_pairs_translation():
    """A multi-domain protein has one row per moving domain pair. The largest movement is the
    magnitude of the movement, and its translation and closure must come from the SAME row -- reading
    the angle from one domain pair and the closure from another would fabricate a screw axis."""
    run = parse_dyndom_run("FORMY1R1", DYNDOM_RUN)
    assert run is not None
    assert run.rotation_deg == pytest.approx(44.5)
    assert run.translation_a == pytest.approx(2.7)
    assert run.closure_percent == pytest.approx(61.0)
    assert run.n_domain_pairs == 2


def test_the_summary_page_is_the_fallback_for_a_run_page_that_will_not_serve():
    """Some runs answer 500 on the run page while their static summary page answers 200. A null angle
    is a row that does not reach the motion-magnitude figure at all, so the lesser page is read rather
    than the row given up -- and the row says which page it came from, because a summary carries no
    translation and a reader comparing translations must be able to tell those rows apart."""
    run = parse_dyndom_translation("ACETA2R1", "<tr><td>Angle of Rotation</td><td>22.1 degrees</td></tr>")
    assert run is not None
    assert run.rotation_deg == pytest.approx(22.1)
    assert run.page == "summary"
    # None and not 1: a row claiming one domain pair it never read is worse than one admitting it
    # does not know
    assert run.n_domain_pairs is None
    assert parse_dyndom_translation("ACETA2R1", "<html>500</html>") is None


def test_a_run_page_without_an_angle_is_none_rather_than_zero():
    """An expired session answers 200 with a page that has no number on it, and a zero-degree
    rotation is a real and interesting measurement."""
    assert parse_dyndom_run("FORMY1R1", "<html>Please log in</html>") is None


def test_bioemu_is_parsed_by_column_name_because_the_order_moves():
    """`domainmotion` leads with `test_case` and `crypticpocket` ends with it. A positional parse
    reads the accession as a PDB id in one of the two categories."""
    motion, _ = parse_bioemu_references(
        "test_case,pdbidchain_i,pdbidchain_j\nP03047,7UBJ_A,7UBL_A\n", category="domainmotion"
    )
    cryptic, _ = parse_bioemu_references(
        "pdbidchain_i,pdbidchain_j,ligand_name,holo_pdbidchain,test_case\n1NEP_A,2HKA_C,C3S,2HKA_C,P79345\n",
        category="crypticpocket",
    )
    assert (motion[0].hint, motion[0].a.pdb, motion[0].b.pdb) == ("P03047", "7ubj", "7ubl")
    assert (cryptic[0].hint, cryptic[0].a.chain, cryptic[0].extra["ligand_name"]) == ("P79345", "A", "C3S")


def test_a_bioemu_case_id_that_is_not_an_accession_falls_back_to_sifts_rather_than_poisoning_a_query():
    """`oodval` carries a test case `P0205`, five characters, which is not an accession in any UniProt
    format. UniProt validates the `accession` filter and answers 400 for the WHOLE query, so leaving
    it in a batch of a hundred costs the other ninety-nine their masses -- which surfaces as rows
    silently missing a mass rather than as a complaint about the bad identifier."""
    pairs, rejections = parse_bioemu_references(
        "pdbidchain_i,pdbidchain_j,test_case\n6jt0_B,7d9r_B,P0205\n", category="oodval"
    )
    assert pairs[0].hint is None
    assert pairs[0].a.pdb == "6jt0"
    assert "is not a UniProt accession" in rejections[0]
    assert uniprot_batches([pair.hint for pair in pairs if pair.hint], field="accession", batch=100) == []


def test_the_accession_syntax_check_accepts_both_uniprot_forms():
    assert is_accession("P00533")
    assert is_accession("A0A1B2CW14")
    assert not is_accession("P0205")
    assert not is_accession("EGFR_HUMAN")


def test_bioemu_says_so_when_its_columns_change_instead_of_returning_nothing():
    pairs, rejections = parse_bioemu_references("test_case,pdbidchain\nP19726,3j7v_G\n", category="localunfolding")
    assert not pairs
    assert "expected a pair and a test case" in rejections[0]


def test_sifts_picks_the_accession_covering_most_of_the_chain():
    """A chain mapping to two accessions is a fusion construct, not an error, and the 20-residue tag
    is not what the particle is. Refusing the chain would discard real depositions for being
    annotated honestly."""
    chains = sifts_chain_accessions(json.loads(SIFTS), "1S3I")
    assert chains["A"] == "O75891"
    assert chains["B"] == "Q9Y6K1"


def test_sifts_offers_an_entry_level_default_for_a_pair_that_names_no_chain():
    """The DynDom rows with one chain missing resolve through this key."""
    chains = sifts_chain_accessions(json.loads(SIFTS), "1s3i")
    assert chains[""] == "O75891"


def test_a_sifts_response_with_no_uniprot_mapping_is_empty_rather_than_an_error():
    assert sifts_chain_accessions({"9zzz": {"UniProt": {}}}, "9zzz") == {}


def test_uniprot_batches_are_deduplicated_sorted_and_bounded():
    """Sorted so the same input gives the same query and therefore the same cache entry; bounded so
    an answer is never truncated by the service's page size, which would lose accessions silently."""
    queries = uniprot_batches(["P00533", "P0DP23", "P00533", ""], field="accession", batch=2)
    assert queries == ["accession:P00533 OR accession:P0DP23"]
    assert len(uniprot_batches([f"P{index:05d}" for index in range(250)], field="id", batch=100)) == 3


def test_uniprot_rows_without_a_readable_mass_are_skipped():
    """A row the mass window cannot filter must not reach the window. UniProt writes the mass with a
    thousands separator in some responses and blank for a few entries."""
    records = {record.accession: record for record in parse_uniprot_tsv(UNIPROT_TSV)}
    assert records["O75891"].mass_da == 98175
    assert "Q9Y6K1" not in records
    assert records["Q2M2I8"].mass_kda == pytest.approx(103.885)


def test_a_source_that_carries_its_own_identifier_is_believed_over_sifts():
    """A SIFTS lookup on a kinase entry returns the kinase, its partner and its crystallization
    nanobody. Kincore already knows which of those is the kinase."""
    pair = kincore_pairs()["dfg_flip"]
    assigned, rejections = assign_accessions([pair], by_name={"AAK1_HUMAN": "Q2M2I8"}, by_chain={})
    assert assigned == [(pair, "Q2M2I8")]
    assert not rejections


def test_conformers_mapping_to_different_proteins_are_rejected_with_both_accessions():
    """An NR pair is meant to be one protein in two conformations. A pair whose conformers are two
    proteins has a motion that is not a motion, and a coin flip between them would put a fabricated
    row in the population."""
    pair = Pair(
        source="dyndom",
        name="Chimera",
        motion="hinge",
        a=Conformer("1s3i", "A", "conformer_1"),
        b=Conformer("1s3i", "B", "conformer_2"),
    )
    assigned, rejections = assign_accessions([pair], by_name={}, by_chain=sifts_chain_accessions_for("1s3i"))
    assert not assigned
    assert "map to different proteins" in rejections[0]


def sifts_chain_accessions_for(pdb):
    """`{(pdb, chain): accession}` for the SIFTS fixture, in the shape `assign_accessions` wants."""
    return {(pdb, chain): accession for chain, accession in sifts_chain_accessions(json.loads(SIFTS), pdb).items()}


def test_an_unresolvable_pair_names_the_chain_that_could_not_be_mapped():
    pair = Pair(
        source="dyndom",
        name="Gone",
        motion="hinge",
        a=Conformer("1eap", "A", "conformer_1"),
        b=Conformer("1a0q", "L", "conformer_2"),
    )
    _, rejections = assign_accessions([pair], by_name={}, by_chain={})
    assert "no UniProt mapping for 1eap:A" in rejections[0]


def test_two_sources_reaching_one_protein_make_one_row_with_two_provenances():
    """The deduplication the whole survey rests on. Counting a protein once per source inflates the
    population exactly where the sources overlap, which is where the interesting proteins are."""
    kinase = kincore_pairs()["dfg_flip"]
    motion = Pair(
        source="dyndom",
        name="Ap2-Associated Protein Kinase 1",
        motion="hinge",
        a=Conformer("5te0", "A", "conformer_1"),
        b=Conformer("7lvi", "A", "conformer_2"),
        extra={"ddid": "AAK1_1R1"},
    )
    proteins = {record.accession: record for record in parse_uniprot_tsv(UNIPROT_TSV)}
    rows, rejections = merge_by_accession([(kinase, "Q2M2I8"), (motion, "Q2M2I8")], proteins)

    assert not rejections
    assert len(rows) == 1
    assert rows[0].sources == ("dyndom", "kincore")
    assert rows[0].motions == ("dfg_flip", "hinge")
    assert len(rows[0].pairs) == 2


def test_an_accession_uniprot_has_no_record_for_is_rejected_by_name():
    """A row with no mass would otherwise escape the mass filter rather than be excluded by it, which
    is the failure mode a filter is supposed to prevent."""
    pair = kincore_pairs()["dfg_flip"]
    rows, rejections = merge_by_accession([(pair, "P99999")], {})
    assert not rows
    assert "P99999: no UniProt record" in rejections[0]


def test_the_rotation_angle_reaches_the_row_it_was_fetched_for():
    """DynDom is in the survey for its angle, and the angle arrives one request per surviving row --
    after the filter, not before. A row that lost it on the way is a row the motion-magnitude figure
    cannot use."""
    proteins = {record.accession: record for record in parse_uniprot_tsv(UNIPROT_TSV)}
    pairs, _ = parse_dyndom_pairs(DYNDOM_NR)
    rows, _ = merge_by_accession([(pair, "O75891") for pair in pairs], proteins)
    details = {"FORMY1R1": parse_dyndom_run("FORMY1R1", DYNDOM_RUN)}
    enriched = attach_dyndom_details(rows, {key: run for key, run in details.items() if run is not None})

    by_ddid = {pair.extra["ddid"]: pair for pair in enriched[0].pairs}
    assert by_ddid["FORMY1R1"].rotation_deg == pytest.approx(44.5)
    assert by_ddid["FORMY1R1"].extra["n_domain_pairs"] == 2
    # the run page for the other pair was not fetched, and an unfetched angle is null and not zero
    assert by_ddid["ADENY1R1"].rotation_deg is None
    assert enriched[0].rotation_deg == pytest.approx(44.5)


def test_a_rows_angle_is_the_largest_its_pairs_report():
    """The axis is motion magnitude: a protein whose NR runs span 3 and 40 degrees has a 40-degree
    movement in it, and the mean of the two is a movement nothing does."""
    protein = Protein("O75891", "AL1L1_HUMAN", "dehydrogenase", "Homo sapiens (Human)", 98175)
    pairs, _ = parse_dyndom_pairs(DYNDOM_NR)
    rows, _ = merge_by_accession([(pair, "O75891") for pair in pairs], {"O75891": protein})
    enriched = attach_dyndom_details(
        rows,
        {
            "FORMY1R1": parse_dyndom_run("FORMY1R1", DYNDOM_RUN.replace("44.5", "3.0")),
            "ADENY1R1": parse_dyndom_run("ADENY1R1", DYNDOM_RUN),
        },
    )
    assert enriched[0].rotation_deg == pytest.approx(44.5)


def test_the_summary_counts_rows_and_pairs_separately():
    """A protein is one member of the population and may carry several motions, so the two counts
    answer different questions and a survey that reported one of them would be misread."""
    proteins = {record.accession: record for record in parse_uniprot_tsv(UNIPROT_TSV)}
    kinase = kincore_pairs()
    dyndom, _ = parse_dyndom_pairs(DYNDOM_NR)
    rows, _ = merge_by_accession(
        [(kinase["dfg_flip"], "Q2M2I8"), (kinase["dihedral_cluster"], "Q2M2I8"), (dyndom[0], "O75891")],
        proteins,
    )
    summary = summarize(rows)

    assert summary["n_rows"] == 2
    assert summary["n_pairs"] == 3
    assert summary["pairs_by_source"] == {"kincore": 2, "dyndom": 1}
    assert summary["rows_by_source"] == {"dyndom": 1, "kincore": 1}
    assert summary["mass_kda"]["max"] == pytest.approx(103.885)
    assert summary["rotation_deg"] is None


def test_the_cache_fetches_a_url_once_and_keys_it_by_the_whole_url(tmp_path):
    """Two SIFTS URLs differ in four characters and two UniProt queries in a batch of accessions, so
    any readable shortening collides -- and a colliding cache answers one query with another's body,
    which is the one failure here that would not look like a failure."""
    calls = []

    def fetcher(url):
        calls.append(url)
        return f"body of {url}"

    fetch = cached(fetcher, tmp_path / "cache")
    first = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/1s3i"
    second = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/1s3j"
    assert fetch(first) == f"body of {first}"
    assert fetch(first) == f"body of {first}"
    assert fetch(second) == f"body of {second}"
    assert calls == [first, second]

    # and a second cache over the same directory reads what the first one wrote
    assert cached(lambda url: pytest.fail(f"refetched {url}"), tmp_path / "cache")(first) == f"body of {first}"


def _row(accession, mass_kda, *, pdb_a="1abc", chain_a="A", pdb_b="2xyz", chain_b="B", angle=10.0):
    """One `SurveyRow` carrying one DynDom-shaped pair, for the selection tests."""
    from reproduction.survey import SurveyRow

    return SurveyRow(
        protein=Protein(
            accession=accession,
            entry_name=f"{accession}_TEST",
            protein_name=f"Protein {accession}",
            organism="Test organism",
            mass_da=int(mass_kda * 1000),
        ),
        pairs=(
            Pair(
                source="dyndom",
                name=f"Protein {accession}",
                motion="hinge",
                a=Conformer(pdb=pdb_a, chain=chain_a, state="conformer_i"),
                b=Conformer(pdb=pdb_b, chain=chain_b, state="conformer_j"),
                rotation_deg=angle,
            ),
        ),
    )


def test_a_pair_without_a_chain_an_angle_or_two_entries_is_not_scoreable():
    """Each exclusion cost a real row. The panel compares one chain against one chain, so a missing
    chain has nothing to select; the angle is the axis `d^2` is plotted against, so a row without one
    cannot serve the figure the survey exists for; and one entry against itself is a DynDom run over
    two chains of ONE deposition, which is crystal packing rather than a conformational change."""
    no_chain = _row("P00001", 60.0, chain_b="")
    no_angle = _row("P00002", 60.0, angle=None)
    self_pair = _row("P00003", 60.0, pdb_a="1abc", pdb_b="1abc")
    good = _row("P00004", 60.0)

    assert [row.protein.accession for row, _ in scoreable_pairs([no_chain, no_angle, self_pair, good])] == ["P00004"]


def test_one_pair_per_accession_and_the_largest_angle_wins():
    """A protein is one row of a population, not several, and the larger movement is the target --
    a protein whose runs span 3 and 40 degrees has a 40-degree movement in it, matching
    `SurveyRow.rotation_deg`. Scoring both would count the same protein twice, which inflates the
    population exactly where the sources are richest."""
    from reproduction.survey import SurveyRow

    small, large = _row("P00005", 60.0, angle=3.0), _row("P00005", 60.0, pdb_b="3def", angle=40.0)
    merged = SurveyRow(protein=small.protein, pairs=small.pairs + large.pairs)

    selected = scoreable_pairs([merged])
    assert len(selected) == 1
    assert selected[0][1].rotation_deg == 40.0


def test_the_sample_spans_the_angle_axis_rather_than_its_mode():
    """The angle distribution is heavily skewed -- real quartiles 9.7 / 15.4 / 31.7 degrees against a
    maximum of 178.9 -- so a sample that took the first n, or a uniform draw, would leave the
    large-motion end to one or two rows. `d^2` is plotted AGAINST this axis, so covering it is the
    requirement."""
    skewed = [_row(f"P{index:05d}", 60.0, angle=float(index) ** 2) for index in range(1, 101)]
    candidates = scoreable_pairs(skewed)

    sample = stratify_by_angle(candidates, n_wanted=10)
    angles = [pair.rotation_deg for _, pair in sample]

    assert len(sample) == 10
    assert angles == sorted(angles)
    # the top decile is represented, which is the property a contiguous or head-first sample loses
    assert max(angles) > 0.5 * max(pair.rotation_deg for _, pair in candidates)
    # and asking for more than there are returns all of them rather than repeating any
    assert len(stratify_by_angle(candidates, n_wanted=1000)) == len(candidates)


def test_a_panel_candidate_relabels_both_chains_onto_one_name():
    """Without this the correspondence is EMPTY and it reads like a mispairing.

    Two depositions name their chains independently, and `structures._KEY` carries the chain, so the
    intersection of a chain called A with a chain called D is nothing at all -- which looks like two
    structures that are not the same molecule. `assembly` is null for a related reason: a DynDom pair
    is defined ON the named chains, and building an assembly would compare quaternary structure the
    source is silent about.
    """
    row = _row("P00006", 61.2, pdb_a="2cww", chain_a="A", pdb_b="1wxx", chain_b="D", angle=7.5)
    candidate = panel_candidate(*scoreable_pairs([row])[0])

    assert candidate["labels_a"] == candidate["labels_b"] == ["A"]
    assert candidate["chains_a"] == ["A"] and candidate["chains_b"] == ["D"]
    assert candidate["assembly"] is None
    assert candidate["pdb_a"] == "2CWW" and candidate["pdb_b"] == "1WXX"
    # provenance a reader of the file needs and `targets.load_panel` ignores
    assert candidate["accession"] == "P00006"
    assert candidate["rotation_deg"] == 7.5
    assert "61 kDa" in candidate["note"] and "7.5 deg" in candidate["note"]


def test_maps_sharing_a_particle_count_are_one_state_not_a_split():
    """Two or more maps at an identical `number_images_used` are consensus or local refinements of one
    reconstruction. Treating those as separate states invents an occupancy split nobody claimed."""
    driver = load_driver("preprocess/emdb_occupancies.py")
    same = [{"n_images": 235957, "emdb_id": f"EMD-{i}"} for i in range(5)]
    assert driver.occupancies(same) is None

    split = driver.occupancies([{"n_images": 24308}, {"n_images": 11501}])
    assert split is not None
    assert split["n_total"] == 35809
    assert split["n_states"] == 2
    assert split["states"][0]["fraction"] == pytest.approx(24308 / 35809)
    # and a duplicate alongside a real split is collapsed, not counted as a third state
    with_dup = driver.occupancies([{"n_images": 24308}, {"n_images": 24308}, {"n_images": 11501}])
    assert with_dup["n_states"] == 2
    assert with_dup["n_collapsed"] == 1


def test_the_doi_comes_from_external_references_not_a_doi_field():
    """EMDB puts it under `citation_type.external_references` as a `doi:`-prefixed typed string. Reading
    a `doi` field gives None, which would group every entry under one key."""
    driver = load_driver("preprocess/emdb_occupancies.py")
    entry = {
        "emdb_id": "EMD-9001",
        "admin": {"title": "beta-galactosidase"},
        "structure_determination_list": {
            "structure_determination": [{"image_processing": [{"final_reconstruction": {"number_images_used": 24308}}]}]
        },
        "crossreferences": {
            "citation_list": {
                "primary_citation": {
                    "citation_type": {
                        "external_references": [
                            {"type_": "PUBMED", "valueOf_": "30833564"},
                            {"type_": "DOI", "valueOf_": "doi:10.1038/s41467-019-08991-8"},
                        ]
                    }
                }
            }
        },
    }
    summary = driver.entry_summary(entry)
    assert summary["doi"] == "10.1038/s41467-019-08991-8"
    assert summary["n_images"] == 24308


def test_the_discriminability_needed_to_matter_is_absurd_for_a_published_split():
    """A published occupancy is pinned by counting, so the discriminability needed to matter for it is
    absurd -- its uncertainty is not statistical."""
    driver = load_driver("preprocess/emdb_occupancies.py")
    need = driver.required_discriminability(24308 / 35809, 35809, 0.05)
    # against v/Delta^2 = 0.046 measured on real CryoBench particles
    assert need > 50.0
    assert need / 0.04626 > 1000


def test_sifts_coverage_is_not_counted_twice_when_the_two_chain_ids_coincide():
    """A mapping is filed under the entry, the author chain and the `struct_asym_id`, and for most
    entries the latter two are the same letter -- so the key tuple held a duplicate and the loop
    added that mapping's residues to it TWICE. An accession covering less of the chain then outranked
    one covering more, and the wrong UniProt accession was attached with no error."""
    payload = {
        "9xyz": {
            "UniProt": {
                "P11111": {"mappings": [{"chain_id": "A", "struct_asym_id": "B", "unp_start": 1, "unp_end": 300}]},
                "P22222": {"mappings": [{"chain_id": "A", "struct_asym_id": "A", "unp_start": 1, "unp_end": 200}]},
            }
        }
    }
    chains = sifts_chain_accessions(payload, "9xyz")

    # 300 residues beats 200 on chain A, as it does at the entry level -- the two cannot disagree
    assert chains["A"] == "P11111"
    assert chains[""] == "P11111"
