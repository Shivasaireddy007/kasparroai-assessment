import pytest

from claimcheck.data import GoldClaim
from claimcheck.schemas import AnswerOutcome, Claim
from claimcheck.scoring import check_run_summary, score

# Ten one-word "sentences" per answer let us build exact spans easily.
WORDS = [f"S{i}." for i in range(12)]
TEXT = " ".join(WORDS)


def span(i):
    s = sum(len(w) + 1 for w in WORDS[:i])
    return s, s + len(WORDS[i])


def gold(aid, i, label):
    s, e = span(i)
    return GoldClaim(aid, f"{aid}:{s}:{e}", s, e, TEXT[s:e], label, ())


def pred(aid, i, label):
    s, e = span(i)
    return Claim(claim_id=f"{aid}:{s}:{e}", start_char=s, end_char=e, quote=TEXT[s:e], label=label, fact_ids=[], reason="r.")


LABELS10 = ["supported"] * 4 + ["contradicted"] * 3 + ["unknown"] * 3
GOLD10 = [gold("A1", i, l) for i, l in enumerate(LABELS10)]
TEXTS = {"A1": TEXT, "A2": TEXT}


def test_perfect_output():
    o = AnswerOutcome(answer_id="A1", status="ok", claims=[pred("A1", i, l) for i, l in enumerate(LABELS10)])
    r = score(["A1"], [o], GOLD10, TEXTS)
    assert r["extraction"]["f1"] == 1.0 and r["joint"]["f1"] == 1.0 and r["macro_f1"] == 1.0
    assert r["missed_gold_ids"] == [] and r["unmatched_prediction_ids"] == []


def test_spec_example_g10_p10_m8_c7():
    claims = [pred("A1", i, l) for i, l in enumerate(LABELS10[:8])]
    claims[7] = pred("A1", 7, "supported")  # wrong label (gold unknown)
    claims += [pred("A1", 10, "supported"), pred("A1", 11, "unknown")]  # two extras (advice)
    r = score(["A1"], [AnswerOutcome(answer_id="A1", status="ok", claims=claims)], GOLD10, TEXTS)
    c = r["counts"]
    assert (c["G"], c["P"], c["M"], c["C"]) == (10, 10, 8, 7)
    for k in ("precision", "recall", "f1"):
        assert r["extraction"][k] == pytest.approx(0.8)
        assert r["joint"][k] == pytest.approx(0.7)
    assert len(r["missed_gold_ids"]) == 2
    assert len(r["unmatched_prediction_ids"]) == 2
    assert r["wrong_label"] == [{"claim_id": GOLD10[7].claim_id, "expected": "unknown", "predicted": "supported"}]
    total = sum(sum(row.values()) for row in r["confusion"].values())
    assert total == 10 + 2
    assert r["confusion"]["no_gold"]["missing_or_error"] == 0


def test_per_label_counts_unmatched_and_wrong_labels():
    claims = [pred("A1", 0, "contradicted"), pred("A1", 10, "unknown")]
    r = score(["A1"], [AnswerOutcome(answer_id="A1", status="ok", claims=claims)], GOLD10[:1], TEXTS)
    pl = r["per_label"]
    assert pl["supported"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "predicted": 0, "actual": 1}
    assert pl["contradicted"]["predicted"] == 1 and pl["contradicted"]["tp"] == 0
    assert pl["unknown"]["predicted"] == 1


def test_missed_claims_and_extra_advice():
    claims = [pred("A1", 0, "supported"), pred("A1", 11, "supported")]  # S11 is advice
    r = score(["A1"], [AnswerOutcome(answer_id="A1", status="ok", claims=claims)], GOLD10, TEXTS)
    assert r["counts"]["M"] == 1 and r["counts"]["P"] == 2
    assert r["extraction"]["precision"] == 0.5 and r["extraction"]["recall"] == 0.1


def test_provider_error_answer_keeps_gold_as_missed():
    ok = AnswerOutcome(answer_id="A1", status="ok", claims=[pred("A1", i, l) for i, l in enumerate(LABELS10)])
    err = AnswerOutcome(answer_id="A2", status="error", error_type="provider_error", error_message="boom")
    g = GOLD10 + [gold("A2", i, "unknown") for i in range(5)]
    r = score(["A1", "A2"], [ok, err], g, TEXTS)
    assert r["counts"]["G"] == 15 and r["counts"]["answer_errors"] == 1
    assert r["joint"]["recall"] == pytest.approx(10 / 15)
    assert r["confusion"]["unknown"]["missing_or_error"] == 5
    assert r["per_answer"]["A2"]["joint_f1"] == 0.0


def test_valid_empty_predictions():
    r = score(["A1"], [AnswerOutcome(answer_id="A1", status="ok", claims=[])], GOLD10, TEXTS)
    assert r["counts"]["P"] == 0 and r["joint"]["f1"] == 0.0 and r["extraction"]["precision"] == 0.0


def test_partial_span_does_not_match():
    s, e = span(0)
    bad = Claim(claim_id=f"A1:{s}:{e-1}", start_char=s, end_char=e - 1, quote=TEXT[s : e - 1], label="supported", fact_ids=[], reason="r.")
    r = score(["A1"], [AnswerOutcome(answer_id="A1", status="ok", claims=[bad])], GOLD10[:1], TEXTS)
    assert r["counts"]["M"] == 0


def test_rejects_missing_or_duplicate_outcomes():
    ok = AnswerOutcome(answer_id="A1", status="ok")
    with pytest.raises(ValueError):
        score(["A1", "A2"], [ok], [], TEXTS)
    with pytest.raises(ValueError):
        score(["A1"], [ok, ok], [], TEXTS)


def test_check_summary_has_null_quality():
    r = check_run_summary(["A1"], [AnswerOutcome(answer_id="A1", status="ok", claims=[pred("A1", 0, "unknown")])])
    assert r["counts"]["P"] == 1 and r["counts"]["G"] is None
    assert r["joint"] is None and r["macro_f1"] is None
