import csv
import io

import pytest

from claimcheck.data import load_answers, load_facts, load_gold, parse_value
from claimcheck.facts import UnknownBrandError, all_fact_ids, get_brand_facts, known_brands
from claimcheck.schemas import Claim, OutputValidationError, validate_claims
from claimcheck.segment import split_sentences


def test_csv_quoted_commas_and_embedded_newlines(tmp_path):
    p = tmp_path / "answers.csv"
    text = 'First, a comma.\n\nSecond "quoted" line, INR 1,299.'
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["answer_id", "split", "brand_id", "question", "ai_answer"])
    w.writerow(["X1", "development", "cleartrail", "q, with comma", text])
    p.write_text(buf.getvalue(), encoding="utf-8")
    [a] = load_answers(p)
    assert a.ai_answer == text
    assert a.question == "q, with comma"


def test_real_answers_preserved_and_counts():
    answers = load_answers()
    assert len(answers) == 50
    assert sum(a.split == "development" for a in answers) == 40
    assert any("\n\n" in a.ai_answer for a in answers)
    assert len(load_facts()) == 132
    assert len(load_gold()) == 376


def test_boolean_parsing_is_not_truthiness():
    assert parse_value("false", "boolean") is False
    assert parse_value("true", "boolean") is True
    assert parse_value("0.35", "number") == 0.35
    with pytest.raises(ValueError):
        parse_value("False", "boolean")


def test_splitter_reproduces_every_gold_span():
    texts = {a.answer_id: a.ai_answer for a in load_answers()}
    spans = {aid: {(s.start, s.end) for s in split_sentences(t)} for aid, t in texts.items()}
    for g in load_gold():
        assert (g.start_char, g.end_char) in spans[g.answer_id], g.claim_id
        assert texts[g.answer_id][g.start_char : g.end_char] == g.quote


def test_fact_tool():
    assert known_brands() == ["cleartrail", "cloudnote", "stonewave"]
    r = get_brand_facts("cleartrail")
    assert r["fact_version"] == "speaker-catalogue-v4"
    assert all(f["fact_id"].startswith("cleartrail.") for f in r["facts"])
    with pytest.raises(UnknownBrandError):
        get_brand_facts("rivalsound")


TEXT = "Advice here. The Mini costs INR 1,299. The Plus weighs 580 grams."


def _claim(s, e, label="supported", fact_ids=(), aid="A1", cid=None, quote=None):
    return Claim(
        claim_id=cid or f"{aid}:{s}:{e}",
        start_char=s,
        end_char=e,
        quote=TEXT[s:e] if quote is None else quote,
        label=label,
        fact_ids=list(fact_ids),
        reason="Because.",
    )


def test_validation_accepts_valid_and_sorts():
    out = validate_claims("A1", TEXT, [_claim(39, 65), _claim(13, 38, fact_ids=["cleartrail.mini.price_2026"])], all_fact_ids())
    assert [c.start_char for c in out] == [13, 39]


@pytest.mark.parametrize(
    "claims",
    [
        [_claim(13, 999)],  # out of bounds
        [_claim(13, 38, quote="The Mini costs INR 1,300.")],  # quote mismatch
        [_claim(12, 38)],  # leading whitespace
        [_claim(13, 38, cid="A1:0:1")],  # wrong claim_id
        [_claim(13, 38), _claim(13, 38)],  # duplicate
        [_claim(13, 38), _claim(20, 50)],  # overlap
        [_claim(13, 38, fact_ids=["made.up.fact"])],  # invented fact id
    ],
)
def test_validation_rejects(claims):
    with pytest.raises(OutputValidationError):
        validate_claims("A1", TEXT, claims, all_fact_ids())


def test_unsupported_label_rejected_by_schema():
    with pytest.raises(Exception):
        _claim(13, 38, label="maybe")
