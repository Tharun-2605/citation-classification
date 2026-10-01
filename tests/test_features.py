"""Tests for features.py.

The ablations (Day 4) depend on build_features() actually respecting the
use_title/use_meta/use_venue flags, and the whole pipeline depends on it
surviving nulls and unseen categories at transform time -- those are the two
things most likely to break silently, so they get the most coverage here.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import config as C
from features import build_features, TextSelector, EmbeddingLookup


@pytest.mark.parametrize("name", list(C.FEATURE_SETS))
def test_each_feature_set_builds_and_row_count_matches(name, synthetic_df):
    pipe = build_features(**C.FEATURE_SETS[name])
    X = pipe.fit_transform(synthetic_df)
    assert X.shape[0] == len(synthetic_df)
    assert X.shape[1] > 0


def test_venue_only_has_far_fewer_columns_than_all(synthetic_df):
    # A3_venue_only should be much narrower than A1_all, since it drops the
    # TF-IDF blocks entirely. This is the cheapest possible check that the
    # use_venue/use_title flags are actually wired to different column sets.
    all_pipe = build_features(**C.FEATURE_SETS["A1_all"])
    venue_pipe = build_features(**C.FEATURE_SETS["A3_venue_only"])
    X_all = all_pipe.fit_transform(synthetic_df)
    X_venue = venue_pipe.fit_transform(synthetic_df)
    assert X_venue.shape[1] < X_all.shape[1]


def test_no_venue_set_has_no_venue_feature_names(synthetic_df):
    pipe = build_features(**C.FEATURE_SETS["A2_no_venue"])
    pipe.fit(synthetic_df)
    names = pipe.named_steps["features"].get_feature_names_out()
    assert not any("venue" in n for n in names)


def test_venue_only_set_has_no_title_feature_names(synthetic_df):
    pipe = build_features(**C.FEATURE_SETS["A3_venue_only"])
    pipe.fit(synthetic_df)
    names = pipe.named_steps["features"].get_feature_names_out()
    assert not any("title" in n for n in names)


def test_invalid_text_mode_raises(synthetic_df):
    with pytest.raises(ValueError, match="unknown text_mode"):
        build_features(text_mode="not_a_mode")


def test_embeddings_mode_without_table_raises():
    with pytest.raises(ValueError, match="needs embeddings"):
        build_features(text_mode="embeddings")


def test_no_blocks_enabled_raises():
    with pytest.raises(ValueError, match="no feature blocks"):
        build_features(use_title=False, use_meta=False, use_venue=False)


def test_realistic_null_rates_do_not_crash(synthetic_df):
    # Matches SCHEMA.md's actual null rates (title ~1%, venue up to ~8%) --
    # this is the shape of null-ness the real OpenAlex pull produces.
    holed = synthetic_df.copy()
    holed.loc[holed.sample(frac=0.05, random_state=1).index, "title"] = None
    holed.loc[holed.sample(frac=0.3, random_state=2).index, "venue"] = None
    pipe = build_features(**C.FEATURE_SETS["A1_all"])
    X = pipe.fit_transform(holed)
    assert X.shape[0] == len(holed)


def test_all_titles_null_is_a_known_limitation(synthetic_df):
    # Documented edge case, not a bug to silently paper over: if EVERY title
    # in a batch is null, TfidfVectorizer has no vocabulary to build and
    # raises. This can't happen on the real pull (~1% null title rate per
    # SCHEMA.md) but would bite a tiny dev/debug sample. If this test starts
    # failing because the behaviour changed, update SCHEMA.md and this note
    # together.
    holed = synthetic_df.copy()
    holed["title"] = None
    pipe = build_features(**C.FEATURE_SETS["B1_title_only"])
    with pytest.raises(ValueError, match="empty vocabulary"):
        pipe.fit_transform(holed)


def test_fit_on_train_transform_on_test_with_unseen_category(synthetic_df):
    # The real failure mode: a venue/subfield that only appears in the test
    # split. handle_unknown="infrequent_if_exist" should absorb it instead of
    # raising at transform time.
    train, test = train_test_split(synthetic_df, test_size=0.2, random_state=42)
    test = test.copy()
    test.iloc[0, test.columns.get_loc("venue")] = "A Venue Never Seen In Training"
    pipe = build_features(**C.FEATURE_SETS["A1_all"])
    pipe.fit(train)
    X_test = pipe.transform(test)  # must not raise
    assert X_test.shape[0] == len(test)


def test_text_selector_fills_nulls_with_empty_string(synthetic_df):
    sel = TextSelector("title")
    holed = synthetic_df.copy()
    holed.loc[0, "title"] = None
    out = sel.transform(holed)
    assert out[0] == ""
    assert isinstance(out[0], str)


def test_embedding_lookup_unknown_id_returns_zero_vector(synthetic_df):
    dim = 8
    table = {"https://openalex.org/W9000000": np.ones(dim, dtype=np.float32)}
    lookup = EmbeddingLookup(table=table, dim=dim)
    out = lookup.transform(synthetic_df.head(5))
    # Every id in synthetic_df except the one seeded above is "unknown".
    assert out.shape == (5, dim)
    known_mask = synthetic_df.head(5)["id"] == "https://openalex.org/W9000000"
    assert (out[known_mask.values] == 1.0).all()
    assert (out[~known_mask.values] == 0.0).all()
