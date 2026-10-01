# Schema contract

Everything downstream — synthetic data, binning, features, models, the demo — is
written against this. Person 2 builds the full pipeline against synthetic rows
matching this schema while Person 1 is still pulling real data from OpenAlex.

## Columns

| Column | dtype | Nullable | Description |
| --- | --- | --- | --- |
| `id` | `str` | no | OpenAlex work id, e.g. `https://openalex.org/W2741809807`. Unique. |
| `title` | `str` | **yes** | Paper title. Primary text feature. OpenAlex returns `None` for some works. |
| `year` | `int` | no | Publication year. One of 2019, 2020, 2021, 2022, 2023. |
| `subfield` | `str` | no | CS subfield used as the stratification key during sampling. |
| `topic` | `str` | yes | OpenAlex `primary_topic.display_name`. |
| `venue` | `str` | **yes** | Journal or conference name. The leakage suspect — used only in ablations. |
| `n_references` | `int` | no | Length of `referenced_works`. Zero is common and meaningful. |
| `cites_total` | `int` | no | Raw `cited_by_count`. Kept for comparison only. **Never used as the target.** |
| `cites_2yr` | `int` | no | **The target.** Citations received in years y+1 and y+2. |

## Target definition

`cites_2yr` = sum of `cited_by_count` from OpenAlex's `counts_by_year` for the
two calendar years following publication:

```
cites_2yr(paper) = counts_by_year[year + 1] + counts_by_year[year + 2]
```

Missing years count as zero. Every paper is measured over an identical two-year
window, so paper age is removed by construction rather than corrected for.

2024 is excluded because its y+2 window falls in 2026, which is incomplete.

## Invariants

These must hold on the real data. `validate()` in `make_synthetic.py` checks them.

1. `id` is unique.
2. `year` is in `{2019, 2020, 2021, 2022, 2023}`.
3. `cites_2yr >= 0` and `cites_total >= 0`.
4. `cites_2yr <= cites_total` — a two-year window cannot exceed the lifetime count.
5. `cites_2yr` is right-skewed with a substantial mass at zero. If it looks
   normally distributed, the pull is wrong.
6. **Median `cites_2yr` grouped by `year` is roughly flat across 2019–2023.**
   If it trends, the target logic is broken and nothing downstream is valid.
   This is the single most important check in the project.

## Nulls

`title` and `venue` can be null in real OpenAlex data. The synthetic generator
injects nulls at a realistic rate so the pipeline is tested against them from
day one. Do not assume they are populated.

## Storage

- Format: `.parquet`
- Published as a versioned Kaggle Dataset, not committed to the repository
- The README records the dataset name and the pull script's random seed