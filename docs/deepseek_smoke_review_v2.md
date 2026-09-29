# Second DeepSeek smoke review

Archive: `/Users/seungjunlee/Downloads/deepseek_smoke_comparison (1).zip`.

## Decision

Do not start full translation with this version. Literal-preservation checks are active, but long reasoning translations fail frequently and a meaning error passes the checks. The first placeholder implementation also protects some English morphology unnecessarily.

| Result | Flash | Pro |
|---|---:|---:|
| Accepted rows overall | 31/35 | 29/35 |
| Accepted s1K rows | 1/5 | 0/5 |
| Accepted DAPO rows | 5/5 | 5/5 |
| Accepted evaluation rows | 25/25 | 24/25 |
| Rows retried | 5 | 6 |
| API responses | 65 | 68 |
| Responses ending at output limit | 3 | 4 |

Both models use protection version 1 and identical source cohorts. Rechecking all 74 stored translation fields in the latest row records found no literal-preservation discrepancies. This verifies literal inventories, not semantic equivalence or placement. A row with successful individual fields may still be rejected because another required field failed.

## Confirmed issues

### An accepted Flash evaluation question changes the relation between conditions

Dataset `aime_2025`, row `20`:

English says BC is a diameter of circle omega_2 **and** BC is perpendicular to AD.

Flash writes:

> $\overline{BC}$는 $\omega_2$과 ${\overline{BC} \perp \overline{AD}}$의 지름이다.

It attaches the perpendicularity statement to the objects of “diameter of,” yielding an invalid mathematical sentence. Correct phrasing is:

> BC는 원 omega_2의 지름이고, BC는 AD에 수직이다.

Pro translates this specific sentence correctly. Formula inventories alone cannot catch this failure. Hiding formula contents may contribute by removing grammatical/semantic context, but the archive cannot establish causation.

### English ordinal suffixes are frozen

Both outputs for `math_500`, `test/algebra/2476.json`, contain `7th학년` and `8th학년` rather than `7학년` and `8학년`. The numeric matcher shields digit-bearing words such as `7th` as a single literal. That is an implementation issue in the protection layer, not a required preservation of mathematical notation.

### Long s1K traces are mostly rejected

Flash retains only row `s1k_1.1:train:535` as a fully accepted s1K row. Pro retains none. Failures include extra numeric literals, changed symbol counts and missing/duplicated/unknown placeholders.

Some numeric failures may reflect legitimate translation of number words into digits: the rejected question in row `544` contains “not all zeroes,” while the validator reports an added `0`. The failed response text is not saved, so this explanation is plausible but cannot be confirmed from this archive. Do not equate every validation flag with a mathematical error.

The additional `hangul_missing`, zero length ratio, boxed-count and missing-chunk failures are downstream effects of discarding a field after a validation exception. They do not establish that the API itself returned empty text or omitted all those items independently.

## Earlier findings

- Both saved question translations now use diameter correctly for s1K row `535`.
- Both now use `순서 있는 세쌍` for AIME 2025 row `14`.
- The reasoning trace for s1K row `991`, containing the original `82` example, is absent from both latest records after validation failure. There is no successful final translation of that trace to approve in this run.
- Existing math/code blocks in stored outputs pass the new literal checks, including the polynomial's `\displaystyle` in AIME 2026 row `26`.

## Next revision should address

1. Preserve original mathematical context for the translator while keeping restoration controlled by code; test the accepted AIME row `20` regression explicitly.
2. Distinguish English ordinal suffixes from genuine digit-bearing mathematical identifiers.
3. Save failed response text and precise failing-chunk diagnostics so genuine changes can be distinguished from benign notation choices before changing validation rules.
4. Reduce the amount of reasoning and placeholder copying per request, and test retrying failed chunks rather than retranslating whole long rows.
5. Rerun the same smoke cohort and review both literal preservation and the relationships expressed in Korean before selecting a full-run model.

Scope: inspected available problem-statement translations, prior known failures, latest row errors and API metadata; reran literal checks on every stored field. This is not a line-by-line semantic certification of all long reasoning/solution text. No notebook or pipeline code was modified during this review, and no API calls were made.
