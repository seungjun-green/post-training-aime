# DeepSeek smoke comparison review

Reviewed archive: `/Users/seungjunlee/Downloads/deepseek_smoke_comparison.zip`.
This is a review of the supplied results, not a new API run or a change to the notebooks.

## Result

Flash is the provisional choice for the next validation pass. Pro does not show a clear advantage in this sample and contains a confirmed diameter/radius mistranslation. Neither model's automatic pass rate establishes that the dataset is ready for a full run under the current exact-preservation requirement.

| Recorded metric | deepseek-flash | deepseek-v4-pro |
|---|---:|---:|
| Rows accepted by existing checks | 35/35 | 35/35 |
| Flagged rows | 0 | 0 |
| Quality-retried rows | 0 | 0 |
| API responses | 56 | 56 |
| Responses finishing with `stop` | 56 | 56 |
| Prompt tokens | 75,461 | 75,405 |
| Completion tokens | 88,908 | 88,898 |

The two models used identical source rows, verified by dataset and row ID (journal order differs because requests complete asynchronously). There are 45 translated fields per model: 35 problem statements plus five reasoning traces and five attempts. No API-level refusal or output-limit termination appears in the recorded responses.

## Confirmed findings

### Pro changes diameter to radius

Dataset `s1k_1.1`, row `s1k_1.1:train:535`, field `question`:

- English: `A semicircle with diameter $d$ is contained in a square whose sides have length 8.`
- Flash: `지름 $d$인 반원이 한 변의 길이가 8인 정사각형 안에 포함되어 있다.`
- Pro: `반지름이 $d$인 반원이 한 변의 길이가 8인 정사각형에 포함되어 있다.`

`지름` means diameter; `반지름` means radius. Pro changes the quantity being maximized. Its translated reasoning starts with diameter correctly, introducing a question/reasoning inconsistency. This passes the notebook's Hangul, length, boxed-answer and chunk-completion checks.

### Both use imprecise tuple terminology

Dataset `aime_2025`, row `14`, field `problem`:

- English: `ordered triples of positive integers $(a, b, c)$`
- Both outputs: `양의 정수의 순서쌍 $(a, b, c)$`

The visible three-component tuple helps retain the intent, but `순서쌍` normally denotes an ordered pair. Prefer `양의 정수로 이루어진 순서 있는 세쌍 $(a, b, c)$` or a consistent equivalent for ordered triples.

### Both repair/reformat raw reasoning mathematics

Dataset `s1k_1.1`, row `s1k_1.1:train:991`, field `deepseek_thinking_trajectory`:

- Source contains `effectively 82 copies of S0` and `8*(8/92) = 82 / 92`.
- Both translations introduce `8^2` and `9^2` in those passages.

These are plausible repairs of lost superscript formatting in the source, but they are still unrequested changes under the instruction to preserve expressions exactly and not correct the reasoning. Both models also introduce math delimiters around previously undelimited expressions. Do not count all such changes as mathematical errors; distinguish presentation changes, source repairs and actual changes of meaning.

Pro also changes some existing `$...$` delimiters to `\(...\)` and removes `\displaystyle` in the polynomial in `aime_2026`, row `26`. These do not change the value of the polynomial, but violate exact LaTeX preservation.

### Paragraph structure is not perfectly preserved

Blank-line-separated paragraph counts change in two Flash reasoning fields and all five Pro reasoning fields. For example, row `991` changes from 925 source paragraphs to 918 for Flash and 923 for Pro. This is evidence of formatting drift, not proof that reasoning steps were omitted. A full paragraph alignment would be needed to establish omissions.

## Next step

Use Flash as the leading candidate, but strengthen preservation checks and rerun the smoke test before launching the full translation:

1. Protect existing math/code spans with validated placeholders; restore them byte-for-byte. Define explicit handling for undelimited mathematics and malformed source LaTeX.
2. Add terminology guidance and review checks for meaning-sensitive terms such as diameter/radius and ordered pairs/triples.
3. Compare mathematical content without treating legitimate Korean word-order changes as errors. A raw ordered list of math spans produces false positives.
4. Continue manual review of long traces and the final Korean evaluation sets. Boxed-answer equality alone cannot validate the translated question.

No translation output was edited, no full run was started, and the notebooks remain unchanged. This review combines programmatic integrity checks with inspection of problem translations and selected reasoning passages; it is not a complete line-by-line semantic certification of every long trace.
