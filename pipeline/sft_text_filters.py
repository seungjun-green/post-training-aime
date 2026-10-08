"""User-requested text-only pool exclusions; no answer correction or model grading."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

VERSION = 'text-only-exclusions-v1'
RULES = {
    'translation_instruction': re.compile(
        r'\btranslat(?:e|ing)\s+(?:the\s+)?(?:above\s+text|text\s+above|following\s+text)\b'
        r'|\b(?:output|provide)\s+the\s+translation\s+result\b'
        r'|\bplease\s+(?:retain|keep|preserve)\s+(?:the\s+)?original\s+text'
        r'|\b(?:keep|retain|preserve)\b[^\n]{0,100}\bline\s*breaks\b[^\n]{0,100}\btranslat(?:ion|ed)\b', re.I),
    # Han ideographs, including compatibility and supplementary extension blocks.
    'chinese_characters': re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ee5f\U0002f800-\U0002fa1f\U00030000-\U000323af]'),
    'visual_reference': re.compile(
        r'\[(?:img|asy)\]|!\[[^\]]*\]\(|<img\b|\\includegraphics\b|\\begin\{tikzpicture\}'
        r'|https?://\S+\.(?:png|jpg|jpeg|svg|gif|webp)\b'
        r'|\b(?:as\s+(?:shown|illustrated|depicted)|shown\s+(?:below|above|opposite|here|on the right|on the left))\b'
        r'|\b(?:in|from|see|using|refer to|according to)\s+(?:(?:the|this|that|a|an|accompanying|following|given|attached)\s+)+(?:figure|diagram|picture|drawing|illustration|image)\b'
        r'|\b(?:this|following|accompanying|attached|above|below)\s+(?:figure|diagram|picture|drawing|illustration|image)\b'
        r'|\b(?:figure|diagram|picture|drawing|illustration|image)\s+(?:below|above|shows|illustrates|depicts|on the right|on the left)\b'
        r'|\b(?:figure|fig\.)\s*\d+\b'
        r'|\b(?:graph|graphs|pattern|patterns|arrangement|arrangements|shape|shapes|region|regions|sector|sectors)\b[^.?!\n]{0,90}\b(?:is|are)?\s*shown\b', re.I),
}


def text_exclusions(problem):
    """Return every matching category and a reproducible excerpt, not executable instructions."""
    matches = []
    for reason, pattern in RULES.items():
        match = pattern.search(problem)
        if match:
            matches.append({'reason': reason, 'match': match.group(), 'start': match.start(),
                            'excerpt': problem[max(0,match.start()-60):match.end()+120]})
    return matches


def filter_parquet(input_path, output_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = Path(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(source)
    if not {'id', 'problem'} <= set(table.column_names):
        raise ValueError('Input must contain id and problem columns')
    kept, removed, decisions = [], [], []
    for index, row in enumerate(table.select(['id','problem']).to_pylist()):
        evidence = text_exclusions(row['problem'])
        if evidence:
            removed.append(index)
            decisions.append({'source_index':index, 'source_row_1_based':index+1, 'id':row['id'],
                              'reasons':[x['reason'] for x in evidence], 'evidence':evidence})
        else:
            kept.append(index)
    clean = table.take(pa.array(kept, type=pa.int64()))
    rejected = table.take(pa.array(removed, type=pa.int64()))
    clean_path, rejected_path = output/'train-cleaned.parquet', output/'excluded.parquet'
    if source.resolve() in {clean_path.resolve(), rejected_path.resolve()}:
        raise ValueError('Output must not overwrite the input')
    pq.write_table(clean, clean_path, compression='zstd')
    pq.write_table(rejected, rejected_path, compression='zstd')
    (output/'exclusions.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in decisions))
    sources = table['sub_source'].to_pylist() if 'sub_source' in table.column_names else ['unknown']*len(table)
    report = {'policy_version':VERSION, 'input_path':str(source.resolve()),
              'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'input_rows':len(table), 'retained_rows':len(clean), 'removed_unique_rows':len(rejected),
              'per_rule_counts':dict(Counter(reason for d in decisions for reason in d['reasons'])),
              'overlap_combinations':dict(Counter(' + '.join(d['reasons']) for d in decisions)),
              'per_rule_sub_sources':{rule:dict(Counter(sources[d['source_index']] for d in decisions if rule in d['reasons'])) for rule in RULES},
              'scope':'Only the three requested text exclusions. All retained fields, gold answers, order, and Arrow schema are unchanged.',
              'visual_policy':'Conservative text-only eligibility: explicit visual-reference phrases and image/diagram markup (including unrendered Asymptote) are excluded. This does not prove that every flagged item requires an image; generic mathematical uses of figure and shown are not sufficient.',
              'outputs':{'cleaned':str(clean_path.resolve()), 'excluded':str(rejected_path.resolve())}}
    # Reopen exported artifacts and check lossless partition/schema preservation.
    assert pq.read_table(clean_path).equals(clean)
    assert pq.read_table(rejected_path).equals(rejected)
    assert clean.schema == table.schema == rejected.schema
    assert len(kept)+len(removed)==len(table) and not set(kept)&set(removed)
    assert not any(text_exclusions(p) for p in clean['problem'].to_pylist())
    report['validation']='Passed lossless partition, output round-trip, schema, and remaining-rule-match checks.'
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',required=True)
    args = parser.parse_args()
    print(json.dumps(filter_parquet(args.input,args.output_dir),ensure_ascii=False,indent=2))
