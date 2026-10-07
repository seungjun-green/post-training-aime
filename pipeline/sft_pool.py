"""Deterministic, disk-backed, inference-free Stage 1 problem preparation."""
import ast
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from common.io import write_json, write_jsonl
from common.math_text import last_boxed


def normalize(text):
    """Lexical normalization, not symbolic algebra. Preserve numbers and command names.

    Formatting variants have identical tokens; punctuation is intentionally discarded
    as required by the protocol. This can conflate operators, so answer conflicts
    are quarantined rather than resolved by source priority.
    """
    text = text.lower()
    text = re.sub(r'\\(?:left|right|displaystyle|textstyle|scriptstyle|scriptscriptstyle|quad|qquad|enspace|thinspace)\b', ' ', text)
    text = re.sub(r'\\[,;! :]|\$', ' ', text)
    text = re.sub(r'\\(?:dfrac|tfrac)\b', r'\\frac', text)
    text = re.sub(r'\\(?:mathrm|mathbf|mathit|text|textrm|operatorname)\s*\{([^{}]*)\}', r' \1 ', text)
    # Braces around single digits and spacing never change token boundaries.
    return ' '.join(re.findall(r'[^\W\d_]+|\d+', text, flags=re.UNICODE))


def grams(normalized, n):
    tokens = normalized.split()
    return {' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}


def shingles(normalized, n):
    return grams(normalized, n) or {normalized}


def numeric_template(normalized):
    return re.sub(r'\b\d+\b', 'NUM', normalized)


def metadata(raw):
    """Retain scalar/source metadata, never s1 reasoning/messages hidden in metadata."""
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    if not isinstance(raw, dict):
        return {}
    keep = {'source', 'domain', 'subject', 'type', 'level', 'difficulty', 'field',
            'subfield', 'answer_type', 'question_type', 'problem_type', 'id', 'year',
            'problem number', 'part', 'category', 'theorem', 'theorem category'}
    return {k: v for k, v in raw.items() if k.lower() in keep}


def adapt(source, raw, index, spec, config):
    meta = metadata(raw.get('metadata'))
    for key in ['source', 'source_type', 'cot_type', 'level', 'type', 'subject',
                'domain', 'problem_type', 'question_type', 'synthetic']:
        if key in raw:
            meta[key] = raw[key]
    for key, value in raw.items():
        if any(part in key.lower() for part in ['valid', 'quality']):
            meta[key] = value
    answer_origin = 'answer'
    if source == 'math':
        answer = last_boxed(raw.get('solution') or '')
        answer_origin = 'solution:last_boxed'
    elif source == 's1':
        # The original solution is a gold reference, distinct from generated traces.
        original_meta = raw.get('metadata', {})
        if isinstance(original_meta, str):
            try:
                original_meta = ast.literal_eval(original_meta)
            except (ValueError, SyntaxError):
                original_meta = {}
        answer = next((v for k, v in original_meta.items() if k.lower() == 'answer'), None) if isinstance(original_meta, dict) else None
        if answer is not None:
            # s1 serialized metadata occasionally double-escapes LaTeX commands.
            answer = str(answer).replace('\\\\', '\\')
            answer_origin = 'metadata:answer'
        else:
            gold = str(raw.get('solution') or '').strip()
            answer = last_boxed(gold)
            answer_origin = 'solution:last_boxed'
            if answer is None:
                answer, answer_origin = gold, 'solution:direct'
    else:
        answer = raw.get('answer')
    identifier = f"{spec['repo']}@{spec['revision']}:{config}:{spec['split']}:{index}"
    return {'id': hashlib.sha256(identifier.encode()).hexdigest(), 'dataset_source': source,
            'source_repo': spec['repo'], 'source_revision': spec['revision'],
            'source_config': config, 'source_split': spec['split'], 'source_index': index,
            'original_id': str(raw.get('id', raw.get('unique_id', index))),
            'sub_source': str(raw.get('source', raw.get('source_type', config))),
            'problem': raw.get('question' if source == 's1' else 'problem', ''),
            'gold_answer': None if answer is None else str(answer).strip(),
            'answer_origin': answer_origin, 'metadata': meta}


@lru_cache(maxsize=32768)
def parse_gold(answer, timeout=5):
    from math_verify import LatexExtractionConfig, parse
    from sympy import Basic, Tuple
    if not answer:
        return None
    value = answer.strip().strip('$').strip()
    # No prose, answers A-E, bools, or separate answers disguised as a parseable tail.
    words = re.sub(r'\\[a-zA-Z]+', ' ', value)
    if (re.search(r'[A-Za-z]{2,}', words) or re.fullmatch(r'[A-Ea-e]', value)
            or re.search(r'[,;\n]|\\(?:text|mbox)\b', value)):
        return None
    try:
        parsed = parse('$' + value + '$', extraction_config=[LatexExtractionConfig()],
                       fallback_mode='no_fallback', parsing_timeout=timeout)
        if len(parsed) != 1 or not isinstance(parsed[0], Basic) or isinstance(parsed[0], Tuple):
            return None
        return parsed
    except Exception:
        return None


def equivalent(a, b, timeout):
    from math_verify import verify
    x, y = parse_gold(a, timeout), parse_gold(b, timeout)
    if x is None or y is None:
        return False
    try:
        return bool(verify(x, y, timeout_seconds=timeout) and verify(y, x, timeout_seconds=timeout))
    except Exception:
        return False


def s1_math(row):
    meta, source = row['metadata'], row['sub_source'].lower()
    labels = json.dumps(meta, ensure_ascii=False).lower()
    if meta.get('cot_type', '').lower() != 'math':
        return False
    if re.search(r'physics|/phy\b|chem|biology|astronomy|science|puzzle|crossword|code|proof|bool', source + ' ' + labels):
        return False
    # TheoremQA contains both math and science; require explicit math domain evidence.
    if 'theoremqa' in source:
        return bool(re.search(r'mathematics|algebra|geometry|number theory|calculus|combinatorics|probability|statistics', labels))
    return any(x in source for x in ['aime', 'numinamath', 'omni-math', 'openaimath',
                                     '/math', 'stats_qual', 'qfq/quant'])


def rejection_reasons(row, cfg):
    reasons, meta, text = [], row['metadata'], row['problem']
    labels = ' '.join(str(meta.get(k, '')) for k in ['question_type', 'problem_type']).lower()
    if row['dataset_source'] == 's1' and not s1_math(row):
        reasons.append('s1_nonmath_or_unconfirmed_domain')
    if row['dataset_source'] == 'numina':
        if row['sub_source'].lower() in cfg['exclude_numina_sources']:
            reasons.append('excluded_sub_source')
        if any(('valid' in k.lower() or 'quality' in k.lower()) and
               str(v).lower() in {'no', 'false', 'invalid', 'incomplete', 'incorrect', 'bad', '0'}
               for k, v in meta.items()):
            reasons.append('dataset_invalid')
    if 'proof' in labels:
        reasons.append('proof_label')
    elif re.search(r'\b(prove|show that|demonstrate|justify)\b', text, re.I):
        reasons.append('proof_heuristic')
    if (re.search(r'multiple.choice|choice', labels) or
            len(re.findall(r'(?:\([A-E]\)|(?<!\w)[A-E][.)])', text)) >= 2):
        reasons.append('multiple_choice')
    if (re.search(r'multi.part|multiple.answer', labels) or
            len(re.findall(r'\([a-d]\)|(?m:^\s*[1-9][.)]\s)', text)) >= 2 or
            len(re.findall(r'\b(?:find|calculate|determine|compute|evaluate)\b', text, re.I)) > 1 or
            re.search(r'\b(?:find|calculate|determine|compute)\b[^.?!\n]{0,100}\band\b', text, re.I) or
            text.count('?') > 1):
        reasons.append('multipart_heuristic')
    if not text.strip() or not normalize(text):
        reasons.append('empty_problem')
    if not row['gold_answer']:
        reasons.append('missing_gold')
    elif not reasons and parse_gold(row['gold_answer'], cfg['verify_timeout_seconds']) is None:
        reasons.append('unparseable_or_nonscalar_gold')
    return reasons


class ReferenceIndex:
    """Distinct n-gram inverted index; score each reference independently."""
    def __init__(self, rows, n):
        self.rows, self.n = list(rows), n
        self.postings, self.short, self.templates = defaultdict(list), [], defaultdict(list)
        self.norms, self.sizes = [], []
        for i, row in enumerate(self.rows):
            norm = normalize(row['problem'])
            if not norm:
                raise ValueError(f"Empty reference: {row['id']}")
            gs = grams(norm, n)
            self.norms.append(norm)
            self.sizes.append(len(gs))
            for g in gs:
                self.postings[g].append(i)
            if not gs:
                self.short.append(i)
            if re.search(r'\d', norm):
                self.templates[numeric_template(norm)].append(i)

    def matches(self, text):
        norm = normalize(text)
        counts = Counter(i for g in grams(norm, self.n) for i in self.postings.get(g, ()))
        scores = {i: c / self.sizes[i] for i, c in counts.items()}
        for i in self.short:
            if self.norms[i] in norm:
                scores[i] = 1.0
        variants = set(self.templates.get(numeric_template(norm), ()))
        for i in sorted(set(scores) | variants):
            yield {'reference_id': str(self.rows[i]['id']), 'reference_set': self.rows[i]['set'],
                   'coverage': scores.get(i, 0.0),
                   'number_variant': i in variants and norm != self.norms[i]}


class Journal:
    def __init__(self, path):
        self.file = Path(path).open('w', encoding='utf-8')

    def add(self, row):
        self.file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')

    def close(self):
        self.file.close()


def iter_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def validate_config(cfg):
    if not 0 < cfg['near_miss_lower'] < cfg['coverage_threshold'] <= 1:
        raise ValueError('Require 0 < near miss < coverage <= 1')
    if not 0 < cfg['dedup_jaccard'] <= 1:
        raise ValueError('Invalid Jaccard threshold')
    for key in ['dedup_ngram', 'reference_ngram']:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Invalid {key}')
    if sorted(cfg['source_priority']) != ['math', 'numina', 's1']:
        raise ValueError('Source priority must list all three sources once')


def build_pool(rows, eval_rows, rl_rows, cfg, output):
    """Rows must be in configured source-priority order, then stable source order.

    Exact Jaccard prefix join is an alternative to probabilistic MinHash/LSH:
    sets with Jaccard >= t must intersect in prefixes of length |S|-ceil(t|S|)+1,
    under a shared total shingle ordering. SHA256 ordering spreads common prefixes;
    the final comparison always uses full strings, never hash-only equality.
    SQLite keeps postings and problem payloads off the Python heap.
    """
    validate_config(cfg)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    dbpath = out / 'work.sqlite'
    if dbpath.exists():
        dbpath.unlink()  # Only this function's scratch database; never source files.
    db = sqlite3.connect(dbpath)
    db.executescript('''
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE items (i INTEGER PRIMARY KEY, h TEXT, norm TEXT, payload TEXT, root INTEGER);
        CREATE INDEX exact_hash ON items(h);
        CREATE TABLE postings (g BLOB, size INTEGER, i INTEGER);
        CREATE INDEX prefix_lookup ON postings(g,size);
        CREATE TABLE misses (kind TEXT, coverage REAL, id TEXT, ref TEXT, payload TEXT);
    ''')
    removed = Journal(out / 'removed.jsonl')
    conflicts = Journal(out / 'conflicts.jsonl')
    merges = Journal(out / 'merged_clusters.jsonl')
    parents, summary, reasons_count, s1_types = [], {}, {}, defaultdict(Counter)
    stages = ['loaded', 'verifiable', 'deduplicated', 'decontaminated', 'rl_disjoint']
    for source in cfg['source_priority']:
        summary[source] = Counter({stage: 0 for stage in stages})
        reasons_count[source] = {stage: Counter() for stage in stages[1:]}
    def remove(row, stage, reasons, **details):
        reasons_count[row['dataset_source']][stage][reasons[0]] += 1
        removed.add(dict(row, removal_stage=stage, reason=reasons[0], all_reasons=reasons, **details))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        parents[max(a, b)] = min(a, b)
    def payload(i):
        return json.loads(db.execute('SELECT payload FROM items WHERE i=?', (i,)).fetchone()[0])
    t, n, last_priority = cfg['dedup_jaccard'], cfg['dedup_ngram'], -1
    for row in rows:
        source = row['dataset_source']
        priority = cfg['source_priority'].index(source)
        if priority < last_priority:
            raise ValueError('Input is not sorted by configured source priority')
        last_priority = priority
        summary[source]['loaded'] += 1
        failures = rejection_reasons(row, cfg)
        if source == 's1':
            label = f"{row['metadata'].get('cot_type', 'unknown')} | {row['sub_source']}"
            s1_types[label]['math_kept' if s1_math(row) else 'math_dropped'] += 1
            s1_types[label]['verifiable_kept' if not failures else 'verifiable_dropped'] += 1
        if failures:
            remove(row, 'verifiable', failures)
            continue
        summary[source]['verifiable'] += 1
        i = len(parents)
        parents.append(i)
        norm = normalize(row['problem'])
        h = hashlib.sha256(norm.encode()).hexdigest()
        exact = db.execute('SELECT i FROM items WHERE h=? AND norm=? LIMIT 1', (h, norm)).fetchone()
        db.execute('INSERT INTO items VALUES (?,?,?,?,?)', (i, h, norm, json.dumps(row, ensure_ascii=False), i))
        if exact:
            union(i, exact[0])
            continue
        gs = shingles(norm, n)
        prefix = sorted(gs, key=lambda g: (hashlib.sha256(g.encode()).digest(), g))[:len(gs)-math.ceil(t*len(gs))+1]
        candidates = set()
        for g in prefix:
            candidates.update(x[0] for x in db.execute(
                'SELECT i FROM postings WHERE g=? AND size>=? AND size<=?',
                (hashlib.sha256(g.encode()).digest(), math.ceil(t*len(gs)-1e-10), math.floor(len(gs)/t+1e-10))))
        for j in sorted(candidates):
            other = db.execute('SELECT norm FROM items WHERE i=?', (j,)).fetchone()[0]
            js = shingles(other, n)
            if len(gs & js) / len(gs | js) + 1e-12 >= t:
                union(i, j)
        db.executemany('INSERT INTO postings VALUES (?,?,?)',
                       [(hashlib.sha256(g.encode()).digest(), len(gs), i) for g in prefix])
        if i % 10000 == 0:
            db.commit()
            print(f'Filtered/indexed {i:,} verifiable rows', flush=True)
    db.executemany('UPDATE items SET root=? WHERE i=?', ((find(i), i) for i in range(len(parents))))
    db.execute('CREATE INDEX clusters ON items(root,i)')
    db.commit()
    eval_index = ReferenceIndex(eval_rows, cfg['reference_ngram'])
    rl_index = ReferenceIndex(rl_rows, cfg['reference_ngram'])
    clean = Journal(out / 'pool.jsonl')
    def overlap(row, index, kind):
        matches = list(index.matches(row['problem']))
        best = max(matches, key=lambda m: m['coverage'], default=None)
        row[f'{kind}_max_coverage'] = best['coverage'] if best else 0.0
        row[f'{kind}_best_match'] = best
        hit = [m for m in matches if m['coverage'] >= cfg['coverage_threshold']]
        row[f'{kind}_near_miss'] = False
        if not hit:
            for m in matches:
                if m['coverage'] >= cfg['near_miss_lower'] or m['number_variant']:
                    row[f'{kind}_near_miss'] = True
                    entry = dict(m, id=row['id'], dataset_source=row['dataset_source'],
                                 review_reason='number_variant' if m['number_variant'] else 'coverage_band')
                    db.execute('INSERT INTO misses VALUES (?,?,?,?,?)',
                               (kind, m['coverage'], row['id'], str(m['reference_id']), json.dumps(entry)))
        return hit
    # Iteration by root is deterministic; never load the full problem pool in RAM.
    roots = db.execute('SELECT DISTINCT root FROM items ORDER BY root')
    for (root,) in roots:
        ids = [x[0] for x in db.execute('SELECT i FROM items WHERE root=? ORDER BY i', (root,))]
        winner = payload(ids[0])
        cluster = 'dup-' + winner['id']
        winner['duplicate_cluster_id'] = cluster
        winner['duplicate_cluster_size'] = len(ids)
        # Comparing every member to the chosen representative avoids merging an
        # answer-conflicting component through an intermediate near-duplicate.
        bad = [j for j in ids[1:] if not equivalent(winner['gold_answer'], payload(j)['gold_answer'], cfg['verify_timeout_seconds'])]
        if bad:
            conflicts.add({'cluster_id': cluster, 'policy': 'quarantine_all',
                           'members': [{'id': payload(j)['id'], 'gold_answer': payload(j)['gold_answer']} for j in ids]})
            for j in ids:
                remove(payload(j), 'deduplicated', ['conflicting_answer_cluster'], duplicate_cluster_id=cluster)
            continue
        if len(ids) > 1:
            merges.add({'cluster_id': cluster, 'kept_id': winner['id'], 'member_ids': [payload(j)['id'] for j in ids]})
            for j in ids[1:]:
                remove(payload(j), 'deduplicated', ['duplicate'], duplicate_of=winner['id'], duplicate_cluster_id=cluster)
        summary[winner['dataset_source']]['deduplicated'] += 1
        hits = overlap(winner, eval_index, 'eval')
        if hits:
            remove(winner, 'decontaminated', ['eval_overlap'], matches=hits)
            continue
        summary[winner['dataset_source']]['decontaminated'] += 1
        hits = overlap(winner, rl_index, 'rl')
        if hits:
            remove(winner, 'rl_disjoint', ['rl_overlap'], matches=hits)
            continue
        summary[winner['dataset_source']]['rl_disjoint'] += 1
        # Source metadata is heterogeneous (e.g. domain can be a string or list).
        # Store losslessly as JSON text so Arrow/HF can load one stable schema.
        winner['metadata'] = json.dumps(winner['metadata'], ensure_ascii=False, sort_keys=True)
        clean.add(winner)
    for kind in ['eval', 'rl']:
        write_jsonl(out / f'{kind}_near_misses.jsonl', (json.loads(x[0]) for x in db.execute(
            'SELECT payload FROM misses WHERE kind=? ORDER BY coverage DESC,id,ref', (kind,))))
    for handle in [removed, conflicts, merges, clean]:
        handle.close()
    for source in summary:
        for before, after in zip(stages, stages[1:]):
            assert summary[source][before] == summary[source][after] + sum(reasons_count[source][after].values())
    report = {'counts': summary, 'removals_primary_reason': reasons_count,
              's1_by_type': dict(s1_types), 'counting': 'One primary reason per removed row; all reasons in removed.jsonl',
              'near_miss_counts': {kind: db.execute('SELECT count(*) FROM misses WHERE kind=?', (kind,)).fetchone()[0] for kind in ['eval', 'rl']}}
    write_json(out / 'summary.json', report)
    db.close()
    dbpath.unlink()
    return report
