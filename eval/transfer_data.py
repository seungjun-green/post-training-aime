"""Pinned transfer benchmarks and explicit generative prompting protocols."""

import hashlib
import json
from pathlib import Path

from common.io import digest, read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def fetch_rows(source):
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=source['repo'], repo_type='dataset',
                               revision=source['revision'], filename=source['filename']))
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise ValueError(f'Dataset hash mismatch: {source["repo"]}/{source["filename"]}')
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = read_jsonl(path)
    if len(rows) != source['rows']:
        raise ValueError('Dataset row count changed')
    return rows


def mmlu_example(row, *, answer=False):
    text = 'Question:\n' + row['question'] + '\nOptions:\n'
    # Upstream removes trailing N/A placeholders from validation demonstrations.
    options = [option for option in row['options'] if option != 'N/A']
    text += ''.join(f'{chr(65+i)}. {option}\n' for i, option in enumerate(options))
    if answer:
        return text + row['cot_content'].replace("A: Let's think step by step.",
                                                "Answer: Let's think step by step.") + '\n\n'
    return text + "Answer: Let's think step by step."


def make_documents(config, loader=fetch_rows):
    docs = []
    for row in loader(config['sources']['arc']):
        choices = row['choices']
        answer = chr(65 + choices['label'].index(row['answerKey']))
        labels = [chr(65+i) for i in range(len(choices['text']))]
        prompt = 'Choose the best answer to the following science question.\n\n'
        prompt += row['question'].strip() + '\n'
        prompt += '\n'.join(f'{label}. {text}' for label, text in zip(labels, choices['text'], strict=True))
        prompt += '\n\nThink step by step, then finish with "The answer is (X)" using the correct option letter.'
        docs.append({'id': 'arc/' + row['id'], 'benchmark': 'arc_challenge', 'subject': 'science',
                     'prompt': prompt, 'answer': answer, 'choices': labels})
    validation = loader(config['sources']['mmlu_validation'])
    prefixes = {}
    initial = (ROOT / 'third_party/mmlu_pro/initial_prompt.txt').read_text()
    for subject in config['mmlu_subjects']:
        examples = [r for r in validation if r['category'] == subject]
        if len(examples) != 5:
            raise ValueError(f'Expected exactly five MMLU-Pro validation examples: {subject}')
        prefixes[subject] = initial.replace('{$}', subject) + '\n' + ''.join(
            mmlu_example(r, answer=True) for r in examples)
    for row in loader(config['sources']['mmlu_test']):
        subject = row['category']
        if subject not in prefixes or row['answer'] != chr(65 + row['answer_index']):
            raise ValueError('Invalid MMLU-Pro subject/answer')
        # Test-set cot_content and gold answer never enter the prompt.
        docs.append({'id': f'mmlu/{row["question_id"]}', 'benchmark': 'mmlu_pro', 'subject': subject,
                     'prompt': prefixes[subject] + mmlu_example(row), 'answer': row['answer'],
                     'choices': [chr(65+i) for i in range(len(row['options']))]})
    for task in config['bbh_tasks']:
        raw = (ROOT / f'third_party/bbh/cot-prompts/{task}.txt').read_text()
        if '-----' not in raw:
            raise ValueError('Missing BBH prompt separator')
        prefix = raw.split('-----', 1)[1].strip()
        if prefix.count('\nQ:') != 3:
            raise ValueError(f'Expected three official BBH demonstrations: {task}')
        for i, row in enumerate(loader(config['sources']['bbh/' + task])):
            docs.append({'id': f'bbh/{task}/{i}', 'benchmark': 'bbh_nonmath', 'subject': task,
                         'prompt': prefix + '\n\nQ: ' + row['input'] + "\nA: Let's think step by step.",
                         'answer': row['target']})
    for row in loader(config['sources']['ifeval']):
        docs.append({'id': f'ifeval/{row["key"]}', 'benchmark': 'ifeval', 'subject': 'all',
                     'prompt': row['prompt'], 'instruction_id_list': row['instruction_id_list'],
                     'kwargs': row['kwargs'], 'key': row['key']})
    ids = [d['id'] for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate benchmark IDs')
    return docs


def smoke_documents(docs):
    """One real example per ARC/IFEval, MMLU subject and selected BBH task."""
    selected = {}
    for doc in docs:
        selected.setdefault((doc['benchmark'], doc['subject']), doc)
    return list(selected.values())


def prepare_data(config, output_root, loader=fetch_rows):
    root = Path(output_root) / 'data'
    path, manifest_path = root / 'documents.jsonl', root / 'manifest.json'
    # Prompt construction and vendored demonstrations are part of this identity.
    inputs = {'config': config, 'builder': digest(Path(__file__).read_text()),
              'prompts': {str(p.relative_to(ROOT)): digest(p.read_text())
                          for folder in ['bbh/cot-prompts', 'mmlu_pro']
                          for p in sorted((ROOT / 'third_party' / folder).glob('*.txt'))}}
    identity = digest(inputs)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity or not path.is_file():
            raise ValueError('Prepared data/settings changed; use a new OUTPUT_ROOT')
        docs = read_jsonl(path)
        if digest(docs) != manifest['documents_digest']:
            raise ValueError('Prepared dataset content changed')
        return docs, manifest
    if path.exists():
        raise ValueError('Incomplete data preparation: remove the data folder and rerun preparation')
    docs = make_documents(config, loader)
    counts = {b: sum(d['benchmark'] == b for d in docs)
              for b in ['arc_challenge', 'mmlu_pro', 'bbh_nonmath', 'ifeval']}
    expected = {'arc_challenge': config['sources']['arc']['rows'],
                'mmlu_pro': config['sources']['mmlu_test']['rows'],
                'bbh_nonmath': sum(config['sources']['bbh/' + t]['rows'] for t in config['bbh_tasks']),
                'ifeval': config['sources']['ifeval']['rows']}
    if counts != expected:
        raise ValueError(f'Incomplete benchmark selection: {counts}')
    manifest = {'identity': identity, 'documents_digest': digest(docs), 'counts': counts,
                'sources': config['sources'], 'bbh_tasks': config['bbh_tasks'],
                'bbh_excluded': config['bbh_excluded'], 'mmlu_subjects': config['mmlu_subjects']}
    write_jsonl(path, docs)
    write_json(manifest_path, manifest)
    return docs, manifest
