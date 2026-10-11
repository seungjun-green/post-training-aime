"""Auditable generated-answer scoring, official IFEval checks, and breakdowns."""

import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def choice_answer(text, choices):
    # Require a final-answer declaration or an answer-only response. Never guess
    # from a random option letter occurring inside the model's reasoning.
    pattern = r'(?:the\s+(?:best\s+)?answer\s+is|final\s+answer\s*:|answer\s*:)\s*\**\s*\(?([A-J])\)?(?=[\s.,:;!*\)]|$)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        value = matches[-1].upper()
        return value if value in choices else None
    match = re.fullmatch(r'\s*\(?([A-J])\)?[.\s]*', text, re.IGNORECASE)
    return match[1].upper() if match and match[1].upper() in choices else None


def bbh_answer(text, target):
    # Official demonstrations finish "So the answer is ...". Select the final
    # declaration; exact matching then preserves list order, parentheses, etc.
    matches = list(re.finditer(r'(?:the\s+answer\s+is|final\s+answer\s*:)', text, re.IGNORECASE))
    if matches:
        value = text[matches[-1].end():].strip().split('\n')[0].strip()
    else:
        value = text.strip()
    value = value.removesuffix('.').strip()
    if re.fullmatch(r'\([A-Z]\)', target):
        match = re.fullmatch(r'\(?([A-Z])\)?', value)
        return '(' + match[1] + ')' if match else None
    return value or None


def ifeval_scores(doc, text):
    vendor = str(ROOT / 'third_party')
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    os.environ['NLTK_DATA'] = str(ROOT / 'third_party/nltk_data')
    import langdetect
    import nltk
    from instruction_following_eval import evaluation_lib

    if os.environ['NLTK_DATA'] not in nltk.data.path:
        nltk.data.path.insert(0, os.environ['NLTK_DATA'])
    langdetect.DetectorFactory.seed = 0
    inp = evaluation_lib.InputExample(key=doc['key'], instruction_id_list=doc['instruction_id_list'],
                                     prompt=doc['prompt'], kwargs=doc['kwargs'])
    result = {}
    for mode in ['strict', 'loose']:
        random.seed(42)  # Stable even when rerunning only unfinished documents.
        output = getattr(evaluation_lib, 'test_instruction_following_' + mode)(inp, {inp.prompt: text})
        result['prompt_' + mode] = bool(output.follow_all_instructions)
        result['instructions_' + mode] = list(map(bool, output.follow_instruction_list))
    return result


def score(doc, text):
    if doc['benchmark'] == 'ifeval':
        return ifeval_scores(doc, text)
    prediction = (choice_answer(text, doc['choices']) if 'choices' in doc
                  else bbh_answer(text, doc['answer']))
    return {'prediction': prediction, 'correct': prediction == doc['answer'],
            'unparsed': prediction is None}


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['benchmark'], row['subject'])].append(row)
        groups[(row['benchmark'], 'overall')].append(row)
        if row['benchmark'] == 'mmlu_pro' and row['subject'] != 'math':
            groups[('mmlu_pro', 'non_math_overall')].append(row)
    rows = []
    for (benchmark, subject), values in sorted(groups.items()):
        common = {'Benchmark': benchmark, 'Subject/task': subject, 'Problems': len(values),
                  'Mean tokens': mean(r['token_count'] for r in values),
                  'Truncated (%)': 100 * mean(r['finish_reason'] == 'length' for r in values)}
        if benchmark == 'ifeval':
            for mode in ['strict', 'loose']:
                prompts = [r['score']['prompt_' + mode] for r in values]
                instructions = [x for r in values for x in r['score']['instructions_' + mode]]
                for metric, scores in [('prompt_' + mode, prompts), ('instruction_' + mode, instructions)]:
                    rows.append({**common, 'Metric': metric, 'Correct': sum(scores),
                                 'Denominator': len(scores), 'Accuracy (%)': 100 * mean(scores)})
        else:
            scores = [r['score']['correct'] for r in values]
            rows.append({**common, 'Metric': 'generated_answer_accuracy', 'Correct': sum(scores),
                         'Denominator': len(scores), 'Accuracy (%)': 100 * mean(scores),
                         'Unparsed (%)': 100 * mean(r['score']['unparsed'] for r in values)})
    # Report unweighted task/subject macro as well as example-weighted overall.
    for benchmark in ['mmlu_pro', 'bbh_nonmath']:
        parts = [r for r in rows if r['Benchmark'] == benchmark and r['Subject/task'] not in
                 ['overall', 'non_math_overall']]
        if parts:
            rows.append({'Benchmark': benchmark, 'Subject/task': 'task_macro',
                         'Metric': 'macro_generated_answer_accuracy',
                         'Problems': sum(r['Problems'] for r in parts),
                         'Correct': None, 'Denominator': None,
                         'Accuracy (%)': mean(r['Accuracy (%)'] for r in parts)})
    return rows
