"""Regrade only other rows in saved JSONL/Parquet without generation or source filtering."""
import argparse
import json
from pathlib import Path

from common.io import write_json
from eval.final_answer import VERSION
from pipeline.sample_sft_pool import annotate, validate_annotation


def regrade_row(row):
    validate_annotation(row)
    if row.get('answer_type') != 'other':
        return row, None
    responses = [{'text':text, 'token_count':tokens, 'finish_reason':reason}
                 for text,tokens,reason in zip(row['responses'],row['response_tokens'],row['finish_reasons'])]
    details = []
    updated, errors = annotate(responses, row['gold_answer'], problem=row['problem'],
                               answer_type='other', audit_output=details)
    # Only these three fields may change. Preserve responses, token counts, and all source fields.
    result = dict(row)
    for key in ['extracted_answers','correct','num_correct']:
        result[key] = updated[key]
    return result, {'id':row.get('id'), 'old_correct':row['correct'], 'new_correct':result['correct'],
        'old_num_correct':row['num_correct'], 'new_num_correct':result['num_correct'],
        'old_extractions':row['extracted_answers'], 'new_extractions':result['extracted_answers'],
        'grading_audit':details, 'errors':errors}


def regrade_file(input_path, output_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, out = Path(input_path), Path(output_dir)
    if source.suffix.lower() not in {'.jsonl','.parquet'}:
        raise ValueError('Choose a JSONL or Parquet file containing the six response columns.')
    out.mkdir(parents=True, exist_ok=True)
    target = out / (source.stem + '.other-regraded' + source.suffix)
    if target.resolve() == source.resolve():
        raise ValueError('Cannot overwrite the input')
    audit_path = target.with_suffix('.audit.jsonl')
    summary_path = target.with_suffix('.summary.json')
    # A completed marker never survives a failed rerun.
    if summary_path.exists():
        summary_path.unlink()
    summary = dict(input=str(source), output=str(target), policy_version=VERSION, rows=0,
                   other_rows=0, old_correct=0, new_correct=0, changed_flags=0, grading_errors=0,
                   new_generation=False, dropped_rows=0)
    def process(row, audit):
        updated, detail = regrade_row(row)
        summary['rows'] += 1
        summary['old_correct'] += row['num_correct']
        summary['new_correct'] += updated['num_correct']
        if detail:
            summary['other_rows'] += 1
            summary['changed_flags'] += sum(a != b for a,b in zip(row['correct'],updated['correct']))
            summary['grading_errors'] += sum(e is not None for e in detail['errors'])
            audit.write(json.dumps(detail,ensure_ascii=False)+'\n')
        return updated
    temporary = target.with_suffix(target.suffix+'.partial')
    with audit_path.open('w') as audit:
        if source.suffix.lower()=='.jsonl':
            with source.open() as src, temporary.open('w') as dest:
                for line in src:
                    if line.strip():
                        dest.write(json.dumps(process(json.loads(line),audit),ensure_ascii=False)+'\n')
        else:
            parquet = pq.ParquetFile(source)
            with pq.ParquetWriter(temporary, parquet.schema_arrow, compression='zstd') as writer:
                for batch in parquet.iter_batches(batch_size=32):
                    updated = [process(row,audit) for row in batch.to_pylist()]
                    writer.write_table(pa.Table.from_pylist(updated,schema=parquet.schema_arrow))
    temporary.replace(target)
    write_json(summary_path, summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    regrade_file(args.input,args.output_dir)
