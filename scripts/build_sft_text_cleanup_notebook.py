"""Build a separate CPU Colab for the user's three text exclusions and Hub update."""
from build_notebooks import ROOT, code, markdown, write_notebook

FILTER_CODE = (ROOT / 'pipeline/sft_text_filters.py').read_text().split('\ndef filter_parquet(')[0]


def cells():
    return [
        markdown('''
        # Clean the math pool and update Hugging Face

        Removes rows matching any of these three categories:
        1. Leftover translation instructions, including “Translate the above text…” variants.
        2. Chinese/Han characters anywhere in the question.
        3. Explicit figure/diagram references or image markup unavailable as rendered input
           to a text-only model (including image links, Asymptote, and TikZ).

        **CPU runtime is sufficient.** Add a write-capable `HF_TOKEN` to Colab Secrets.
        **Run all cleans the dataset and publishes to `Seungjun/clean-math-sft-pool-30k`.**
        Set `PUSH_TO_HUB=False` to preview and download without publishing.
        Counts are calculated from the actual data, with overlaps counted only once.
        The approximate counts from your inspection are not hardcoded targets.

        This notebook applies only these three filters. It does not exclude entire source
        categories, change reference answers, generate responses, or run Math-Verify.
        Retained rows preserve every column and their original order. Explicit visual
        references are a conservative filter: some flagged questions may also be solvable
        from text. Generic geometry words alone do not cause exclusion.
        The exclusion report records IDs, all matching reasons, and matched excerpts.
        The cleaned train split becomes the default on HF; prior files/history stay available.
        '''),
        code('''
        REPO_ID = 'Seungjun/clean-math-sft-pool-30k'
        SOURCE_CONFIG = 'default'
        SOURCE_REVISION = 'main'  # Resolved to an immutable commit at load time.
        PUSH_TO_HUB = True
        OUTPUT_DIR = '/content/math-pool-text-cleanup'
        '''),
        markdown('## 1. Install dependencies and read the Colab secret'),
        code('''
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
            'datasets==5.0.1', 'huggingface-hub==0.36.2', 'pyarrow==25.0.1', 'PyYAML==6.0.3'])
        import json, hashlib, re, importlib.metadata
        from pathlib import Path
        from collections import Counter
        import pyarrow.parquet as pq
        import yaml
        from datasets import load_dataset, get_dataset_split_names
        from huggingface_hub import HfApi, DatasetCard, CommitOperationAdd, hf_hub_download
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError('Enable HF_TOKEN in Colab Secrets for this notebook.')
        api = HfApi(token=HF_TOKEN)
        out = Path(OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        # Never print, export, or place the token in a report.
        '''),
        markdown('''
        ## 2. Inspect the exclusion rules

        These patterns run on the `problem` field. Every matched rule is recorded so
        overlap counts and exclusions can be audited. They are editable before running
        the filtering cell. References to an external image count as unavailable visual
        input; this notebook does not fetch or render those images.
        '''),
        code(FILTER_CODE),
        code('''
        # Small in-memory checks only; these do not download or mutate a dataset.
        def reasons(text):
            return {x['reason'] for x in text_exclusions(text)}
        assert 'translation_instruction' in reasons('Translate the above text into English.')
        assert 'translation_instruction' in reasons('Translate the text above into English.')
        assert 'chinese_characters' in reasons('Find x. 求 x。')
        assert 'visual_reference' in reasons('As shown in the figure, find x.')
        assert 'visual_reference' in reasons('[asy]draw((0,0)--(1,1));[/asy]')
        assert not reasons('Find the area of the triangle with sides 3, 4, 5.')
        assert not reasons('A triangle is translated by vector (1,2). Find its area.')
        assert not reasons('Figure out the number of integers from 1 to 20.')
        assert not reasons('Find the area of the figure bounded by y=x and y=x^2.')
        print('Filter checks passed.')
        '''),
        markdown('## 3. Load a pinned snapshot from Hugging Face'),
        code('''
        source_revision = api.dataset_info(REPO_ID, revision=SOURCE_REVISION).sha
        head_before = api.dataset_info(REPO_ID).sha
        if source_revision != head_before:
            raise ValueError('The selected revision is not the current repository head. '
                             'Use main to avoid replacing newer data with an older snapshot.')
        splits = get_dataset_split_names(REPO_ID, config_name=SOURCE_CONFIG,
                                         revision=source_revision, token=HF_TOKEN)
        if splits != ['train']:
            raise ValueError(f'Expected only a train split; found {splits}. Review before modifying its configuration.')
        dataset = load_dataset(REPO_ID, name=SOURCE_CONFIG, split='train',
                               revision=source_revision, token=HF_TOKEN)
        if not {'id', 'problem'} <= set(dataset.column_names):
            raise ValueError('Expected id and problem columns.')
        if len(set(dataset['id'])) != len(dataset):
            raise ValueError('Duplicate source IDs; inspect the source first.')
        if any(not isinstance(x, str) or not x.strip() for x in dataset['problem']):
            raise ValueError('Found missing or invalid question text.')
        source_card_path = hf_hub_download(REPO_ID, 'README.md', repo_type='dataset',
                                           revision=source_revision, token=HF_TOKEN)
        original_card_text = Path(source_card_path).read_text()
        original_card = DatasetCard(original_card_text)
        print('Source commit:', source_revision)
        print('Input rows:', len(dataset))
        print('Columns:', dataset.column_names)
        '''),
        markdown('## 4. Filter, save, and inspect exact counts'),
        code('''
        keep_indices, drop_indices, decisions = [], [], []
        for index, row in enumerate(dataset.select_columns(['id', 'problem'])):
            evidence = text_exclusions(row['problem'])
            if evidence:
                drop_indices.append(index)
                decisions.append({'source_index': index, 'source_row_1_based': index + 1,
                    'id': row['id'], 'reasons': [x['reason'] for x in evidence], 'evidence': evidence})
            else:
                keep_indices.append(index)
        clean = dataset.select(keep_indices)
        excluded = dataset.select(drop_indices)
        if not len(clean):
            raise ValueError('The rules removed every row; refusing an empty upload.')
        clean_path = out / 'train-cleaned.parquet'
        excluded_path = out / 'excluded.parquet'
        clean.to_parquet(str(clean_path))
        excluded.to_parquet(str(excluded_path))
        (out / 'exclusions.jsonl').write_text(''.join(
            json.dumps(d, ensure_ascii=False) + '\\n' for d in decisions))
        (out / 'README.before.md').write_text(original_card_text)
        rule_spec = {name: {'pattern': p.pattern, 'flags': p.flags} for name, p in RULES.items()}
        (out / 'rules.json').write_text(json.dumps(rule_spec, ensure_ascii=False, indent=2))
        rule_hash = hashlib.sha256(json.dumps(rule_spec, sort_keys=True).encode()).hexdigest()
        counts = Counter(reason for d in decisions for reason in d['reasons'])
        sub_sources = dataset['sub_source'] if 'sub_source' in dataset.column_names else ['unknown']*len(dataset)
        summary = {'policy_version': VERSION, 'repo_id': REPO_ID, 'source_revision': source_revision,
            'source_config': SOURCE_CONFIG, 'rule_hash': rule_hash, 'input_rows': len(dataset),
            'removed_unique_rows': len(excluded), 'retained_rows': len(clean),
            'per_rule_counts': {name: counts[name] for name in RULES},
            'overlap_combinations': dict(Counter(' + '.join(d['reasons']) for d in decisions)),
            'per_rule_sub_sources': {name: dict(Counter(sub_sources[d['source_index']]
                for d in decisions if name in d['reasons'])) for name in RULES},
            'scope': 'Three text filters only; retained rows and reference answers unchanged.',
            'packages': {name: importlib.metadata.version(name) for name in
                         ['datasets', 'huggingface-hub', 'pyarrow', 'PyYAML']}}
        (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        assert len(clean) + len(excluded) == len(dataset)
        assert clean.features == dataset.features
        assert not set(keep_indices) & set(drop_indices)
        assert not any(text_exclusions(text) for text in clean['problem'])
        saved = load_dataset('parquet', data_files=str(clean_path), split='train')
        assert saved.features == clean.features
        for actual, expected in zip(saved, clean, strict=True):
            if actual != expected:
                raise ValueError('Parquet round-trip changed a retained row.')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('Overlapping categories do not count as separate removed rows.')
        for decision in decisions[:10]:
            print(json.dumps(decision, ensure_ascii=False))
        '''),
        markdown('''
        ## 5. Publish the cleaned default train split

        `PUSH_TO_HUB=True` publishes when this cell runs. Use False in settings for a
        preview-only run. One atomic commit publishes the cleaned Parquet, reports,
        exact rules, and updated dataset card. It refuses to overwrite a repository
        that changed after loading. Existing source files and repository visibility
        are preserved. A verification step downloads the published files and loads
        the new default train split to confirm its contents.
        '''),
        code('''
        def sha256_file(path):
            h = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    h.update(chunk)
            return h.hexdigest()

        run_key = source_revision[:12] + '-' + rule_hash[:12]
        remote_root = f'cleaning/text-only/{run_key}'
        remote_train = f'{remote_root}/train-cleaned.parquet'
        metadata = original_card.data.to_dict()
        metadata.pop('dataset_info', None)  # Recompute row counts/features for the new files.
        configurations = metadata.get('configs') or []
        configurations = [dict(c) for c in configurations if c['config_name'] != SOURCE_CONFIG]
        for config in configurations:
            config.pop('default', None)
        configurations.append({'config_name': SOURCE_CONFIG, 'default': True,
            'data_files': [{'split': 'train', 'path': remote_train}]})
        metadata['configs'] = configurations
        note = (f'## Text-only cleanup: {run_key}\\n\\n'
                f'The default train split now contains **{len(clean):,}** rows after removing '
                f'**{len(excluded):,}** rows matching translation instructions, Han characters, '
                f'or explicit visual-reference/image-markup rules. Category counts overlap.\\n\\n'
                f'Source commit: `{source_revision}`. Retained columns and gold answers are unchanged. '
                f'This is text screening, not mathematical validation of references. '
                f'Rules, exclusions, and summary: [`{remote_root}`]({remote_root}). '
                'Previous dataset-card content is retained below for provenance; its older counts '
                'may describe the source rather than the current default split.\\n\\n---\\n\\n')
        (out / 'README.md').write_text('---\\n' + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
                                     + '---\\n' + note + original_card.text)
        files_to_upload = {f'{remote_root}/{name}': out/name for name in
            ['train-cleaned.parquet', 'excluded.parquet', 'exclusions.jsonl', 'summary.json',
             'rules.json', 'README.before.md']}
        files_to_upload['README.md'] = out/'README.md'
        checksums = {remote: sha256_file(local) for remote, local in files_to_upload.items()}
        (out / 'manifest.json').write_text(json.dumps({'source_revision': source_revision,
            'rule_hash': rule_hash, 'files_sha256': checksums}, indent=2))
        files_to_upload[f'{remote_root}/manifest.json'] = out/'manifest.json'
        checksums[f'{remote_root}/manifest.json'] = sha256_file(out/'manifest.json')
        receipt_path = out/'upload_receipt.json'
        if not PUSH_TO_HUB:
            print('Preview only. Results saved in', out)
        elif not len(excluded):
            print('No rows match the filters; no Hub update is necessary.')
        else:
            current_head = api.dataset_info(REPO_ID).sha
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            if (receipt and receipt.get('source_revision') == source_revision and
                    receipt.get('checksums') == checksums and receipt.get('revision') == current_head):
                commit_sha = receipt['revision']
                print('This upload already exists; verifying it again.')
            else:
                if current_head != source_revision:
                    raise ValueError('Repository changed since loading. Reload and rerun cleanup; no files were uploaded.')
                commit = api.create_commit(REPO_ID, repo_type='dataset', parent_commit=source_revision,
                    commit_message='Remove translation artifacts, Han text, and visual-reference rows',
                    operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                                for remote, local in files_to_upload.items()])
                commit_sha = commit.oid
                receipt = {'repo': REPO_ID, 'source_revision': source_revision, 'revision': commit_sha,
                           'checksums': checksums, 'verified': False}
                receipt_path.write_text(json.dumps(receipt, indent=2))
            for remote, expected_sha in checksums.items():
                downloaded = hf_hub_download(REPO_ID, remote, repo_type='dataset',
                                             revision=commit_sha, token=HF_TOKEN)
                if sha256_file(downloaded) != expected_sha:
                    raise ValueError(f'Published checksum mismatch: {remote}')
            published = load_dataset(REPO_ID, name=SOURCE_CONFIG, split='train',
                                      revision=commit_sha, token=HF_TOKEN)
            if len(published) != len(clean) or published.features != clean.features:
                raise ValueError('Published default dataset has an unexpected row count or schema.')
            for actual, expected in zip(published, clean, strict=True):
                if actual != expected:
                    raise ValueError('Published default dataset differs from the cleaned data.')
            receipt['verified'] = True
            receipt_path.write_text(json.dumps(receipt, indent=2))
            print('Verified dataset:', f'https://huggingface.co/datasets/{REPO_ID}')
            print('Published commit:', commit_sha)
            print('Retained rows:', len(published))
        '''),
        markdown('## 6. Optional download of the cleaned file and reports'),
        code('''
        DOWNLOAD_RESULTS = False
        if DOWNLOAD_RESULTS:
            import shutil
            from google.colab import files
            archive = shutil.make_archive('/content/math-pool-text-cleanup-results', 'zip', out)
            files.download(archive)
        '''),
    ]


if __name__ == '__main__':
    write_notebook('clean_sft_pool_text_filters_and_upload.ipynb', cells())
