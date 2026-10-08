import pytest
from eval.final_answer import score_final


def grade(answer, gold, problem='Compute the value.', answer_type='other'):
    return score_final('\\boxed{' + answer + '}', gold, problem, answer_type=answer_type)


@pytest.mark.parametrize('answer,gold,problem,correct', [
 ('12:00','12:00','What time should the clock show?',True),
 ('12:10','12:00','What time should the clock show?',False),
 ('6:05','12:10','What time should the clock show?',False),
 ('8:36','08:36','At what time does the train arrive?',True),
 ('12 PM','12:00','What time should the clock show?',True),
 ('12:00 AM','00:00','What time should the clock show?',True),
 ('25:00','01:00','What time should the clock show?',False),
 ('8:60','09:00','What time should the clock show?',False),
 ('720','12:00','What time should the clock show?',False),
 ('2:6','1:3','Find the ratio of the areas.',True),
 ('3:1','1:3','Find the ratio of the areas.',False),
 ('1/3','1:3','Find the ratio of the areas.',True),
 ('12:10','6:05','Find the ratio of the areas.',True),
 ('1:0','1:0','Find the ratio.',False),
 ('12:10','6:05','Compute the answer.',False),
 ('12:10','6:05','Find the ratio of clock hands.',False),
 ('3^{23}>5^{15}','3^{23}>5^{15}','Compare the numbers.',True),
])
def test_time_ratio_and_comparison(answer,gold,problem,correct):
    assert grade(answer,gold,problem)[1] == correct


@pytest.mark.parametrize('answer', ['Answer: 12:00', r'The answer is $12:00$.', 'Work first.\n12:00'])
def test_unboxed_clock_extraction(answer):
    extracted, correct, audit = score_final(answer,'12:00','What time is it?',answer_type='other')
    assert (extracted,correct)==('12:00',True)
    assert audit['normalized_answer']=='720'


@pytest.mark.parametrize('gold,decimal', [
 ('(10010110110)_{2}',int('10010110110',2)),
 ('1061_{9}',int('1061',9)), ('1203_5',int('1203',5)),
 ('1162_7',int('1162',7)), ('1242_6',int('1242',6)),
 ('22_6',int('22',6)), ('100001_2',33), ('(10010)_{2}',18)])
def test_radix_forms(gold,decimal):
    assert grade(str(decimal),gold)[1]
    assert not grade(str(decimal+1),gold)[1]


@pytest.mark.parametrize('value',['102_2','19_9','1_1','1_37'])
def test_invalid_radix(value):
    assert not grade(value,value)[1]


@pytest.mark.parametrize('pct',['20','8.571','309.6','71.43','178','42','51.25','29.6296','17','38.4','55','11'])
def test_percent_scaling(pct):
    assert grade(pct+r'\%',pct+r'\\%')[1]
    assert grade(r'\frac{'+pct+'}{100}',pct+r'\%')[1]
    assert not grade(pct,pct+r'\%')[1]


@pytest.mark.parametrize('gold', ["['9']", "['140']"])
def test_singleton_numeric_string_list(gold):
    value=gold[2:-2]
    assert grade(value,gold)[1]
    assert grade(gold,value)[1]
    assert not grade(str(int(value)+1),gold)[1]


@pytest.mark.parametrize('bad',["['9','140']", "['__import__(\"os\")']", '62.08^244.98^2', '39-18=22'])
def test_ambiguous_or_invalid_not_repaired(bad):
    assert not grade(bad,'9')[1]
    assert not grade('9',bad)[1]


@pytest.mark.parametrize('answer,gold',[
 ('21','39-18=1+9+8+3'),
 ('0.7677568','0.59614528 + 0.17161152 = 0.7677568'),
 ('1','0.32+0.22+0.23+0.23=1'), ('-1','0-1'),
 ('2008!-1','2008! - 1'), ('21!-1','21!-1'), ('16!','16!'), ('10!','10!'),
 ('2^{2010}','2^{2010}'), ('2^{5947}','2^{5947}'), ('2^{81}','2^{81}'),
 ('11^{2020}','11^{2020}'), ('4(499^2-1)','4(499^{2}-1)'),
 ('2^{29}-2','2^{29} - 2'), ('9^3+2','9^{3}+2'), ('10^{10}','10^{10}'),
 ('2^{27}-2^{11}','2^{27} - 2^{11}'),
])
def test_user_numeric_examples(answer,gold):
    assert grade(answer,gold)[1]


def test_numeric_chains_do_not_ignore_wrong_work():
    assert not grade('39-18=22','22')[1]
    assert not grade('22','39-18=22')[1]


def test_other_only_scope_and_no_gold_hunting():
    assert not grade('12:00','12:00','What time is it?',answer_type='expression/text')[1]
    assert not grade("['9']",'9',answer_type='integer')[1]
    result=score_final('At first I considered 12:00. Final answer: 12:10',
                       '12:00','What time is it?',answer_type='other')
    assert not result[1]


@pytest.mark.parametrize('suffix',['.jsonl','.parquet'])
def test_regrade_saved_files_preserves_rows_and_generations(tmp_path,suffix):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pipeline.regrade_other import regrade_file
    rows=[]
    for index,kind in enumerate(['other','expression/text']):
        rows.append({'id':str(index),'problem':'What time is it?','gold_answer':'12:00',
          'answer_type':kind,'responses':[r'\boxed{12:00}']*8,'extracted_answers':['']*8,
          'correct':[False]*8,'num_correct':0,'response_tokens':[10]*8,'finish_reasons':['stop']*8,
          'source_note':'unchanged'})
    source=tmp_path/('input'+suffix)
    if suffix=='.jsonl':source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    else:pq.write_table(pa.Table.from_pylist(rows),source)
    before=source.read_bytes()
    report=regrade_file(source,tmp_path/'out')
    assert source.read_bytes()==before
    if suffix=='.jsonl':updated=[json.loads(s) for s in __import__('pathlib').Path(report['output']).read_text().splitlines()]
    else:
        table=pq.read_table(report['output'])
        assert table.schema==pq.read_schema(source)
        updated=table.to_pylist()
    assert updated[0]['num_correct']==8
    assert updated[1]==rows[1]
    for key in rows[0].keys()-{'correct','extracted_answers','num_correct'}:
        assert updated[0][key]==rows[0][key]
    assert report['rows']==2 and report['other_rows']==1 and report['dropped_rows']==0
    assert report['grading_errors']==0 and report['changed_flags']==8
