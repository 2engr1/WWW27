import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import enterprise_rag as er

class Tests(unittest.TestCase):
    def doc(self, n=0, body='content'):
        return {'dataset_doc_uuid':f'dsid_{n:032x}', 'title_field_name':'title',
                'content_field_names':[body], 'title':'Title',body:'Evidence',
                'created_at':'2026-01-01','dataset_noise_document':True}
    def test_content_and_body_per_doc(self):
        for key in ['content','body']:
            row=er.normalize_doc(self.doc(body=key),Path('confluence/sample.json'))
            self.assertEqual(row['text'],'Title\n\nEvidence')
            self.assertEqual(row['extracted_fields']['content'],[key])
            self.assertEqual(row['metadata']['created_at'],'2026-01-01')
            self.assertNotIn('dataset_noise_document',row['text'])
    def test_alternative_title_and_structured_messages(self):
        obj=self.doc()
        obj.update(title_field_name='subject',subject='An email',content_field_names=['summary','messages'],summary='Summary',messages=[{'speaker':'A','body':'Proof'}])
        row=er.normalize_doc(obj,Path('gmail/f.json'))
        self.assertTrue(row['text'].startswith('An email'))
        self.assertIn('Proof',row['text'])
        self.assertIn('Summary',row['text'])
    def test_missing_declared_field_rejected(self):
        obj=self.doc();obj['content_field_names']=['missing']
        with self.assertRaisesRegex(ValueError,'Declared content field missing'):
            er.normalize_doc(obj,Path('slack/f.json'))
    def test_title_only_and_unknown_source_rejected(self):
        obj=self.doc();obj['content']=''
        with self.assertRaisesRegex(ValueError,'no text'):
            er.normalize_doc(obj,Path('slack/f.json'))
        with self.assertRaisesRegex(ValueError,'Unknown source'):
            er.normalize_doc(self.doc(),Path('other/f.json'))
    def setup_corpus(self, root):
        docs=root/'sources'
        for n in range(30):
            source='slack' if n<20 else 'gmail'
            d=docs/source;d.mkdir(parents=True,exist_ok=True)
            (d/f'{n}.json').write_text(json.dumps(self.doc(n,'body' if n%2 else 'content')))
        qs=[dict(question_id='qst_1',question_type='completeness',question='Q',expected_doc_ids=[f'dsid_{0:032x}',f'dsid_{21:032x}'],gold_answer='A',answer_facts=['F']),
            dict(question_id='qst_2',question_type='high_level',question='Q2',expected_doc_ids=[],gold_answer='A2',answer_facts=['F2'])]
        er.write_jsonl(root/'questions.jsonl',qs)
        return argparse.Namespace(docs=str(docs),questions=str(root/'questions.jsonl'),out=str(root/'processed'),sizes=[5,10,20],seed=42,anchors=None,format="auto",json_id_field=None,json_text_fields=None,json_source_field=None,source_map=None,allow_unknown_source=False)
    def test_ladder_contract_reproducibility(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            base=root/'processed'
            sets=[set((base/'tiers'/t/'doc_ids.txt').read_text().splitlines()) for t in ['5','10','20','full']]
            self.assertEqual([len(s) for s in sets],[5,10,20,30])
            self.assertTrue(all(x<y for x,y in zip(sets,sets[1:])))
            self.assertTrue({f'dsid_{0:032x}',f'dsid_{21:032x}'}<=sets[0])
            self.assertNotIn('gold_answer',next(er.read_jsonl(base/'queries.jsonl')))
            self.assertEqual(len(list(er.read_jsonl(base/'queries_retrieval.jsonl'))),1)
            self.assertEqual(len(list(er.read_jsonl(base/'corpus.jsonl'))),30)
            a.out=str(root/'processed2')
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            self.assertEqual((base/'order.txt').read_bytes(),(root/'processed2/order.txt').read_bytes())
    def test_missing_gold_and_conflicting_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            (root/'sources/slack/0.json').unlink()
            with self.assertRaisesRegex(ValueError,'Missing 1 gold'):er.build(a)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            obj=self.doc(0);obj['content']='Different evidence'
            (root/'sources/slack/duplicate.json').write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'Conflicting duplicate ID'):er.build(a)
    def test_audit_samples_sources_without_loading_uuid_index(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            (root/'uuid_index.json').write_text('{}')
            a.per_source=2;a.full=False
            buf=io.StringIO()
            with contextlib.redirect_stdout(buf):er.audit(a)
            r=json.loads(buf.getvalue())
            self.assertEqual(r['checked_documents'],4)
            self.assertFalse(r['is_full_scan'])


class EvaluationTests(unittest.TestCase):
    setup_corpus = Tests.setup_corpus
    doc = Tests.doc
    def test_multigold_recall_missing_and_duplicate_ranks(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            er.write_jsonl(root/'pred.jsonl',[{'question_id':'qst_1','document_ids':[f'dsid_{0:032x}',f'dsid_{0:032x}',f'dsid_{21:032x}']}])
            b=argparse.Namespace(data=a.out,tier='5',predictions=str(root/'pred.jsonl'),ks=[1,2],out=str(root/'scores.json'))
            with contextlib.redirect_stdout(io.StringIO()):er.recall(b)
            r=json.loads(Path(b.out).read_text())
            self.assertEqual(r['overall']['recall@1'],.5)
            self.assertEqual(r['overall']['recall@2'],1)
            self.assertEqual(r['overall']['all_gold@1'],0)
            er.write_jsonl(root/'pred.jsonl',[])
            with contextlib.redirect_stdout(io.StringIO()):er.recall(b)
            r=json.loads(Path(b.out).read_text());self.assertEqual(r['missing_predictions'],1)
            self.assertEqual(r['overall']['recall@2'],0)
    def test_out_of_tier_rejected_and_materialize(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            order=(Path(a.out)/'order.txt').read_text().splitlines()
            er.write_jsonl(root/'pred.jsonl',[{'question_id':'qst_1','document_ids':[order[-1]]}])
            with self.assertRaisesRegex(ValueError,'out-of-tier'):
                er.recall(argparse.Namespace(data=a.out,tier='5',predictions=str(root/'pred.jsonl'),ks=[1],out=str(root/'s.json')))
            with contextlib.redirect_stdout(io.StringIO()):
                er.materialize(argparse.Namespace(data=a.out,tier='5',out=str(root/'five.jsonl')))
            self.assertEqual(len(list(er.read_jsonl(root/'five.jsonl'))),5)
    def test_judge_scope_and_resume_no_api_calls(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            er.write_jsonl(root/'pred.jsonl',[{'question_id':'qst_1','answer':'A','document_ids':[]}])
            b=argparse.Namespace(data=a.out,predictions=str(root/'pred.jsonl'),out=str(root/'quality.jsonl'),scope='retrieval',model='mock',endpoint='https://example.test/chat/completions',votes=1)
            original=er.api_judge
            er.api_judge=lambda a,m: {'correct':True} if 'gold_answer' in json.loads(m[1]['content']) else {'supported':True}
            try:
                with contextlib.redirect_stdout(io.StringIO()):er.judge(b);er.judge(b)
                rows=list(er.read_jsonl(b.out));self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['correctness'],1);self.assertEqual(rows[0]['completeness'],1)
                b.model='changed'
                with self.assertRaisesRegex(ValueError,'config mismatch'):er.judge(b)
            finally:er.api_judge=original
    def test_txt_build_compatible(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            txt=root/'txt/slack';txt.mkdir(parents=True)
            for n in range(30):(txt/f'dsid_{n:032x}_name.txt').write_text(f'Title\nEvidence {n}')
            a.docs=str(txt.parent)
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            self.assertEqual(len(list(er.read_jsonl(Path(a.out)/'corpus.jsonl'))),30)

class ExportTests(unittest.TestCase):
    setup_corpus = Tests.setup_corpus
    doc = Tests.doc
    def test_independent_exports_match_tiers_and_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            base=Path(a.out)
            for label in ['5','10','20','full']:
                ids=(base/'tiers'/label/'doc_ids.txt').read_text().splitlines()
                path=base/'exports'/f'corpus_{label}.jsonl'
                rows=list(er.read_jsonl(path))
                self.assertEqual([r['doc_id'] for r in rows],ids)
                stats=json.loads((base/'tiers'/label/'stats.json').read_text())
                self.assertEqual(er.checksum(path),stats['corpus_sha256'])
            self.assertEqual((base/'corpus.jsonl').read_bytes(),(base/'exports/corpus_full.jsonl').read_bytes())
    def test_manifests_only_skips_exports(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root);a.manifests_only=True
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            base=Path(a.out)
            self.assertFalse((base/'exports').exists())
            self.assertTrue((base/'corpus.jsonl').exists())
            self.assertFalse(json.loads((base/'manifest.json').read_text())['independent_corpora'])


class EmptyContentTests(unittest.TestCase):
    setup_corpus = Tests.setup_corpus
    doc = Tests.doc
    def test_retained_title_and_gold_diagnostic(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            path=root/'sources/slack/0.json'
            obj=json.loads(path.read_text());obj['content']='';path.write_text(json.dumps(obj))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):er.build(a)
            base=Path(a.out)
            rows=list(er.read_jsonl(base/'empty_content_documents.jsonl'))
            self.assertEqual(len(rows),1);self.assertTrue(rows[0]['is_gold'])
            self.assertEqual(rows[0]['gold_question_ids'],['qst_1'])
            corpus=list(er.read_jsonl(base/'corpus.jsonl'))
            self.assertEqual(len(corpus),30)
            title_only=next(r for r in corpus if r['doc_id']==obj['dataset_doc_uuid'])
            self.assertEqual(title_only['text'],'Title')
            self.assertEqual(title_only['content_status'],'title_only')
            self.assertEqual(json.loads((base/'manifest.json').read_text())['title_only_gold_documents'],1)
    def test_empty_lists_and_no_title(self):
        for val in [None,'',[],{}]:
            obj=self.doc();obj['content']=val
            self.assertEqual(er.normalize_doc(obj,Path('slack/f.json'),'keep-title')['text'],'Title')
            with self.assertRaisesRegex(ValueError,'no text'):
                er.normalize_doc(obj,Path('slack/f.json'),'error')
        obj=self.doc();obj['title']='';obj['content']=''
        with self.assertRaisesRegex(ValueError,'Both declared'):
            er.normalize_doc(obj,Path('slack/f.json'),'keep-title')


class QuestionSelectionTests(unittest.TestCase):
    setup_corpus = Tests.setup_corpus
    doc = Tests.doc
    def test_outputs_keep_only_retrieval_qa(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            with contextlib.redirect_stdout(io.StringIO()):er.build(a)
            base=Path(a.out)
            for name in ['queries.jsonl','queries_retrieval.jsonl','gold.jsonl']:
                rows=list(er.read_jsonl(base/name))
                self.assertEqual([r['question_id'] for r in rows],['qst_1'])
            m=json.loads((base/'manifest.json').read_text())
            self.assertEqual(m['input_questions'],2);self.assertEqual(m['questions'],1)
            self.assertEqual(m['retrieval_questions'],1);self.assertEqual(m['excluded_questions'],1)
    def test_no_reference_answer_is_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            qs=list(er.read_jsonl(a.questions))
            additional=dict(qs[0]);additional.update(question_id='qst_no_answer',gold_answer='   ')
            selected,excluded=er.select_retrieval_qa(qs+[additional])
            self.assertEqual([q['question_id'] for q in selected],['qst_1'])
            self.assertEqual(len(excluded),2)
            self.assertIn('no_reference_answer',excluded[-1]['reasons'])


class IDConflictTests(unittest.TestCase):
    setup_corpus = Tests.setup_corpus
    doc = Tests.doc
    def test_indexed_canonical_keeps_id_and_noise_keeps_content(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            noise=root/'sources/confluence/noise.json';noise.parent.mkdir()
            obj=self.doc(0);obj.update(title='Noise version',content='Changed evidence',dataset_noise_document=True)
            noise.write_text(json.dumps(obj))
            index=root/'uuid_index.json';er.write_json(index,{obj['dataset_doc_uuid']:'slack/0.json'})
            a.uuid_index=str(index)
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):er.build(a)
            base=Path(a.out);rows=list(er.read_jsonl(base/'corpus.jsonl'))
            self.assertEqual(len(rows),31)
            canonical=next(r for r in rows if r['doc_id']==obj['dataset_doc_uuid'])
            self.assertEqual(canonical['original_path'],'slack/0.json')
            remaps=list(er.read_jsonl(base/'document_id_remapping.jsonl'));self.assertEqual(len(remaps),1)
            variant=next(r for r in rows if r['doc_id']==remaps[0]['new_doc_id'])
            self.assertIn('Changed evidence',variant['text'])
            self.assertEqual(variant['original_doc_id'],obj['dataset_doc_uuid'])
            self.assertTrue(remaps[0]['is_original_id_gold'])
            a.out=str(root/'second')
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):er.build(a)
            self.assertEqual((base/'document_id_remapping.jsonl').read_bytes(),(Path(a.out)/'document_id_remapping.jsonl').read_bytes())
    def test_metadata_only_deduplicates_and_retains_variants(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            obj=self.doc(0);obj['created_at']='Other date'
            (root/'sources/slack/duplicate.json').write_text(json.dumps(obj))
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):er.build(a)
            base=Path(a.out);rows=list(er.read_jsonl(base/'corpus.jsonl'))
            self.assertEqual(len(rows),30)
            chosen=next(r for r in rows if r['doc_id']==obj['dataset_doc_uuid'])
            self.assertEqual(chosen['duplicate_variants'][0]['metadata']['created_at'],'Other date')
            self.assertEqual(len(list(er.read_jsonl(base/'duplicate_documents.jsonl'))),1)
    def test_missing_index_records_conflict_before_no_export(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=self.setup_corpus(root)
            obj=self.doc(0);obj['content']='Different'
            (root/'sources/slack/duplicate.json').write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'unresolved groups'):er.build(a)
            base=Path(a.out)
            self.assertEqual(next(er.read_jsonl(base/'conflicting_documents.jsonl'))['status'],'unresolved')
            self.assertFalse((base/'exports').exists())


class EmptyFieldNameTests(unittest.TestCase):
    doc = Tests.doc
    def test_empty_string_key_preserves_both_content_values(self):
        obj=self.doc();obj.update(title_field_name='channel',channel='product',content_field_names=['text',''],text='Slack discussion',**{'':'Additional evidence'})
        row=er.normalize_doc(obj,Path('slack/product/f.json'),'keep-title')
        self.assertEqual(row['text'],'product\n\nSlack discussion\n\nAdditional evidence')
        self.assertEqual(row['extracted_fields']['content'],['text',''])
        self.assertNotIn('',row['metadata'])
        obj['']=''
        self.assertIn('Slack discussion',er.normalize_doc(obj,Path('slack/product/f.json'))['text'])
    def test_missing_empty_key_and_nonstring_keys_still_fail(self):
        obj=self.doc();obj['content_field_names']=['content','']
        with self.assertRaisesRegex(ValueError,'Declared content field missing'):
            er.normalize_doc(obj,Path('slack/f.json'))
        obj['content_field_names']=['content',None]
        with self.assertRaisesRegex(ValueError,'list of strings'):
            er.normalize_doc(obj,Path('slack/f.json'))

if __name__=='__main__':unittest.main()
