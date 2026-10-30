#!/usr/bin/env python3
"""Read-only validator for EnterpriseRAG six-tier corpora. Python 3.10+, standard library.
Example:
  python validate_enterprise_rag.py --data /path/cache_date_v231 \
    --original-questions /path/EnterpriseRAG-Bench-main/questions.jsonl \
    --source-root /path/EnterpriseRAG-Bench-main/generated_data/sources \
    --uuid-index /path/EnterpriseRAG-Bench-main/generated_data/uuid_index.json \
    --source-check all --report validation_report.json
Generated files are all scanned. Source-check gold checks gold and remapped records;
source-check all also checks every source body, duplicate alias and file coverage.
Exit 0 = structurally valid; 2 = errors; 3 = gold evidence needs manual review.
A passing report cannot prove the semantic correctness/completeness of gold labels.
"""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import sqlite3
import sys
import tempfile

VERSION = '1.0.1'


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def evidence_hash(row):
    return hashlib.sha256(json.dumps([row['title'],row['text']],ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            obj=json.loads(line)
            if not isinstance(obj,dict):raise ValueError(f'{path}:{line_no}: JSON object required')
            yield obj


def safe_path(root,rel):
    if not isinstance(rel,str):raise ValueError('Path must be a string')
    path=(root/rel).resolve()
    if path==root.resolve() or root.resolve() not in path.parents:raise ValueError(f'Path escapes source root: {rel}')
    return path


def text_value(v):
    if v is None or v==[] or v=={}:return ''
    return v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,sort_keys=True)


def source_evidence(obj):
    title_key=obj['title_field_name'];fields=obj['content_field_names']
    if isinstance(fields,str):fields=[fields]
    if not isinstance(fields,list) or not fields or not all(isinstance(k,str) for k in fields):raise ValueError('Invalid content declaration')
    fields=list(dict.fromkeys(fields))
    title=text_value(obj[title_key])
    parts=[text_value(obj[k]) for k in fields if k!=title_key]
    text='\n\n'.join([title]+parts) if any(p.strip() for p in parts) else title
    excluded=set(fields)|{title_key,'dataset_doc_uuid','title_field_name','content_field_names'}
    return title,text,{k:v for k,v in obj.items() if k not in excluded}, {'title':title_key,'content':fields}


class Validator:
    def __init__(self,a):
        self.a=a;self.base=Path(a.data).resolve()
        self.counts=Counter();self.messages=[];self.checks={};self.stats={}
        self.gold=set();self.gold_qids={};self.remaps={};self.raw_index={};self.source_checked=0
    def issue(self,level,check,message):
        self.counts[level]+=1
        if len(self.messages)<self.a.max_issues:self.messages.append({'level':level,'check':check,'message':str(message)})
    def require(self,condition,check,message):
        if not condition:self.issue('ERROR',check,message)
        return condition
    def json_file(self,rel):
        return json.loads((self.base/rel).read_text(encoding='utf-8'))
    def hash_check(self,path,expected,check):
        self.require(isinstance(expected,str) and sha256(path)==expected,check,f'Hash mismatch/missing: {path}')
    def check_queries(self):
        gold_rows=list(rows(self.base/'gold.jsonl'));gold_by_id={}
        for q in gold_rows:
            qid=q.get('question_id')
            self.require(isinstance(qid,str) and qid not in gold_by_id,'questions',f'Duplicate/invalid qid: {qid}')
            gold_by_id[qid]=q
            ids=q.get('expected_doc_ids')
            valid=isinstance(ids,list) and bool(ids) and all(isinstance(x,str) for x in ids)
            self.require(valid,'questions',f'{qid}: no valid nonempty gold ID list')
            self.require(isinstance(q.get('gold_answer'),str) and bool(q['gold_answer'].strip()),'questions',f'{qid}: empty reference answer')
            self.require(isinstance(q.get('answer_facts'),list),'questions',f'{qid}: missing fact list')
            if not q.get('answer_facts'):self.issue('REVIEW','answer_facts',f'{qid}: no facts for completeness evaluation')
            if valid:
                if len(ids)!=len(set(ids)):
                    self.issue('WARNING','duplicate_gold_ids',f'{qid}: repeated gold IDs; evaluated as a unique document set; original annotation retained')
                self.gold.update(ids)
                for d in set(ids):self.gold_qids.setdefault(d,[]).append(qid)
        self.require(len(gold_by_id)==self.a.expected_questions,'questions',f'Expected {self.a.expected_questions}, got {len(gold_by_id)}')
        self.require(len(self.gold)==self.a.expected_gold,'questions',f'Expected {self.a.expected_gold} gold docs, got {len(self.gold)}')
        expected_public=[{k:q[k] for k in ('question_id','question_type','question')} for q in gold_rows]
        for name in ['queries.jsonl','queries_retrieval.jsonl']:
            public=list(rows(self.base/name))
            self.require(public==expected_public,'query_alignment',f'{name}: content/order differs from gold public fields or leaked fields present')
        expected_qrels={(q['question_id'],d) for q in gold_rows for d in q['expected_doc_ids']}
        actual=[]
        for n,line in enumerate((self.base/'qrels.tsv').read_text().splitlines(),1):
            parts=line.split()
            if not self.require(len(parts)==4 and parts[1]=='0' and parts[3]=='1','qrels',f'Invalid row {n}'):continue
            actual.append((parts[0],parts[2]))
        self.require(len(actual)==len(set(actual)) and set(actual)==expected_qrels,'qrels','Qrels differs from gold or contains duplicates')
        self.hash_check(self.base/'gold.jsonl',self.manifest.get('questions_sha256'),'gold_hash')
        self.require(self.manifest.get('questions')==len(gold_rows) and self.manifest.get('retrieval_questions')==len(gold_rows),'manifest','Question counts disagree')
        self.require(self.manifest.get('gold_union_count')==len(self.gold),'manifest','Gold count disagrees')
        if self.a.original_questions:
            original=list(rows(self.a.original_questions))
            selected=[q for q in original if q.get('expected_doc_ids') and isinstance(q.get('gold_answer'),str) and q['gold_answer'].strip()]
            self.require(gold_rows==selected,'original_questions','Output annotations changed, lost, added, or reordered vs original selected questions')
            self.require(self.manifest.get('input_questions')==len(original),'manifest','Original question count mismatch')
            self.require(self.manifest.get('excluded_questions')==len(original)-len(selected),'manifest','Excluded count mismatch')
            selection=self.json_file('question_selection.json')
            selected_ids={q['question_id'] for q in selected}
            excluded_ids=[q['question_id'] for q in original if q['question_id'] not in selected_ids]
            self.require(selection.get('selected_questions')==len(selected) and selection.get('input_questions')==len(original),'selection','Selection report counts mismatch')
            self.require([q['question_id'] for q in selection.get('excluded_ids_and_reasons',[])]==excluded_ids,'selection','Excluded IDs mismatch')
        else:self.issue('WARNING','original_questions','Original questions not supplied; gold provenance was not verified')
        self.stats['questions']=len(gold_rows);self.stats['gold_documents']=len(self.gold)
        self.stats['question_types']=dict(Counter(q['question_type'] for q in gold_rows))
    def check_tiers(self):
        self.order=(self.base/'order.txt').read_text().splitlines()
        self.require(len(self.order)==len(set(self.order)),'order','Duplicate IDs in order.txt')
        self.require(len(self.order)==self.manifest.get('documents'),'order','Full count mismatch')
        self.hash_check(self.base/'order.txt',self.manifest.get('order_sha256'),'order_hash')
        expected_labels=[str(n) for n in self.a.sizes]+['full']
        tiers=self.manifest.get('tiers',[])
        self.require([t.get('tier') for t in tiers]==expected_labels,'tiers','Expected exactly the six prescribed tier labels/order')
        self.tiers=[];previous=set();previous_n=0
        for t in tiers:
            label=t['tier'];n=int(t['documents']);path=self.base/'tiers'/label
            ids=(path/'doc_ids.txt').read_text().splitlines();s=set(ids);stored=self.json_file(f'tiers/{label}/stats.json')
            self.require(stored==t,'tier_stats',f'{label}: stats.json differs from manifest tier')
            self.require(len(ids)==len(s)==n,'tier_size',f'{label}: size/duplicate mismatch')
            target=len(self.order) if label=='full' else int(label)
            self.require(n==target,'tier_size',f'{label}: wrong target size')
            self.require(ids==self.order[:n],'nested_prefix',f'{label}: not the same global-order prefix')
            self.require(previous<s and n>previous_n,'nested_sets',f'{label}: not strictly nested')
            self.require(self.gold<=s,'gold_coverage',f'{label}: missing gold {sorted(self.gold-s)[:8]}')
            self.require(t.get('all_gold_present') is True and t.get('gold_documents')==len(self.gold),'gold_coverage',f'{label}: gold statistics wrong')
            self.hash_check(path/'doc_ids.txt',t.get('doc_ids_sha256'),'tier_id_hash')
            self.tiers.append(t);previous=s;previous_n=n
        if self.tiers:self.require(previous==set(self.order),'full','Full tier is not full corpus')
        self.stats['tier_sizes']={t['tier']:t['documents'] for t in self.tiers}
    def load_diagnostics(self):
        remaps=list(rows(self.base/'document_id_remapping.jsonl'))
        for r in remaps:
            new=r['new_doc_id'];old=r['original_doc_id']
            self.require(new not in self.remaps and new!=old,'remaps',f'Duplicate/unchanged remap: {new}')
            expected='dsid_'+hashlib.sha256((old+'\0'+r['evidence_sha256']).encode()).hexdigest()[:32]
            self.require(new==expected,'remaps',f'{new}: unstable derived ID')
            self.require(new not in self.gold,'remaps',f'New noise ID entered fixed gold: {new}')
            self.require(r.get('is_original_id_gold')==(old in self.gold),'remaps',f'{new}: incorrect gold flag')
            self.remaps[new]=r
        conflicts=list(rows(self.base/'conflicting_documents.jsonl'))
        for c in conflicts:
            self.require(c.get('status')=='resolved_by_uuid_index','conflicts',f'Unresolved collision: {c.get("original_doc_id")}')
            if c.get('is_gold'):self.issue('REVIEW','gold_conflicts',f'Gold ID {c["original_doc_id"]} has multiple different-evidence versions; verify reference answer against indexed copy')
        self.conflicts=conflicts
        self.empty_report=list(rows(self.base/'empty_content_documents.jsonl'))
        self.duplicates=list(rows(self.base/'duplicate_documents.jsonl'))
        self.require(len(remaps)==self.manifest.get('remapped_documents'),'diagnostics','Remapping count mismatch')
        self.require(len(conflicts)==self.manifest.get('resolved_id_conflicts'),'diagnostics','Conflict count mismatch')
        if self.a.uuid_index:
            raw=json.loads(Path(self.a.uuid_index).read_text(encoding='utf-8-sig'))
            for k,v in raw.items():
                v=v.replace('\\','/')
                for prefix in ('generated_data/sources/','sources/'):
                    if v.startswith(prefix):v=v[len(prefix):];break
                self.raw_index[k]=Path(v).as_posix()
            self.require(sha256(self.a.uuid_index)==self.manifest.get('uuid_index_sha256'),'uuid_index','Original UUID index hash mismatch')
            for c in conflicts:self.require(self.raw_index.get(c['original_doc_id'])==c.get('canonical_path'),'uuid_index',f'{c["original_doc_id"]}: wrong canonical path')
        elif conflicts:self.issue('WARNING','uuid_index','UUID index not supplied; resolution provenance not independently verified')
    def verify_source(self,row):
        root=Path(self.a.source_root).resolve()
        path=safe_path(root,row['original_path']);obj=json.loads(path.read_text(encoding='utf-8-sig'))
        title,text,metadata,fields=source_evidence(obj)
        raw_id=row.get('original_doc_id',row['doc_id'])
        self.require(obj.get('dataset_doc_uuid')==raw_id,'source_id',f'{path}: raw UUID differs')
        self.require(title==row['title'] and text==row['text'],'source_content',f'{path}: title/body lost or changed')
        self.require(metadata==row.get('metadata') and fields==row.get('extracted_fields'),'source_metadata',f'{path}: metadata/extraction fields changed')
        self.source_checked+=1
        if self.a.source_check=='all':
            for alias in row.get('duplicate_variants',[]):
                ap=safe_path(root,alias['original_path']);ao=json.loads(ap.read_text(encoding='utf-8-sig'))
                at,ax,am,af=source_evidence(ao)
                self.require(ao.get('dataset_doc_uuid')==raw_id and at==title and ax==text,'duplicate_source',f'{ap}: merged alias is different evidence')
                self.require(am==alias['metadata'] and af==alias['extracted_fields'],'duplicate_source',f'{ap}: alias metadata changed')
                self.source_checked+=1
    def check_corpus(self,db):
        db.execute('CREATE TABLE docs (id TEXT PRIMARY KEY, row_hash TEXT, evidence TEXT, source TEXT, path TEXT, raw_id TEXT, status TEXT)')
        db.execute('CREATE TABLE paths (path TEXT PRIMARY KEY)')
        sources=Counter();empty=[];n=0;aliases=0
        relevant=set(self.gold)|set(self.remaps)|{r['original_doc_id'] for r in self.remaps.values()}
        for n,row in enumerate(rows(self.base/'corpus.jsonl'),1):
            if n%10000==0:print(f'Checked corpus {n:,}',file=sys.stderr,flush=True);db.commit()
            required=('doc_id','title','text','source_type','original_path','metadata','extracted_fields')
            if not self.require(all(k in row for k in required),'corpus_schema',f'Row {n}: missing fields'):continue
            doc_id=row['doc_id']
            if not self.require(isinstance(doc_id,str) and re.fullmatch(r'dsid_[0-9a-fA-F]+',doc_id) is not None,'corpus_id',f'Invalid doc ID at row {n}'):continue
            if not isinstance(row['text'],str) or not isinstance(row['title'],str):self.issue('ERROR','corpus_schema',f'{doc_id}: nonstring text');continue
            self.require(bool(row['text'].strip()),'empty_text',f'{doc_id}: completely empty text')
            self.require(n<=len(self.order) and doc_id==self.order[n-1],'corpus_order',f'Row {n}: ID differs from order.txt')
            try:db.execute('INSERT INTO docs VALUES(?,?,?,?,?,?,?)',(doc_id,canonical_hash(row),evidence_hash(row),row['source_type'],row['original_path'],row.get('original_doc_id',doc_id),row.get('content_status')))
            except sqlite3.IntegrityError:self.issue('ERROR','duplicate_id',f'Duplicate corpus ID: {doc_id}')
            for rel in [row['original_path']]+[v['original_path'] for v in row.get('duplicate_variants',[])]:
                try:db.execute('INSERT INTO paths VALUES(?)',(rel,))
                except sqlite3.IntegrityError:self.issue('ERROR','duplicate_path',f'Original path represented twice: {rel}')
            aliases+=len(row.get('duplicate_variants',[]));sources[row['source_type']]+=1
            self.require(row['original_path'].split('/')[0]==row['source_type'],'source_type',f'{doc_id}: source inconsistent with original path')
            if row.get('content_status')=='title_only':empty.append(doc_id);self.require(row['text']==row['title'],'empty_report',f'{doc_id}: title-only text differs')
            if self.a.source_root and (self.a.source_check=='all' or doc_id in relevant):
                try:self.verify_source(row)
                except Exception as e:self.issue('ERROR','source_read',f'{doc_id}: {e}')
        db.commit()
        self.hash_check(self.base/'corpus.jsonl',self.manifest.get('corpus_sha256'),'corpus_hash')
        self.require(n==len(self.order) and n==self.manifest.get('documents'),'corpus_count','Full corpus count differs')
        self.require(dict(sources)==self.manifest.get('source_counts'),'sources','Full source counts differ')
        self.require(aliases==self.manifest.get('duplicates_identical_evidence'),'duplicates','Alias count differs')
        self.require(sum(len(r['other_variants']) for r in self.duplicates)==aliases,'duplicates','Duplicate diagnostic count differs')
        for d in self.gold:self.require(db.execute('SELECT 1 FROM docs WHERE id=?',(d,)).fetchone() is not None,'gold_coverage',f'Gold absent from full corpus: {d}')
        self.require(set(empty)=={r['doc_id'] for r in self.empty_report} and len(empty)==len(self.empty_report),'empty_report','Title-only report IDs/count mismatch')
        self.require(len(empty)==self.manifest.get('title_only_documents'),'empty_report','Title-only manifest count mismatch')
        gold_empty=set(empty)&self.gold
        self.require(len(gold_empty)==self.manifest.get('title_only_gold_documents'),'empty_report','Title-only gold count mismatch')
        for r in self.empty_report:
            self.require(r.get('is_gold')==(r['doc_id'] in self.gold) and r.get('gold_question_ids')==self.gold_qids.get(r['doc_id'],[]),'empty_report',f'{r["doc_id"]}: incorrect gold associations')
        if gold_empty:self.issue('REVIEW','gold_empty',f'{len(gold_empty)} gold documents have no declared body: {sorted(gold_empty)[:8]}')
        if set(empty)-self.gold:self.issue('WARNING','title_only',f'{len(set(empty)-self.gold)} non-gold documents contain only titles')
        for new,r in self.remaps.items():
            actual=db.execute('SELECT evidence,path,raw_id FROM docs WHERE id=?',(new,)).fetchone()
            self.require(actual==(r['evidence_sha256'],r['path'],r['original_doc_id']),'remap_content',f'{new}: remapped content/path mismatch')
            old=db.execute('SELECT path FROM docs WHERE id=?',(r['original_doc_id'],)).fetchone()
            self.require(old==(r['canonical_path'],),'canonical_id',f'{r["original_doc_id"]}: canonical original ID missing/wrong path')
        self.require(n+aliases==self.manifest.get('input_document_files'),'source_coverage','Input file count vs retained records/aliases mismatch')
        if self.a.source_root and self.a.source_check=='all':
            original_n=0
            for p in Path(self.a.source_root).resolve().rglob('*.json'):
                original_n+=1;rel=p.relative_to(Path(self.a.source_root).resolve()).as_posix()
                self.require(db.execute('SELECT 1 FROM paths WHERE path=?',(rel,)).fetchone() is not None,'source_coverage',f'Original file not retained: {rel}')
            path_n=db.execute('SELECT COUNT(*) FROM paths').fetchone()[0]
            self.require(original_n==path_n,'source_coverage','Source file inventory differs from retained paths')
        self.stats.update(documents=n,source_counts=dict(sources),title_only_documents=len(empty),title_only_gold_documents=len(gold_empty),remapped_documents=len(self.remaps),source_records_checked=self.source_checked)
    def check_catalog(self,db):
        path=self.base/'catalog.sqlite'
        if not path.is_file():self.issue('ERROR','catalog','catalog.sqlite missing');return
        catalog=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        try:
            integrity=catalog.execute('PRAGMA quick_check').fetchall()
            self.require(integrity==[('ok',)],'catalog','SQLite integrity check failed')
            n=0
            for doc_id,source,payload in catalog.execute('SELECT id,source,payload FROM docs'):
                n+=1;row=json.loads(payload)
                stored=db.execute('SELECT row_hash FROM docs WHERE id=?',(doc_id,)).fetchone()
                self.require(stored is not None and stored[0]==canonical_hash(row) and row['doc_id']==doc_id and row['source_type']==source,'catalog',f'{doc_id}: catalog disagrees with corpus')
            self.require(n==self.stats['documents'],'catalog','Catalog document count differs')
        finally:catalog.close()

    def check_exports(self,db):
        if not self.manifest.get('independent_corpora'):self.issue('ERROR','exports','Independent six corpus files were not generated');return
        results=[]
        for t in self.tiers:
            label=t['tier'];path=safe_path(self.base,t.get('corpus_file',f'exports/corpus_{label}.jsonl'))
            sources=Counter();n=0
            for n,row in enumerate(rows(path),1):
                doc_id=row.get('doc_id')
                self.require(n<=t['documents'] and doc_id==self.order[n-1],'export_order',f'{label}: row {n} is not expected prefix')
                full=db.execute('SELECT row_hash FROM docs WHERE id=?',(doc_id,)).fetchone()
                self.require(full is not None and full[0]==canonical_hash(row),'export_content',f'{label}: content differs from full corpus at {doc_id}')
                sources[row.get('source_type')]+=1
            self.require(n==t['documents'],'export_count',f'{label}: expected {t["documents"]}, found {n}')
            self.require(dict(sources)==t.get('source_counts'),'export_sources',f'{label}: source counts differ')
            self.hash_check(path,t.get('corpus_sha256'),'export_hash')
            proportions={s:c/n for s,c in sources.items()} if n else {}
            full_props={s:c/self.stats['documents'] for s,c in self.stats['source_counts'].items()}
            max_drift=max((abs(proportions.get(s,0)-p) for s,p in full_props.items()),default=0)
            if max_drift>self.a.source_drift:self.issue('WARNING','source_distribution',f'{label}: maximum source proportion difference vs Full is {max_drift:.4%}')
            results.append({'tier':label,'rows_checked':n,'max_source_proportion_drift':max_drift})
            print(f'Checked export {label}: {n:,} rows',file=sys.stderr,flush=True)
        self.stats['exports']=results
    def run(self):
        self.manifest=self.json_file('manifest.json')
        steps=[('questions',self.check_queries),('tiers',self.check_tiers),('diagnostics',self.load_diagnostics)]
        for name,fn in steps:
            before=self.counts['ERROR']
            try:fn();self.checks[name]='PASS' if self.counts['ERROR']==before else 'FAIL'
            except Exception as e:self.issue('ERROR',name,f'{type(e).__name__}: {e}');self.checks[name]='FAIL'
        if all(self.checks.get(k)=='PASS' for k in ('questions','tiers','diagnostics')):
            with tempfile.TemporaryDirectory(prefix='enterprise_validation_',dir=self.a.temp_dir) as td:
                db=sqlite3.connect(Path(td)/'checks.sqlite')
                for name,fn in [('full_corpus',lambda:self.check_corpus(db)),('catalog',lambda:self.check_catalog(db)),('independent_exports',lambda:self.check_exports(db))]:
                    before=self.counts['ERROR']
                    try:fn();self.checks[name]='PASS' if self.counts['ERROR']==before else 'FAIL'
                    except Exception as e:self.issue('ERROR',name,f'{type(e).__name__}: {e}');self.checks[name]='FAIL'
                db.close()
        else:self.issue('WARNING','skipped','Large corpus checks skipped until foundational errors are corrected')
        if not self.a.source_root:self.issue('WARNING','source_fidelity','No original source directory supplied; originals/body fidelity and original file coverage were not verified')
        elif self.a.source_check!='all':self.issue('WARNING','source_fidelity','Original body checks limited to gold and remapped/canonical records; full source coverage not verified')
        status='FAIL' if self.counts['ERROR'] else 'NEEDS_REVIEW' if self.counts['REVIEW'] else 'PASS_WITH_WARNINGS' if self.counts['WARNING'] else 'PASS'
        report={'validator_version':VERSION,'status':status,'structural_pass':self.counts['ERROR']==0,
                'data_directory':str(self.base),'checks':self.checks,'issue_counts':dict(self.counts),
                'statistics':self.stats,'issues':self.messages,'issues_truncated':sum(self.counts.values())>len(self.messages),
                'scope':{'generated_corpora':'all six exports and full corpus','source_check':self.a.source_check if self.a.source_root else 'not supplied',
                         'original_questions_supplied':bool(self.a.original_questions),'uuid_index_supplied':bool(self.a.uuid_index)},
                'limitation':'Checks structural integrity and exact source extraction, not semantic relevance, completeness of gold judgments, or correctness of reference answers.'}
        path=Path(self.a.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'status':status,'checks':self.checks,'issue_counts':dict(self.counts),'statistics':self.stats,'report':str(path.resolve())},ensure_ascii=False,indent=2))
        return 2 if self.counts['ERROR'] else 3 if self.counts['REVIEW'] else 0


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',required=True);p.add_argument('--report',default='validation_report.json')
    p.add_argument('--original-questions');p.add_argument('--source-root');p.add_argument('--uuid-index')
    p.add_argument('--source-check',choices=['gold','all'],default='gold')
    p.add_argument('--expected-questions',type=int,default=470);p.add_argument('--expected-gold',type=int,default=722)
    p.add_argument('--sizes',type=int,nargs='+',default=[20000,50000,100000,200000,400000])
    p.add_argument('--source-drift',type=float,default=.01,help='Warning threshold as fraction (0.01 = one percentage point)')
    p.add_argument('--temp-dir',help='Directory for disposable SQLite check index; fast local disk preferred')
    p.add_argument('--max-issues',type=int,default=100)
    a=p.parse_args()
    if a.max_issues<1:p.error('--max-issues must be positive')
    report_path=Path(a.report).resolve();base=Path(a.data).resolve()
    protected_names={'manifest.json','gold.jsonl','queries.jsonl','queries_retrieval.jsonl','qrels.tsv','order.txt','corpus.jsonl','catalog.sqlite','question_selection.json','document_id_remapping.jsonl','duplicate_documents.jsonl','conflicting_documents.jsonl','empty_content_documents.jsonl'}
    protected=(report_path in {base/name for name in protected_names} or any(base/folder in report_path.parents for folder in ('tiers','exports'))
               or any(value and report_path==Path(value).resolve() for value in (a.original_questions,a.uuid_index))
               or (a.source_root and (Path(a.source_root).resolve()==report_path or Path(a.source_root).resolve() in report_path.parents)))
    if protected:
        print('ERROR: --report cannot overwrite dataset inputs or be written inside the original source tree',file=sys.stderr);return 2
    try:return Validator(a).run()
    except Exception as e:
        # Early errors (e.g. absent manifest) still produce a report.
        report={'validator_version':VERSION,'status':'FAIL','structural_pass':False,'issues':[{'level':'ERROR','check':'startup','message':f'{type(e).__name__}: {e}'}]}
        Path(a.report).parent.mkdir(parents=True,exist_ok=True)
        Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
