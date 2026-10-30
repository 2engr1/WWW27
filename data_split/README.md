# EnterpriseRAG-Bench 六级数据集构建与评估

此文件夹是统一的完整版本，主程序为 `enterprise_rag.py`（2.3.1）。它已合并原始 JSON 适配，不需要旧版脚本，也不需要 `prepare_enterprise_json.py`。

适用实验规模：**20K / 50K / 100K / 200K / 400K / Full**。

## 文件与环境

| 文件 | 用途 |
|---|---|
| `enterprise_rag.py` | 检查、原始数据适配、六级构建、导出、Recall、答案质量评估 |
| `test_enterprise_rag.py` | 合成数据测试，不访问你的真实语料或模型接口 |
| `README.md` | 从解压到构建、验证和评估的操作流程 |

Python **3.10+**。只使用标准库，不需要 `pip install`。数据检查、构建、导出和 Recall 不联网、不调用模型。可选 `judge` 才调用你指定的模型服务，并产生相应费用。

本工具不实现 Dense、NS-Retrieval 或 GraphRAG 模型，也不替你运行答案生成；它统一准备数据并评估你系统返回的预测。

## 1. 解压代码包

把 ZIP 上传到服务器的工作目录，解压并进入文件夹：

```bash
unzip enterprise_rag_complete.zip
cd enterprise_rag_complete
python enterprise_rag.py --version
python enterprise_rag.py --help
```

预期版本为 `2.3.1`。文件夹无需放进数据仓库，可以独立使用。

## 2. 确认你的输入数据

本 README 使用你提供的实际路径：

| 输入 | 路径 |
|---|---|
| 仓库 | `/srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main` |
| 原始文档根目录 | `/srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/sources` |
| 问题与标注 | `/srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/questions.jsonl` |

你已经运行过检查：500 个问题、470 个有 gold 文档的问题、722 个不同的 gold 文档。仓库里 511,965 个 JSON 文件还包括配置与索引，不代表唯一文档数。**本工具只将 sources 目录下实际文档加入语料**，不把 `uuid_index.json`、组织脚手架和配置当正文。

如需重新检查：

```bash
python enterprise_rag.py inspect \
  --repo /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main
```

新版本 inspect 会按来源抽样，并显示声明字段的实际值。不要对整个仓库运行 build；`--docs` 必须指向 `generated_data/sources`。

## 3. 抽查 JSON 结构

```bash
python enterprise_rag.py audit \
  --docs /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/sources \
  --questions /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/questions.jsonl \
  --per-source 3
```

默认每个来源抽查 3 个文档，检查：

| 字段 | 提取规则 |
|---|---|
| `dataset_doc_uuid` | 原样作为稳定 `doc_id`，与 gold ID 精确匹配 |
| `title_field_name` | 按其值找到本文件的标题字段 |
| `content_field_names` | 按声明顺序读取本文件的内容字段，适配 content/body/messages/comments 等变化 |
| sources 下一级目录 | 作为 `source_type` |
| 其他字段 | 保留在 `metadata` 中，不自动加入正文 |

`content_field_names` 支持字段名字符串或非空字符串列表。列表中的字段名可以是空字符串 `""`：JSON允许这种键名，若文档实际存在该键，程序也会提取其值。不会直接删除空键，因为它可能包含正文；声明的键不存在或列表混入非字符串时仍会报错。嵌套对象和数组序列化为 JSON 文本，保留其中的内容。声明缺字段或格式错误时，程序指出路径并停止。声明正文为空但标题非空时，默认保留标题和稳定文档ID，并写入 empty_content_documents.jsonl，既不猜其他正文，也不跳过文档。若标题与声明正文都为空仍会停止，需检查原文件。

`text` 只由声明的标题和正文组成；日期、作者、噪声标记等其他字段留在 metadata。**不要将 dataset 标记或 gold 标注提供给检索/生成模型。** 如研究 metadata-aware 检索，另行统一声明各方法能使用的字段。

抽查通过不代表全库已经通过。build 会检查全库，因此通常不用再完整 audit。若需要独立全量校验，可在上面的命令增加 `--full`，会额外扫描一次全库。

## 4. 构造六级数据集

```bash
python enterprise_rag.py build \
  --format json \
  --docs /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/sources \
  --questions /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/questions.jsonl \
  --uuid-index /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/uuid_index.json \
  --out /srv/nfs/home/njnu_zb/ns-retrieval/cache_date \
  --seed 42
```


## 验证
```bash
python validate_enterprise_rag.py \
  --data /srv/nfs/home/njnu_zb/ns-retrieval/cache_date \
  --original-questions /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/questions.jsonl \
  --source-root /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/sources \
  --uuid-index /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/uuid_index.json \
  --source-check all \
  --report validation_report.json
```

**输出目录必须不存在或为空，且不能位于 sources 目录内。** 若已有同名结果，请使用新的输出路径，例如 `EnterpriseRAG-NS-v2-seed42`。不覆盖旧结果。

默认规模：

| Tier 目录名 | 文档数 |
|---|---:|
| `20000` | 20,000 |
| `50000` | 50,000 |
| `100000` | 100,000 |
| `200000` | 200,000 |
| `400000` | 400,000 |
| `full` | 实际唯一文档数 N |

Full 不截成 510,000。你的最大预设层为 400,000，因此要求实际 Full > 400,000。

构造协议：
1. 先筛选 expected_doc_ids 非空且 gold_answer 为非空字符串的问题；只将这组检索+答案问题及其gold用于构建和评估。你的版本预期470个。再取其gold ID并集G。
2. 从最小的 20K 起保留 G；你的版本预期为 722 个。
3. 剩余文档按 source 分组，用固定 seed 在每组内打乱。
4. 按各来源的均匀进度槽合并成唯一的背景顺序。
5. 各层取“固定核心 + 背景前缀”，保证严格嵌套。

因此：D20K ⊂ D50K ⊂ D100K ⊂ D200K ⊂ D400K ⊂ DFull。背景前缀近似保持剩余背景的来源比例；固定 gold 核心可能造成总比例的小幅偏离，各层统计会给出真实比例。

每扫描 10,000 个文档打印进度，扫描完成后统一解决ID重复与冲突。扫描 NFS 上 50 多万个小文件可能较慢。全文先存入SQLite input_records表，再按ID分组解决冲突，输出docs表及JSONL，不把全量正文读入内存；ID排序/映射和UUID索引占O(N)内存。中间阶段会临时保存原始记录和解析后记录；完成解析后删除中间表并VACUUM，需为临时SQLite增长预留磁盘空间。默认另外写出六份独立 corpus，包含相交层的重复文本；磁盘需要为 catalog、全量 corpus 和六份导出预留空间。只需要 ID 清单时可加 --manifests-only。程序失败可能留下中间目录，修复输入后请换一个新输出目录。

可选参数：
- `--sizes 20000 50000 100000 200000 400000`：显式指定非 Full 层。
- `--anchors fixed_ids.txt`：每行一个经过审核的固定锚点 ID；默认为空。
- `--format auto`：有 JSON 就选择 JSON，否则 TXT；你的输入建议明确 `json`。
- `--format txt`：兼容 TXT 发布语料，需保留来源目录。

原始 EnterpriseRAG JSON 不需要 `--json-id-field` 或 `--json-text-fields`。这两个选项保留给其他自定义顶层 JSON 格式，不能在本数据上硬写全库都使用 content。

非 gold 文档只能视作未标注候选，不自动作为已确认 hard negatives。不同ID的近重复文档保留；同ID标题/正文相同的记录合并，其他路径、来源和metadata保存在duplicate_variants中。正文不同则按显式提供的UUID索引确定原ID归属，为其他正文版本分配稳定新ID。无法确定原ID归属的冲突、缺gold等情况会中止，不丢弃问题。可加 --empty-content-policy error 恢复严格的空正文报错；默认 keep-title。

## 5. 输出结构与完整性检查

| 输出路径 | 用途 |
|---|---|
| `corpus.jsonl` | 全量文档文件，按全局 tier 顺序排列 |
| `exports/corpus_20000.jsonl` 等 | 六份可直接独立用于建索引的语料文件，build 默认生成 |
| `catalog.sqlite` | 构建 catalog，包含全文 |
| `order.txt` | 全局文档 ID 顺序 |
| `queries.jsonl` | 筛选后的470个问题输入，不含答案和gold IDs |
| `queries_retrieval.jsonl` | 同一组470个问题，保留文件名以兼容原有流程 |
| `gold.jsonl` | 同一组470个问题的完整参考标注，仅用于评估 |
| `question_selection.json` | 输入/筛选数量，以及被排除问题的ID与原因；不属于评估问题集 |
| `qrels.tsv` | 四列 qid / 0 / docid / relevance=1 |
| `tiers/20000/doc_ids.txt` 等 | 六级语料的准确 ID 列表 |
| `tiers/*/stats.json` | 各层文档数、来源计数、gold 完整性与 ID 哈希 |
| `manifest.json` | 全量统计、seed、schema 统计与文件哈希；包含 title_only_documents 与 title_only_gold_documents |
| `duplicate_documents.jsonl` | 同ID、同正文的合并记录与保留的metadata差异 |
| `conflicting_documents.jsonl` | 同ID、不同正文的全部冲突及解决状态 |
| `document_id_remapping.jsonl` | 版本文档的旧ID→新ID映射、路径及gold相关标记 |
| `empty_content_documents.jsonl` | 空正文但标题非空的文档清单，标记 is_gold 和 gold_question_ids |

成功构建后检查：

```bash
python - <<'PY'
import json
from pathlib import Path
base = Path('/srv/nfs/home/njnu_zb/ns-retrieval/cache_date')
m = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
print('Full documents:', m['documents'])
print('Questions:', m['questions'])
print('Retrieval questions:', m['retrieval_questions'])
print('Gold union:', m['gold_union_count'])
previous = set()
for t in m['tiers']:
    ids = (base / 'tiers' / t['tier'] / 'doc_ids.txt').read_text().splitlines()
    current = set(ids)
    assert len(ids) == len(current) == t['documents']
    assert t['all_gold_present']
    assert previous < current
    print(t['tier'], t['documents'], t['all_gold_present'])
    previous = current
PY
```

对于你贴出的 questions，预期 input_questions=500、questions=470、retrieval_questions=470、excluded_questions=30、gold_union_count=722。程序按实际字段筛选，不硬编码数量。inspect 检查的是原始仓库，因此其 question_count 仍为500，不能与构建后 manifest 的 questions 混淆。

**新版 build 默认直接输出六份独立 corpus 文件，同时保留六份 ID 列表。** 所有方法必须按相同 tier ID 过滤原始输入后再建索引。不能在全量图上检索再过滤结果，否则小层已经接触层外信息。

## 6. 直接使用六份独立 corpus

运行第4步的 build 后，下面六个文件已经生成，不需要再运行 materialize：

| 文件 | 规模 |
|---|---:|
| `exports/corpus_20000.jsonl` | 20K |
| `exports/corpus_50000.jsonl` | 50K |
| `exports/corpus_100000.jsonl` | 100K |
| `exports/corpus_200000.jsonl` | 200K |
| `exports/corpus_400000.jsonl` | 400K |
| `exports/corpus_full.jsonl` | Full |

每份文件只包含对应层的文档，可直接交给模型建索引。每层 stats.json 记录 corpus_file 与 SHA256。六层问题与评估标注共用根目录文件，保证评估问题一致。

程序在写全量 corpus 的同一遍循环中写出这六个文件，不再逐层扫描全量正文。但相交层会重复保存文本，磁盘占用增加。希望仅保留原来的全量正文+ID清单，可在 build 命令末尾添加 `--manifests-only`。

已用旧版构建的数据不必重新构建；旧版没有独立 corpus 时仍可导出：

```bash
mkdir -p /srv/nfs/home/njnu_zb/ns-retrieval/cache_date/exports
for tier in 20000 50000 100000 200000 400000 full; do
  python enterprise_rag.py materialize \
    --data /srv/nfs/data/njnu_zb/EnterpriseRAG-NS-seed42 \
    --tier "$tier" \
    --out "/srv/nfs/home/njnu_zb/ns-retrieval/cache_date/exports/corpus_${tier}.jsonl"
done
```

只对缺失的文件运行 materialize，输出已存在时会拒绝覆盖。新版默认 build 已有六个文件，请不要重复导出。

corpus 每行包含：doc_id、title、text、source_type、original_path、metadata、extracted_fields。实际检索输入通常取 text。分块时保存明确的 chunk→parent_doc_id 映射；不要通过截断 chunk ID 猜 parent ID。固定分块器参数和模型配置，同一文档的分块边界在各层保持相同。

## 7. 运行你的检索与生成模型

每个 method × tier 输出独立预测 JSONL：

```json
{"question_id":"qst_0001","answer":"系统生成的答案","document_ids":["dsid_abc123","dsid_def456"]}
```

上面的 ID 只是格式示例，实际必须使用 corpus 中的稳定 ID。document_ids 是按排名排列的检索结果，不是仅保留答案引用。若返回 chunks，先映射到父文档，按第一次出现的位置稳定去重，文档级 K 表示唯一父文档数。测 Recall@20 时，建议返回至少 20 个唯一父文档。

主规模实验对同一组 queries_retrieval 的问题评估检索和生成；参考证据在各层均存在。固定生成模型、提示、检索 K、上下文 token 预算和裁剪策略。gold.jsonl / qrels 不进入模型索引或答案生成输入。

当前任务仅关注有检索标注和参考答案的470个问题。10个high_level和20个info_not_found不写入queries或gold，不进入当前检索或生成质量评估；原始questions.jsonl不被修改。

## 8. 计算检索 Recall

```bash
python enterprise_rag.py recall \
  --data /srv/nfs/home/njnu_zb/ns-retrieval/cache_date \
  --tier 20000 \
  --predictions predictions_dense_20k.jsonl \
  --ks 1 5 10 20 \
  --out recall_dense_20k.json
```

其他层更改 tier 和预测/输出文件。Recall@K = |TopK(q) ∩ G(q)| / |G(q)|，总体按问题宏平均，同时按 question_type 分组。还输出 Hit@K、All-Gold@K、nDCG、MRR 和 precision_qrels。

多证据问题命中一个文档不等于 Recall=1。无 gold 问题不进入分母；缺失有 gold 的预测按0计入并报告。重复父文档按首次排名去重，未知/重复 qid、层外/未知 doc ID 会报错。非 gold 未必无效，precision_qrels 不能称作实际支持性精度。

## 9. 可选：生成答案质量评估

使用固定 gold answer/facts，比较 Correctness 与事实 Completeness。判定目标与官方这两类指标相近，但本实现是固定参考协议，不是官方 evaluator 的逐行复刻，不会自动修订 gold。它也不直接测答案对 retrieved context 的 faithfulness。

你需要可用的 HTTPS OpenAI-compatible Chat Completions 接口。接口须接受 temperature=0，返回 choices[0].message.content，其中是要求的 JSON 判定。以下 URL/model 是占位符，替换为你的服务：

```bash
export JUDGE_API_KEY="你的接口密钥"
python enterprise_rag.py judge \
  --data /srv/nfs/data/njnu_zb/EnterpriseRAG-NS-seed42 \
  --predictions predictions_dense_20k.jsonl \
  --scope retrieval \
  --endpoint "https://YOUR_PROVIDER/chat/completions" \
  --model "YOUR_JUDGE_MODEL" \
  --votes 1 \
  --out quality_dense_20k.jsonl
```

`--scope retrieval` 只评估有 gold 文档的问题；选定范围内每个问题必须有字符串 answer，空答案可以，缺行会报错。`--scope all` 评估输出gold.jsonl中的全部问题，也就是筛选后的470个，不会重新读取原始500个。

每个问题一次 correctness 请求，每个 answer_fact 一次独立验证。总体 completeness 是问题事实覆盖率的宏平均，空事实列表记 null。`--votes 3` 用多数票，费用约三倍，同一模型重复调用不代表三个独立评估者。保持所有方法/规模使用同一模型和提示。

输出逐条理由/投票；摘要保存到 `<out>.summary.json`，配置到 `<out>.config.json`。成功完成的问题可续跑；gold、predictions、模型、接口、scope、votes 或程序版本变化需要新的输出文件。**不能用新版续跑旧版 judge 的结果文件。** 不要多个进程同时写同一结果。

若 HTTP/JSON 错误，程序停止并保留已完成的问题，不把失败伪装成0。temperature=0也不保证远端服务完全确定。

## 空正文报错后的重新运行

旧版因文档的声明正文为空而停止时，失败目录可能已有 catalog.sqlite，不能直接在原非空目录重跑。请下载本包2.3.1替换代码，并使用一个新的输出目录，例如你的实际输出路径对应：

```bash
python enterprise_rag.py build \
  --format json \
  --docs /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/sources \
  --questions /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/questions.jsonl \
  --uuid-index /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/uuid_index.json \
  --out /srv/nfs/home/njnu_zb/ns-retrieval/cache_date_v231 \
  --seed 42 \
  --empty-content-policy keep-title
```

检查 manifest 的 title_only_documents、title_only_gold_documents，以及 empty_content_documents.jsonl。声明字段为空，不证明JSON的所有其他字段都没有正文；若其他字段实际有证据，需要检查文档声明与内容后再适配，程序不会自动猜。若清单中包含gold，必须在解释Recall/答案质量前审查原文；保留ID不能代替缺失的证据内容。修复后所有方法/层应使用同一份公共语料，而不是对某个模型单独补文本。空正文策略不额外删除筛选后的问题、不改gold、不伪造答案。

## 不同正文共用同一个UUID的处理

你已确认的例子：
- UUID索引指向 `google_drive/users/priya-docs-quickwins-sprint.json`。
- Confluence版本正文不同，带 `dataset_noise_document=True`，并用supersedes_source_path指向Google Drive文档。
- 该原始ID未被gold问题引用。

新版保留索引指定文档的原ID，将另一份不同正文保留为独立文档，使用 `dsid_` + SHA256(原ID、标题正文哈希)前32位作为稳定新ID；不会修改原始JSON，也不会将来源标记拼入检索正文。相同原ID且相同标题/正文的多个副本只保留一份正文，同时保存其他metadata变体。遇到hash碰撞会报错，不会覆盖记录。

请明确提供 `--uuid-index /srv/nfs/data/njnu_zb/EnterpriseRAG-Bench-main/generated_data/uuid_index.json`。索引需要是ID→相对sources路径的对象，与当前数据属于同一版本。没有索引、索引未覆盖冲突ID、索引路径不在实际候选文件中时，程序扫描并报告全部未解决冲突，不输出六级corpus。

`conflicting_documents.jsonl` 记录 resolved_by_uuid_index/unresolved 状态；`document_id_remapping.jsonl` 保留新旧ID与路径，并标记原始ID是否为gold。gold标注ID保持不变，仍对应索引文档；如其他冲突涉及gold，需审查索引映射与参考答案是否吻合。不要将多个不同正文错误合并后用同一个gold ID评分。

manifest额外报告 input_document_files、duplicates_identical_evidence、resolved_id_conflicts、remapped_documents、uuid_index_sha256。Full使用修复后实际文档数量，可能大于原始唯一UUID数量，不能按旧UUID数强制截断。

之前失败目录不可直接续跑；本版没有跨版本构建续跑功能。换新目录 cache_date_v231 重新构建，或保留旧失败目录作诊断，勿混用其catalog。评估模型输出的document_ids必须使用新corpus的doc_id，不能全部映射回original_doc_id，否则噪声文档会错误算作gold。

## 10. 测试与限制

```bash
python -m unittest discover -s . -v
```

22 项合成测试已通过，覆盖声明字段适配、嵌套内容、空正文/缺字段、来源抽查、gold保留、严格嵌套、seed复现、ID冲突、多gold与缺失预测Recall、层外ID、导出、TXT兼容、mock judge及续跑，并验证独立 corpus 与 ID 列表/哈希一致、manifests-only行为，以及标题保留、空数组识别、空正文gold诊断，以及输入500/输出470筛选语义和无参考答案过滤，以及UUID索引canonical归属、噪声正文保留、稳定重映射、metadata重复记录、未解决冲突不导出，以及空字符串键的内容完整保留与缺失键检查。没有调用付费模型接口。

本包根据你提供的 inspect 输出和此前格式说明适配，没有访问你的服务器，也没有在真实全量 corpus 上执行。audit/build 会在你的服务器继续验证实际声明字段、文档数和 gold 匹配。若报错，保留报错文件路径和消息以便定位；不要通过跳过文件或删除问题规避 gold 缺失。
