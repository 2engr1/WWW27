# NS-Retrieval Modular Experiment Project

This project targets the six corpus scales of EnterpriseRAG. It provides separate implementations for data preprocessing, E5 + Qdrant dense indexing, NS bidirectional token-level constraints, Cross-Encoder reranking, answer generation, and retrieval evaluation.

## 1. Project Structure

```text
dense_retrieval_project_modular/
├── prepare.py                  # Raw data conversion, chunking, stable IDs, and parent-document mapping
├── index.py                    # E5 passage embedding and Qdrant dense-index construction
├── main.py                     # Main retrieval program; orchestrates seven available modes
├── evaluate.py                 # Computes retrieval metrics per query and generates Summary JSON
├── summarize_ns_metrics.py     # Aggregates results for five NS modes across six corpus scales
├── config.py                   # Fixed experiment budgets, mode definitions, and evaluation K values
├── configs/
│   └── default.yaml            # Reference configuration
├── run_ns_experiments.sh       # Batch runner for six corpus scales × five NS modes
├── requirements.txt
├── src/
│   ├── dense_retrieval.py      # E5 query embedding and Qdrant retrieval
│   ├── ns_constraints.py       # Token-level sufficiency, necessity, and bidirectional constraints
│   ├── ranker.py               # Standard Cross-Encoder and NS-Rerank
│   ├── generate.py             # Answer generation using the Top-1 context
│   ├── dataset.py              # Data loading, Gold mapping, and parent-ID mapping
│   ├── models.py               # Retrieval-result data structures and shared scoring functions
│   ├── qdrant_backend.py       # Qdrant Local / Server clients
│   └── utils.py                # GPU checks, file I/O, hashing, and timing utilities
```

### Module Responsibilities

- `dense_retrieval.py`: Responsible only for global dense candidate retrieval.
- `ns_constraints.py`: Responsible only for first-stage E5 token-level NS constraints; it does not contain Cross-Encoder logic.
- `ranker.py`: Implements standard Cross-Encoder reranking and the second-stage NS-Rerank.
- `generate.py`: Responsible only for LLM-based answer generation.
- `main.py`: Loads components, orchestrates the pipeline according to the selected mode, and saves raw results. It does not reimplement the algorithms.
- `evaluate.py`: Computes metrics from retrieval results; it does not perform retrieval.
- `summarize_ns_metrics.py`: Reads the Summary JSON files produced by individual evaluations; it does not recompute metrics.

---

## 3. Environment and Dependencies

Install the dependencies in the Python environment used for this project:

```bash
cd /path/to/dense_retrieval_project_modular

pip install -r requirements.txt
```

To use the Tsinghua PyPI mirror:

```bash
pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Main dependencies:

- PyTorch
- Sentence-Transformers
- Transformers
- Qdrant Client
- OpenAI Python SDK (only required for generation or Judge)
- tiktoken (for chunk preprocessing)
- openpyxl (for Excel result aggregation)

An OpenAI API key is not required for retrieval or offline evaluation. Configure one only when enabling `--generate` or `--judge`:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
# For an OpenAI-compatible service, configure this endpoint as required:
# export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

---

## 4. Input Data Requirements

The `--data` argument of `prepare.py` must point to the root directory of the original EnterpriseRAG data. The directory is expected to contain:

```text
EnterpriseRAG/
├── manifest.json
├── corpus.jsonl
├── gold.jsonl
├── order.txt
└── tiers/
    ├── 20000/
    │   └── doc_ids.txt
    ├── 50000/
    │   └── doc_ids.txt
    ├── 100000/
    │   └── doc_ids.txt
    ├── 200000/
    │   └── doc_ids.txt
    ├── 400000/
    │   └── doc_ids.txt
    └── full/
        └── doc_ids.txt
```

The files have the following roles:

- `manifest.json` provides the corpus scales and document counts.
- Each line in `corpus.jsonl` must contain at least `doc_id` and `text`. `original_path`, `source_type`, `title`, and `metadata` may be provided as additional fields.
- Each line in `gold.jsonl` contains question information and must include at least `question_id`, `question`, a non-empty `expected_doc_ids` list, and a non-empty `gold_answer`. Optional fields include `question_type` and `answer_facts`.
- `order.txt` stores the ordering of all documents.
- `tiers/<scale>/doc_ids.txt` stores the document IDs for the corresponding scale. The tiers must be strictly nested according to `order.txt`.

Preprocessing expects 470 questions by default. This can be changed with `--expected-questions`. Question IDs must be unique, and every Gold document must be present at every corpus scale.

---

## 5. Data Preprocessing
*(Run this again only when the source data or chunking parameters change.)*

### 5.1 Chunk Mode (Recommended Experimental Setting)

```bash
python prepare.py prepare \
  --data /path/to/EnterpriseRAG \
  --out /path/to/hybrid_data_chunk \
  --unit chunk \
  --chunk-tokens 400 \
  --overlap 80
```

### 5.2 Document Mode

```bash
python prepare.py prepare \
  --data /path/to/EnterpriseRAG \
  --out /path/to/hybrid_data_document \
  --unit document
```

### 5.3 Output Structure

The preprocessing root directory contains `prepared.json` and six scale-specific directories:

```text
hybrid_data_chunk/
├── prepared.json
├── 20000/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   ├── gold.jsonl
│   ├── qrels.tsv
│   ├── doc_ids.txt
│   ├── id_map.jsonl
│   └── dataset.json
├── 50000/
├── 100000/
├── 200000/
├── 400000/
└── full/
```

| File | Purpose |
|---|---|
| `corpus.jsonl` | Text of each retrieval unit; in Chunk mode, each line represents one chunk |
| `queries.jsonl` | Retrieval queries, Gold parent-document UUIDs, reference answers, and answer facts |
| `gold.jsonl` | Copy of the original Gold annotations |
| `qrels.tsv` | Retrieval relevance judgments mapping queries to Gold parent-document UUIDs |
| `doc_ids.txt` | Original document IDs included at the current scale |
| `id_map.jsonl` | Mapping from retrieval-unit ID → parent-document UUID → original document ID |
| `dataset.json` | Corpus scale, number of retrieval units, hashes, chunking parameters, and collection name |
| `indexes/` | Created when the index is built |

Notes:

1. `20K`, `50K`, etc. refer to the number of original parent documents, not the number of chunks. In Chunk mode, the number of lines in `corpus.jsonl` will be greater than the number of parent documents.
2. Parent-document IDs use UUIDv5 with a fixed namespace. Chunk IDs are generated deterministically from the original document ID, chunk index, and chunk-content hash.
3. `relevant_doc_ids` in `queries.jsonl` contains parent-document UUIDs; `expected_doc_ids` retains the original document IDs.
4. `gold_answer` and `answer_facts` are used for evaluation / Judge and are not used as Dense Retrieval input. Answer generation receives only the query and the final Top-1 retrieval context.
5. If the preprocessing output directory already exists, the program validates its signature and reuses it. If the input data or chunking configuration changes, use a new output directory.

---

## 6. Build the E5 + Qdrant Dense Index

Build an index separately for each corpus scale. The following example uses the 20K scale:

```bash
python index.py index \
  --dataset /path/to/hybrid_data_chunk/20000 \
  --backend dense \
  --embed-model /path/to/e5-base-v2 \
  --device cuda:0 \
  --encode-batch-size 64 \
  --upload-batch-size 256
```

Parameter descriptions:

| Parameter | Description |
|---|---|
| `--dataset` | Directory of the current corpus scale |
| `--backend dense` | This project builds only the Dense / Qdrant index |
| `--embed-model` | E5 model name or local model directory |
| `--device` | `cuda:N` or explicitly `cpu` |
| `--encode-batch-size` | Number of texts encoded by E5 in each batch |
| `--upload-batch-size` | Number of retrieval units uploaded to Qdrant in each batch |
| `--qdrant-path` | Optional path for local Qdrant persistence |
| `--qdrant-url` | Optional Qdrant Server address |

Embedding conventions:

- Query prefix: `query: `
- Passage prefix: `passage: `
- Embeddings: L2-normalized vectors
- Qdrant distance: Cosine

The index files are stored in the directory for the corresponding scale:

```text
20000/
└── indexes/
    ├── dense.json
    └── qdrant_local/
        └── ...
```

`dense.json` stores the model, corpus hash, collection name, vector dimension, and runtime configuration. During retrieval, the program checks the corpus hash, model configuration, and number of Qdrant points associated with the index. NS does not require modifying or rebuilding this index.

---

## 7. Run the Five NS Modes

The project root contains `run_ns_experiments.sh`. The script is configured with the following server paths:

```bash
DATA_ROOT="hybrid_data_chunk"
MODEL_ROOT="ns-retrieval"
DEVICE="${1:-cuda:6}"
```

If your server uses different directories, update `DATA_ROOT` and `MODEL_ROOT` at the top of the script before running it.

Run in the foreground:

```bash
chmod +x run_ns_experiments.sh
bash run_ns_experiments.sh cuda:6
```

Run in the background:

```bash
mkdir -p logs

nohup bash run_ns_experiments.sh cuda:6 \
  > logs/ns_experiments.log 2>&1 &
```

Monitor the log:

```bash
tail -f logs/ns_experiments.log
```

Check the process:

```bash
ps -ef | grep run_ns_experiments.sh
```

The script runs the following five modes by default:

```bash
MODES=(
  ns
  ns-ce
  ns-retrieval
  ns-suf
  ns-nec
)
```

The six corpus scales are `20000`, `50000`, `100000`, `200000`, `400000`, and `full`, resulting in 30 retrieval runs in total. For each scale, the script first builds or reuses the Dense index and then runs the five modes. `evaluate.py` is executed immediately after each retrieval run, and the aggregation script is called after all runs finish.

---

## 10. Aggregate Results for Five NS Modes × Six Corpus Scales

After all corpus scales have been processed and evaluated, the batch script automatically runs:

```bash
python summarize_ns_metrics.py \
  --root outputs/<RUN_ID> \
  --out outputs/<RUN_ID>/summary_tables
```

The aggregation program can also be run manually. It reads:

```text
outputs/<RUN_ID>/<scale>/<mode>.jsonl.summary.json
```

It does not rerun retrieval or recompute per-query metrics.

Output:

```text
summary_tables/
├── ns-retrieval_table.csv
├── ns-ce_table.csv
├── ns_table.csv
├── ns-nec_table.csv
├── ns-suf_table.csv
├── all_modes.csv
├── all_modes.md
├── ns_metrics_tables.xlsx
└── warnings.txt
```
