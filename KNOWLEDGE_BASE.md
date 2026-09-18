# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. 1 files, 9 symbols, 13 imports. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM, Ruby, Swift, Kotlin, Scala, Lua, Elixir.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Start here:** Statistics Dashboard for scope, God Nodes for blast radius, Architecture Reference for per-file API. Agents: prefer `readmenator-agent/INDEX.md` + `SYMBOLS.md`.

**Wiki:** prefer `readmenator-wiki/index.md` for progressive disclosure: one synthesis page per community, `connections.json` with EXTRACTED vs INFERRED confidence, `queries.md` log, `REPORT.md` audit.

**Confidence:** EXTRACTED = parsed from source, INFERRED = heuristic bridge, AMBIGUOUS = reported, never hidden. See `readmenator-wiki/REPORT.md`.

**Total Files Parsed:** 1 | **Total Symbols Extracted:** 9 | **Total Imports:** 13

<!-- ranking_model: v1.0 | weights: {ppr:0.45,auth:0.2,test:0.15,doc:0.1,fresh:0.1} | alpha:0.85 | commit:05a4468 | date:2026-07-18 -->


## Table of Contents

1. [Statistics Dashboard](#statistics-dashboard)
2. [Architectural Layers](#architectural-layers)
3. [Ranked Context](#ranked-context)
4. [God Nodes](#god-nodes)
5. [Suggested Questions](#suggested-questions)
6. [Hotspot Analysis](#hotspot-analysis)
7. [Change Impact Analysis](#change-impact-analysis)
8. [Suggested Linting Rules](#suggested-linting-rules)
9. [Orphans](#orphans)
10. [Query Recipes](#query-recipes)
11. [Structural Knowledge Map](#structural-knowledge-map)
12. [UML Class Diagram](#uml-class-diagram)
13. [Code Property Graph](#code-property-graph)
14. [Architecture Reference](#architecture-reference)
    - [PY (1 files)](#py-1-files)

---

## Statistics Dashboard

| Metric | Value |
|--------|-------|
| Total Files | 1 |
| Total Symbols | 9 |
| Total Imports | 13 |
| Call Edges | 48 |
| Inheritance Edges | 0 |
| Languages | 1 |
| Avg Symbols/File | 9.0 |
| Avg Imports/File | 13.0 |

### Top Files by Import Count (Fan-Out)

| File | Imports | Symbols | Language |
|------|---------|---------|----------|
| `main.py` | 13 | 9 | py |

---

## Architectural Layers

Auto-detected from path patterns, naming conventions, and imported frameworks.

| Layer | Files |
|-------|-------|
| utility | 1 |

### utility

- `main.py` (py, 9 symbols)

---

## Ranked Context

Files ranked by composite score for the current query context. The ranking combines Personalized PageRank (query relevance), global authority, test coverage, documentation coverage, and code freshness. Model: v1.0.

| Rank | File | Composite | PPR | Authority | Test | Doc |
|------|------|-----------|-----|-----------|------|-----|
| 1 | `main.py` | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |

---

## God Nodes

Most architecturally central files ranked by combined import/export degree and symbol richness.

| File | Score | Connections | PageRank |
|------|-------|-------------|----------|
| `main.py` | 0.9 | | 0.0000 |

---

## Suggested Questions

Auto-generated exploration prompts based on graph structure:

- What does main.py depend on, and what depends on it? (0 connections)
- What is the overall architecture of this codebase?

---

## Hotspot Analysis

Files ranked by combined complexity (symbol count) and centrality (connection count). High-scoring files are architecturally critical and may need refactoring attention.

| File | Complexity | Centrality | Combined | Symbols | Connections |
|------|-----------|------------|----------|---------|-------------|
| `main.py` | 1.000 | 1.000 | 1.000 | 9 | 13 |

---

## Change Impact Analysis

Files sorted by how many other files would be affected if they changed. High-impact files should be changed with caution.

| File | Direct Dependents | Transitive Dependents | Total Impact |
|------|------------------|----------------------|--------------|
| `main.py` | 0 | 0 | 0 |

---

## Suggested Linting Rules

Automatically suggested linting and security rules based on patterns detected in the codebase. These can be exported as Semgrep rules using the `--export-rules` flag.

| Rule ID | Severity | Description | Language | Matches |
|---------|----------|-------------|----------|---------|
| `RM001` | info | Large number of functions in py: 9 total | py | 9 |
| `RM002` | info | Print statement found (consider logging instead) | python | 12 |

---

## Orphans

Files with no documentation or low connectivity. These are candidates for documentation investment or cleanup.

- `main.py` (9 symbols, no doc)

---

## Query Recipes

Example queries you can run against this knowledge base using the ranking engine:

```
# Find files most relevant to a concept
readmenator query "Where is the import resolver implemented?"

# Rank files by relevance to a topic
readmenator query "How does documentation generation work?"

# Explain why a file ranks highly
readmenator query "explain readmenator/_documentation.py"

# Trace dependency paths with ranked context
readmenator query "path from CLI to exporter"
```

The ranking model uses the following signals:

- **Personalized PageRank** (45% weight): query-specific relevance via seed propagation
- **Global Authority** (20% weight): structural importance via standard PageRank
- **Test Coverage** (15% weight): fraction of symbols referenced in test files
- **Doc Coverage** (10% weight): presence of docstrings and file-level docs
- **Freshness** (10% weight): recent modification activity

Results include score decomposition and justification paths for each ranked item.

---

## Structural Knowledge Map

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
    main_py["main.py (py)"]
    class main_py mod;
    main_py_convert_ip_to_octets["convert_ip_to_octets"]
    class main_py_convert_ip_to_octets fn;
    main_py --> main_py_convert_ip_to_octets
    main_py_preprocess_packet["preprocess_packet"]
    class main_py_preprocess_packet fn;
    main_py --> main_py_preprocess_packet
    main_py_capture_packet["capture_packet"]
    class main_py_capture_packet fn;
    main_py --> main_py_capture_packet
    main_py_extract_features["extract_features"]
    class main_py_extract_features fn;
    main_py --> main_py_extract_features
    main_py_create_database["create_database"]
    class main_py_create_database fn;
    main_py --> main_py_create_database
    ext_socket["socket"]
    class ext_socket ext;
    main_py -.->|imports| ext_socket
    ext_sqlite3["sqlite3"]
    class ext_sqlite3 ext;
    main_py -.->|imports| ext_sqlite3
    ext_pandas["pandas"]
    class ext_pandas ext;
    main_py -.->|imports| ext_pandas
    ext_numpy["numpy"]
    class ext_numpy ext;
    main_py -.->|imports| ext_numpy
    ext_sklearn_model_selection["sklearn.model_selection"]
    class ext_sklearn_model_selection ext;
    main_py -.->|imports| ext_sklearn_model_selection
    ext_sklearn_preprocessing["sklearn.preprocessing"]
    class ext_sklearn_preprocessing ext;
    main_py -.->|imports| ext_sklearn_preprocessing
    ext_scapy_all["scapy.all"]
    class ext_scapy_all ext;
    main_py -.->|imports| ext_scapy_all
    ext_scapy_layers_dot11["scapy.layers.dot11"]
    class ext_scapy_layers_dot11 ext;
    main_py -.->|imports| ext_scapy_layers_dot11
    ext_scapy_layers_inet["scapy.layers.inet"]
    class ext_scapy_layers_inet ext;
    main_py -.->|imports| ext_scapy_layers_inet
    ext_scapy_layers_l2["scapy.layers.l2"]
    class ext_scapy_layers_l2 ext;
    main_py -.->|imports| ext_scapy_layers_l2
    ext_tensorflow_keras_models["tensorflow.keras.models"]
    class ext_tensorflow_keras_models ext;
    main_py -.->|imports| ext_tensorflow_keras_models
    ext_tensorflow_keras_layers["tensorflow.keras.layers"]
    class ext_tensorflow_keras_layers ext;
    main_py -.->|imports| ext_tensorflow_keras_layers
    ext_tensorflow_keras_losses["tensorflow.keras.losses"]
    class ext_tensorflow_keras_losses ext;
    main_py -.->|imports| ext_tensorflow_keras_losses
```

---

## Code Property Graph

Machine-readable Code Property Graph (CPG) in JSON-LD format. This block allows AI agents to parse the full structural graph without additional file reads. Compatible with GraphRAG pipelines.

```json
{"@context": "https://schema.org", "analysis": {"communities": [], "god_nodes": [{"node_id": "main.py", "score": 0.9}], "surprising_connections": []}, "edges": [{"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "socket"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "sqlite3"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "sklearn.model_selection"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "sklearn.preprocessing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scapy.all"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scapy.layers.dot11"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scapy.layers.inet"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "scapy.layers.l2"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "tensorflow.keras.models"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "tensorflow.keras.layers"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "main.py", "target": "tensorflow.keras.losses"}], "generator": "readmenator", "metadata": {"edge_count": 61, "file_count": 1, "language_count": 1, "symbol_count": 9}, "nodes": [{"id": "main.py", "kind": "module", "label": "main.py", "language": "py", "sha256": "bd1fcf6555ae4c07", "symbol_count": 9, "symbols": [{"kind": "function", "line": 15, "name": "convert_ip_to_octets", "signature": "def convert_ip_to_octets(ip)"}, {"kind": "function", "line": 18, "name": "preprocess_packet", "signature": "def preprocess_packet(packet)"}, {"kind": "function", "line": 23, "name": "capture_packet", "signature": "def capture_packet()"}, {"kind": "function", "line": 34, "name": "extract_features", "signature": "def extract_features(packet)"}, {"kind": "function", "line": 54, "name": "create_database", "signature": "def create_database()"}, {"kind": "function", "line": 62, "name": "store_positive_packet", "signature": "def store_positive_packet(packet)"}, {"kind": "function", "line": 73, "name": "train_model", "signature": "def train_model(X_train, y_train, X_val, y_val)"}, {"kind": "function", "line": 81, "name": "evaluate_model", "signature": "def evaluate_model(model, X_val, y_val)"}, {"kind": "function", "line": 85, "name": "real_time_packet_capture", "signature": "def real_time_packet_capture(model)"}]}], "type": "CodePropertyGraph", "version": "1.0"}
```

---

## Architecture Reference

### PY (1 files)

#### `main.py`
**Path:** `main.py`

**Functions:**
- `convert_ip_to_octets` (line 15) `def convert_ip_to_octets(ip)`
- `preprocess_packet` (line 18) `def preprocess_packet(packet)`
- `capture_packet` (line 23) `def capture_packet()`
- `extract_features` (line 34) `def extract_features(packet)`
- `create_database` (line 54) `def create_database()`
- `store_positive_packet` (line 62) `def store_positive_packet(packet)`
- `train_model` (line 73) `def train_model(X_train, y_train, X_val, y_val)`
- `evaluate_model` (line 81) `def evaluate_model(model, X_val, y_val)`
- `real_time_packet_capture` (line 85) `def real_time_packet_capture(model)`
