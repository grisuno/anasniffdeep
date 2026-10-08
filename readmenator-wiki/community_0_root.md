# root

*Community 0 | 1 files | cohesion 1.00*

## Definition

This community groups 1 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `capture_packet`, `convert_ip_to_octets`, `create_database`, `evaluate_model`, `extract_features`, `preprocess_packet`, `real_time_packet_capture`, `store_positive_packet`. Core file: `main.py` (9 symbols).

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `main.py` | py | utility | 9 | no |

## Key Symbols

- `convert_ip_to_octets` (function, `main.py:15`) `def convert_ip_to_octets(ip)`
- `preprocess_packet` (function, `main.py:18`) `def preprocess_packet(packet)`
- `capture_packet` (function, `main.py:23`) `def capture_packet()`
- `extract_features` (function, `main.py:34`) `def extract_features(packet)`
- `create_database` (function, `main.py:54`) `def create_database()`
- `store_positive_packet` (function, `main.py:62`) `def store_positive_packet(packet)`
- `train_model` (function, `main.py:73`) `def train_model(X_train, y_train, X_val, y_val)`
- `evaluate_model` (function, `main.py:81`) `def evaluate_model(model, X_val, y_val)`
- `real_time_packet_capture` (function, `main.py:85`) `def real_time_packet_capture(model)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `main.py`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `main.py`
