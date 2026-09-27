# Business Entity Resolution

ML pipeline for the Business Entity Resolution Challenge.

## Architecture

```text
main.py
   │
   ├── preprocessing.py
   │      └── normalization
   │
   ├── blocking.py
   │      └── scalable candidate generation
   │
   ├── features.py
   │      └── name/address/country similarity
   │
   ├── training.py
   │      └── training-pair construction + XGBoost
   │
   ├── evaluation.py
   │      └── validation + macro F0.5 threshold selection
   │
   ├── inference.py
   │      └── test candidate scoring
   │
   └── io_utils.py
          └── TSV input/output
```

## Run

From the repository root:

```bash
pip install -r code/business_entity_resolution/requirements.txt

python code/business_entity_resolution/main.py     --train-dir dataset/train     --test-dir dataset/test     --output-dir output
```

Windows PowerShell:

```powershell
python code/business_entity_resolution/main.py `
    --train-dir dataset/train `
    --test-dir dataset/test `
    --output-dir output
```

To use a fixed threshold:

```powershell
python code/business_entity_resolution/main.py --threshold 0.70
```

## Outputs

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

The candidate file contains the exact candidates scored by the final matching model.
