# Business Entity Resolution — ML Challenge 2026

This pipeline uses only the supplied training/test TSV files.

## Pipeline

1. Normalize business names, addresses and country labels.
2. Generate bounded candidates using multiple inverted blocking keys:
   exact normalized name/address, country+tokens, and selective character 3-grams.
3. Build pairwise features from name/address/country similarity.
4. Train an XGBoost binary classifier on positive and blocked negative pairs.
5. Tune the decision threshold on a held-out Source-1 validation split using macro F_0.5.
6. Run the same blocking + model pipeline on the test set.
7. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Run

From the challenge `student_resource/` directory:

```bash
pip install -r requirements_entity_resolution.txt
python code/main.py \
  --train-dir dataset/train \
  --test-dir dataset/test \
  --output-dir output
```

For a fixed threshold:

```bash
python code/main.py --threshold 0.70
```

Then validate:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The challenge requires the final candidate file to contain exactly the candidates
fed into the matching model, and final matches must be a subset of those candidates.
