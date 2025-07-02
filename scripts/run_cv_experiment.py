#!/usr/bin/env python3
"""
Master script to run a k-fold cross-validation experiment.

This script orchestrates the entire pipeline:
1. Defines folds either by using a pattern for ir_datasets (e.g., for TREC Robust)
   or by loading a custom JSON file.
2. For each fold, it calls the `create_training_data.py`, `train_weights.py`,
   and `evaluate_model.py` scripts.
3. Aggregates the results from all test folds into a single run file.
4. Provides a final command to evaluate the aggregated run file.

Usage examples:

# --- For TREC Robust (using ir_datasets folds) ---
python run_cv_experiment.py \
    --experiment_name robust2004_in_domain \
    --num_folds 5 \
    --dataset_pattern "disks45/nocr/trec-robust-2004/fold{}" \
    --index_path ./indexes/disks45_nocr_trec-robust-2004_bert-base-uncased/ \
    --lucene_path /path/to/your/lucene/jars/

# --- For a custom dataset (using a JSON folds file) ---
python run_cv_experiment.py \
    --experiment_name custom_dataset_cv \
    --dataset_name "my-custom-collection" \
    --folds_json ./path/to/my_folds.json \
    --index_path ./indexes/my-custom-collection_bert-base-uncased/ \
    --lucene_path /path/to/your/lucene/jars/
"""

import subprocess
import os
import json
import argparse
from collections import defaultdict
from pathlib import Path


def get_qids_from_dataset(dataset_name):
    """Helper to get all query IDs from an ir_datasets split."""
    import ir_datasets
    dataset = ir_datasets.load(dataset_name)
    return {q.query_id for q in dataset.queries_iter()}


def main():
    parser = argparse.ArgumentParser(description="Run a k-fold cross-validation experiment.")
    # --- Experiment Setup ---
    parser.add_argument('--experiment_name', type=str, required=True, help="A name for this experiment run.")
    parser.add_argument('--index-path', type=str, required=True,
                        help="Path to the pre-built BM25 index for the full collection.")
    parser.add_argument('--lucene-path', type=str, required=True, help="Path to Lucene JAR files.")

    # --- Fold Definition (Choose ONE) ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset-pattern', type=str,
                       help="Pattern for ir_datasets with folds, e.g., 'disks45/nocr/trec-robust-2004/fold{}'")
    group.add_argument('--folds-json', type=str, help="Path to a custom JSON file defining the folds.")

    # --- Optional arguments for custom folds ---
    parser.add_argument('--dataset-name', type=str,
                        help="The base ir_datasets name for the custom collection (used with --folds_json).")
    parser.add_argument('--num-folds', type=int, default=5, help="Number of folds (used with --dataset_pattern).")

    args = parser.parse_args()

    # --- Setup experiment directories ---
    base_dir = Path(f"./cv_experiments/{args.experiment_name}")
    feature_base_dir = ensure_dir(base_dir / "features")
    model_base_dir = ensure_dir(base_dir / "models")
    eval_base_dir = ensure_dir(base_dir / "evaluations")

    final_aggregated_run = defaultdict(list)
    folds_config = []

    # --- Logic to define folds ---
    if args.dataset_pattern:
        print(f"Using ir_datasets pattern: {args.dataset_pattern}")
        for i in range(1, args.num_folds + 1):
            folds_config.append({
                'name': f"fold{i}",
                'train_dataset': args.dataset_pattern.format(i) + "/train",
                'test_dataset': args.dataset_pattern.format(i) + "/test"
            })
    else:  # Using --folds_json
        print(f"Using custom folds from: {args.folds_json}")
        with open(args.folds_json, 'r') as f:
            custom_folds = json.load(f)
        for fold_key, fold_data in custom_folds.items():
            # Write query IDs to temporary files for the child scripts to use
            train_qids_file = f"{feature_base_dir}/fold_{fold_key}_train_qids.txt"
            test_qids_file = f"{feature_base_dir}/fold_{fold_key}_test_qids.txt"
            with open(train_qids_file, 'w') as f_train:
                f_train.write('\n'.join(fold_data['training']))
            with open(test_qids_file, 'w') as f_test:
                f_test.write('\n'.join(fold_data['testing']))

            folds_config.append({
                'name': f"fold{fold_key}",
                'train_dataset': args.dataset_name,
                'test_dataset': args.dataset_name,
                'train_qids_file': train_qids_file,
                'test_qids_file': test_qids_file
            })

    # --- Main Cross-Validation Loop ---
    for fold in folds_config:
        fold_name = fold['name']
        print(f"\n{'=' * 20} Processing {fold_name} {'=' * 20}")

        feature_dir = ensure_dir(feature_base_dir / fold_name)
        model_dir = ensure_dir(model_base_dir / fold_name)
        eval_dir = ensure_dir(eval_base_dir / fold_name)

        # 1. Generate features for the training split
        print(f"  [1/3] Generating features for {fold['train_dataset']}...")
        create_cmd = [
            "python", "scripts/create_training_data.py",
            "--dataset", fold['train_dataset'],
            "--output-dir", str(feature_dir),
            "--index-path", args.index_path,
            "--lucene-path", args.lucene_path,
        ]
        if 'train_qids_file' in fold:
            create_cmd.extend(["--query_ids_file", fold['train_qids_file']])
        subprocess.run(create_cmd, check=True)

        # 2. Learn weights on the training split
        print(f"  [2/3] Learning weights...")
        feature_file = f"{feature_dir}/{fold['train_dataset'].replace('/', '_')}_features.json.gz"
        train_cmd = [
            "python", "scripts/train_weights.py",
            "--feature-file", feature_file,
            "--validation-dataset", fold['train_dataset'],
            "--output-dir", str(model_dir),
        ]
        if 'train_qids_file' in fold:
            train_cmd.extend(["--query-ids-file", fold['train_qids_file']])  # Note: train_weights needs this too
        subprocess.run(train_cmd, check=True)

        # 3. Evaluate on the held-out test split
        print(f"  [3/3] Evaluating on the test set...")
        weights_file = f"{model_dir}/learned_weights.json"
        eval_cmd = [
            "python", "scripts/evaluate_model.py",
            "--weights-file", weights_file,
            "--dataset", fold['test_dataset'],
            "--output-dir", str(eval_dir),
            "--index-path", args.index_path,
            "--lucene-path", args.lucene_path,
            "--save-runs",
        ]
        if 'test_qids_file' in fold:
            eval_cmd.extend(["--query-ids-file", fold['test_qids_file']])  # Note: evaluate_model needs this
        subprocess.run(eval_cmd, check=True)

        # Aggregate the run file from this fold
        run_file_path = f"{eval_dir}/runs/our_method.txt"
        with open(run_file_path, 'r') as f_run:
            for line in f_run:
                qid = line.split()[0]
                final_aggregated_run[qid].append(line.strip())

    # --- Final Aggregation ---
    print(f"\n{'=' * 20} Aggregating and Finalizing {'=' * 20}")
    final_run_path = base_dir / "final_aggregated_run.txt"
    with open(final_run_path, 'w') as f_out:
        for qid in sorted(final_aggregated_run.keys(), key=int):
            lines = sorted(final_aggregated_run[qid], key=lambda x: float(x.split()[4]), reverse=True)
            for i, line in enumerate(lines):
                parts = line.split()
                f_out.write(f"{parts[0]} Q0 {parts[2]} {i + 1} {parts[4]} {args.experiment_name}\n")

    print(f"Final aggregated run file created at: {final_run_path}")
    print("\nTo get the final score, use pytrec_eval:")
    print(f"python -m pytrec_eval -c -m all_trec <qrels_file> {final_run_path}")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()