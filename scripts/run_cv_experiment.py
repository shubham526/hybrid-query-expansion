#!/usr/bin/env python3
"""
Master script to run a k-fold cross-validation experiment.

This script orchestrates the entire pipeline. It requires a pre-existing
baseline run file to use as the source for pseudo-relevance feedback and as the
candidate set for reranking.
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
    try:
        dataset = ir_datasets.load(dataset_name)
        return {q.query_id for q in dataset.queries_iter()}
    except KeyError:
        print(f"ERROR: Dataset '{dataset_name}' not found.")
        return set()


def ensure_dir(path):
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run a k-fold cross-validation experiment.")
    # --- Use hyphens for all arguments for consistency ---
    parser.add_argument('--output-dir', type=str, required=True, help='Base output directory.')
    parser.add_argument('--experiment-name', type=str, required=True, help="A name for this experiment run.")
    parser.add_argument('--index-path', type=str, required=True, help="Path to the BM25 index.")
    parser.add_argument('--lucene-path', type=str, required=True, help="Path to Lucene JAR files.")
    parser.add_argument('--run-file-path', type=str, required=True,
                        help="Path to the baseline TREC run file for PRF and reranking.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset-pattern', type=str,
                       help="Pattern for ir_datasets folds, e.g., 'disks45/nocr/trec-robust-2004/fold{}'")
    group.add_argument('--folds-json', type=str, help="Path to a custom JSON file defining the folds.")

    parser.add_argument('--base-dataset-name', type=str, required=True,
                        help="The base ir_datasets name for the full collection.")
    parser.add_argument('--num-folds', type=int, default=5, help="Number of folds.")
    args = parser.parse_args()

    base_dir = ensure_dir(Path(args.output_dir) / args.experiment_name)
    feature_base_dir = ensure_dir(base_dir / "features")
    model_base_dir = ensure_dir(base_dir / "models")
    eval_base_dir = ensure_dir(base_dir / "evaluations")
    temp_files_dir = ensure_dir(base_dir / "temp_fold_files")

    final_aggregated_run = defaultdict(list)
    folds = []

    if args.dataset_pattern:
        all_fold_qids = {i: get_qids_from_dataset(args.dataset_pattern.format(i)) for i in range(1, args.num_folds + 1)}
        for i in range(1, args.num_folds + 1):
            test_qids = all_fold_qids[i]
            train_qids = set().union(*(all_fold_qids[j] for j in range(1, args.num_folds + 1) if i != j))
            train_qids_file = temp_files_dir / f"fold_{i}_train_qids.txt"
            test_qids_file = temp_files_dir / f"fold_{i}_test_qids.txt"
            with open(train_qids_file, 'w') as f: f.write('\n'.join(sorted(list(train_qids))))
            with open(test_qids_file, 'w') as f: f.write('\n'.join(sorted(list(test_qids))))
            folds.append(
                {'name': f"fold{i}", 'train_qids_file': str(train_qids_file), 'test_qids_file': str(test_qids_file)})
    else:
        with open(args.folds_json, 'r') as f:
            custom_folds = json.load(f)
        for fold_key, fold_data in custom_folds.items():
            train_qids_file = temp_files_dir / f"fold_{fold_key}_train_qids.txt"
            test_qids_file = temp_files_dir / f"fold_{fold_key}_test_qids.txt"
            with open(train_qids_file, 'w') as f: f.write('\n'.join(fold_data['training']))
            with open(test_qids_file, 'w') as f: f.write('\n'.join(fold_data['testing']))
            folds.append({'name': f"fold{fold_key}", 'train_qids_file': str(train_qids_file),
                          'test_qids_file': str(test_qids_file)})

    for fold in folds:
        fold_name = fold['name']
        print(f"\n{'=' * 20} Processing {fold_name} {'=' * 20}")

        feature_dir = ensure_dir(feature_base_dir / fold_name)
        model_dir = ensure_dir(model_base_dir / fold_name)
        eval_dir = ensure_dir(eval_base_dir / fold_name)

        print(f"  [1/3] Generating features for the training set...")
        safe_dataset_name = args.base_dataset_name.replace('/', '_')
        train_qids_stem = Path(fold['train_qids_file']).stem
        feature_file = feature_dir / f"{safe_dataset_name}_{train_qids_stem}_features.json.gz"

        subprocess.run([
            "python", "scripts/create_training_data.py",
            "--dataset", args.base_dataset_name,
            "--output-dir", str(feature_dir),
            "--index-path", args.index_path,
            "--lucene-path", args.lucene_path,
            "--query-ids-file", fold['train_qids_file'],
            "--run-file-path", args.run_file_path,
        ], check=True)

        print(f"  [2/3] Learning weights...")
        subprocess.run([
            "python", "scripts/train_weights.py",
            "--feature-file", str(feature_file),
            "--validation-dataset", args.base_dataset_name,
            "--output-dir", str(model_dir),
            "--query-ids-file", fold['train_qids_file'],
            "--run-file-path", args.run_file_path,
        ], check=True)

        print(f"  [3/3] Evaluating on the test set...")
        weights_file = model_dir / "learned_weights.json"
        subprocess.run([
            "python", "scripts/evaluate_model.py",
            "--weights-file", str(weights_file),
            "--dataset", args.base_dataset_name,
            "--output-dir", str(eval_dir),
            "--index-path", args.index_path,
            "--lucene-path", args.lucene_path,
            "--query-ids-file", fold['test_qids_file'],
            "--run-file-path", args.run_file_path,
            "--save-runs",
        ], check=True)

        run_file_path = eval_dir / "runs/our_method.txt"
        with open(run_file_path, 'r') as f_run:
            for line in f_run:
                qid = line.split()[0]
                final_aggregated_run[qid].append(line.strip())

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
    print(f"python -m pytrec_eval -c -m all_trec <path_to_full_qrels_file> {final_run_path}")


if __name__ == "__main__":
    main()