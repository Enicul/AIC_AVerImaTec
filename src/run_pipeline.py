"""Entry point for running the AVerImaTeC pipeline with a custom vector store."""

import argparse
import json
import os
import pickle
import sys

# Ensure the src directory is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from averitec import Datapoint
from retrieval import CustomVectorStoreRetriever
from evidence_generation import GptEvidenceGenerator, DynamicFewShotEvidenceGenerator
from classification import DefaultClassifier, NoTiebreakClassifier
from pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AVerImaTeC pipeline.")
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Root path of the project (contains data/, src/, etc.)",
    )
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default=None,
        help="Path to the custom vector store directory (per-claim subdirectories).",
    )
    parser.add_argument(
        "--knowledge_store_path",
        type=str,
        default=None,
        help="Legacy: path to the knowledge store (not used by CustomVectorStoreRetriever).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to run on.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run on a small subset for debugging.",
    )
    parser.add_argument(
        "--save_num",
        type=int,
        default=1,
        help="Save a checkpoint every N claims.",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="gpt-4o",
        help="LLM name / model identifier for evidence generation.",
    )
    parser.add_argument(
        "--mllm_name",
        type=str,
        default="gpt-4o",
        help="Multi-modal LLM name (currently unused in this runner, kept for compatibility).",
    )
    parser.add_argument(
        "--reference_corpus_path",
        type=str,
        default=None,
        help="Path to the train.json reference corpus for few-shot evidence generation. "
             "If not provided, few-shot examples are skipped.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save pipeline results pickle. Defaults to "
             "{root_path}/output/pipeline_results_{split}.pkl",
    )
    parser.add_argument(
        "--ris_path",
        type=str,
        default=None,
        help="Path to reverse-image-search results directory. If not provided, RIS is skipped.",
    )
    parser.add_argument(
        "--image_store_path",
        type=str,
        default=None,
        help=(
            "Path to prepared per-claim image-related/RIS text store. "
            "This can be used instead of --ris_path for the AIC shared-task setup."
        ),
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing claim images. Falls back to IMAGES_DIR env var.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=9,
        help="Number of documents to retrieve per claim.",
    )
    return parser.parse_args()


def load_datapoints(data_path: str, split: str, debug: bool = False):
    json_path = os.path.join(data_path, f"{split}.json")
    with open(json_path, "r") as f:
        raw = json.load(f)
    if debug:
        raw = raw[:5]
    datapoints = []
    for i, item in enumerate(raw):
        if "claim_id" not in item:
            item["claim_id"] = i
        if "split" not in item:
            item["split"] = split
        datapoints.append(Datapoint.from_dict(item))
    return datapoints


def main():
    args = parse_args()

    # Propagate images_dir to environment so evidence_generation picks it up
    if args.images_dir:
        os.environ["IMAGES_DIR"] = args.images_dir

    data_path = os.path.join(args.root_path, "data", "data_clean", "split_data")
    datapoints = load_datapoints(data_path, args.split, args.debug)
    print(f"Loaded {len(datapoints)} datapoints from {args.split} split.")

    # Build retriever
    if args.vector_store_path is None:
        raise ValueError("--vector_store_path is required.")
    retriever = CustomVectorStoreRetriever(
        path=args.vector_store_path,
        k=args.k,
        ris_path=args.ris_path,
        image_store_path=args.image_store_path,
    )

    # Build evidence generator
    evidence_generator = DynamicFewShotEvidenceGenerator(
        model=args.llm_name,
        reference_corpus_path=args.reference_corpus_path,
    )

    # Build classifier (DefaultClassifier reads verdict from evidence generator metadata)
    classifier = NoTiebreakClassifier()

    pipeline = Pipeline(
        retriever=retriever,
        evidence_generator=evidence_generator,
        classifier=classifier,
    )

    # Determine output path
    output_path = args.output_path
    if output_path is None:
        out_dir = os.path.join(args.root_path, "output")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"pipeline_results_{args.split}.pkl")

    results = []
    for i, datapoint in enumerate(datapoints):
        print(f"[{i+1}/{len(datapoints)}] Processing claim {datapoint.claim_id}: {datapoint.claim[:80]}")
        try:
            result = pipeline(datapoint)
            results.append(result)
        except Exception as e:
            print(f"  ERROR on claim {datapoint.claim_id}: {e}")
            results.append(None)

        if args.save_num > 0 and (i + 1) % args.save_num == 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)
            print(f"  Checkpoint saved to {output_path}")

    # Final save
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Done. Results saved to {output_path}")

    # Also save a JSON submission file
    submission_path = output_path.replace(".pkl", "_submission.json")
    submission = []
    for r in results:
        if r is not None:
            try:
                submission.append(r.to_submission())
            except Exception as e:
                print(f"  Warning: could not convert result to submission format: {e}")
    with open(submission_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission JSON saved to {submission_path}")


if __name__ == "__main__":
    main()
