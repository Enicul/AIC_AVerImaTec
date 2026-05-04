"""Run AIC AVerImaTeC CoT generation with local Qwen and JSON-constrained decoding."""

import argparse
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import nltk
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT_DEFAULT = Path("/home/aied_test/AIC_AVerImaTeC")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--model_path",
        default="/home/aied_test/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--vector_store_path",
        default=(
            "/home/aied_test/VILLAIN/dataset/AVerImaTeC_Shared_Task/Vector_Store/val/"
            "text_related/text_related_store_text_val_filled_0d3B"
        ),
    )
    parser.add_argument(
        "--image_store_path",
        default=(
            "/home/aied_test/VILLAIN/dataset/AVerImaTeC-Filled/Knowledge_Store/val/"
            "text_related/image_related_store_text_val_filled"
        ),
    )
    parser.add_argument("--retrieval_k", type=int, default=9)
    parser.add_argument("--fewshot_k", type=int, default=3)
    parser.add_argument("--max_source_chars", type=int, default=700)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--constrained", action="store_true", default=False)
    parser.add_argument("--include_claim_images", action="store_true", default=False)
    return parser.parse_args()


def build_json_schema(source_ids):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "reasoning",
            "questions",
            "claim_veracity",
            "veracity_verdict",
            "verdict_justification",
        ],
        "properties": {
            "reasoning": {"type": "string"},
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["question", "answer", "source", "answer_type", "evidence_text"],
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "source": {"type": "string", "enum": source_ids},
                        "answer_type": {
                            "type": "string",
                            "enum": ["Boolean", "Extractive", "Abstractive", "Unanswerable"],
                        },
                        "evidence_text": {"type": "string"},
                    },
                },
            },
            "claim_veracity": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "Supported",
                    "Refuted",
                    "Not Enough Evidence",
                    "Conflicting Evidence/Cherrypicking",
                ],
                "properties": {
                    "Supported": {"type": "string", "enum": ["1", "2", "3", "4", "5"]},
                    "Refuted": {"type": "string", "enum": ["1", "2", "3", "4", "5"]},
                    "Not Enough Evidence": {
                        "type": "string",
                        "enum": ["1", "2", "3", "4", "5"],
                    },
                    "Conflicting Evidence/Cherrypicking": {
                        "type": "string",
                        "enum": ["1", "2", "3", "4", "5"],
                    },
                },
            },
            "veracity_verdict": {
                "type": "string",
                "enum": [
                    "Supported",
                    "Refuted",
                    "Not Enough Evidence",
                    "Conflicting Evidence/Cherrypicking",
                ],
            },
            "verdict_justification": {"type": "string"},
        },
    }


def source_ids_for_retrieval(retrieval):
    source_ids = [str(i + 1) for i in range(len(retrieval.documents))]
    for image_idx, image_sources in enumerate(retrieval.images):
        for source_idx, _ in enumerate(image_sources):
            source_ids.append(str((image_idx + 1) * 10 + source_idx + 1))
    return source_ids or ["1"]


def load_datapoints(root, split, limit):
    from averitec import Datapoint

    data_path = root / "data" / "data_clean" / "split_data" / f"{split}.json"
    raw = json.load(open(data_path))
    if limit is not None:
        raw = raw[:limit]
    datapoints = []
    for i, item in enumerate(raw):
        item = dict(item)
        item.setdefault("claim_id", i)
        item.setdefault("split", split)
        datapoints.append(Datapoint.from_dict(item))
    return datapoints


def build_summary(records):
    return {
        "n": len(records),
        "errors": sum(1 for r in records if "error" in r),
        "parsed_ok": sum(1 for r in records if r.get("parsed_ok")),
        "parse_errors": sum(1 for r in records if r.get("parse_error")),
        "reasoning_nonempty": sum(1 for r in records if (r.get("reasoning") or "").strip()),
        "total_questions": sum(len(r.get("questions") or []) for r in records),
        "total_evidence": sum(len(r.get("evidence") or []) for r in records),
        "no_answer_evidence": sum(
            1
            for r in records
            for e in (r.get("evidence") or [])
            if "no answer could be found" in str(e).lower() or "no answer" in str(e).lower()
        ),
        "image_tag_evidence": sum(
            1 for r in records for e in (r.get("evidence") or []) if "[IMG" in str(e)
        ),
        "source_resolved_evidence": sum(
            1 for r in records for e in (r.get("evidence") or []) if e.get("url")
        ),
        "rough_verdict_correct": sum(1 for r in records if r.get("verdict") == r.get("gold")),
        "verdicts": [r.get("verdict") for r in records],
        "gold": [r.get("gold") for r in records],
    }


def save_outputs(output_dir, records, updated_results):
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(output_dir / "records.json", "w"), indent=2)
    pickle.dump(updated_results, open(output_dir / "pipeline_results.pkl", "wb"))

    submission = []
    for result in updated_results:
        try:
            submission.append(result.to_submission())
        except Exception as exc:
            print("submission_convert_error", getattr(result.datapoint, "claim_id", None), repr(exc))
    json.dump(submission, open(output_dir / "submission.json", "w"), indent=2)
    json.dump(build_summary(records), open(output_dir / "summary.json", "w"), indent=2)


def main():
    args = parse_args()
    sys.path.insert(0, str(args.root / "src"))

    from classification import NoTiebreakClassifier
    from evidence_generation import DynamicFewShotBatchedEvidenceGenerator
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from pipeline import PipelineResult
    from retrieval import CustomVectorStoreRetriever

    prefix_builder = None
    json_parser_cls = None
    if args.constrained:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        prefix_builder = build_transformers_prefix_allowed_tokens_fn
        json_parser_cls = JsonSchemaParser

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("constrained=", args.constrained, flush=True)
    print("include_claim_images=", args.include_claim_images, flush=True)
    print("max_source_chars=", args.max_source_chars, flush=True)

    datapoints = load_datapoints(args.root, args.split, args.limit)
    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={"device": "cpu"},
    )
    retriever = CustomVectorStoreRetriever(
        path=args.vector_store_path,
        embeddings=embeddings,
        k=args.retrieval_k,
        image_store_path=args.image_store_path,
    )
    generator = DynamicFewShotBatchedEvidenceGenerator(
        model=args.model_id,
        reference_corpus_path=str(args.root / "data" / "data_clean" / "split_data" / "train.json"),
        k=args.fewshot_k,
        include_claim_images=args.include_claim_images,
        max_source_chars=args.max_source_chars,
    )
    classifier = NoTiebreakClassifier()

    prepared = []
    for datapoint in tqdm(datapoints, desc="prepare-prompts"):
        retrieval = retriever(datapoint)
        scores = generator.bm25.get_scores(nltk.word_tokenize(datapoint.claim))
        top_n = np.argsort(scores)[::-1][: generator.k]
        few_shot_examples = [generator.reference_corpus[i] for i in top_n]
        system_prompt = generator.format_system_prompt(
            retrieval, few_shot_examples, datapoint.speaker, datapoint.claim_date
        )
        user_content = [{"type": "text", "text": datapoint.claim}]
        prepared.append((datapoint, retrieval, system_prompt, user_content))

    first_prompt = prepared[0][2]
    print("prepared_requests=", len(prepared), flush=True)
    print("first_prompt_chars=", len(first_prompt), flush=True)
    print("first_prompt_text_source_count=", first_prompt.count("## Source ID:"), flush=True)
    print(
        "first_prompt_image_source_count=", first_prompt.count("## Image Source ID:"), flush=True
    )
    print(
        "image_source_bucket_sizes=",
        [[len(bucket) for bucket in retrieval.images] for _, retrieval, _, _ in prepared[:10]],
        flush=True,
    )

    print("loading_qwen...", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=64 * 28 * 28, max_pixels=64 * 28 * 28
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    model.eval()
    print("qwen_loaded_device=", model.device, flush=True)

    records = []
    updated_results = []
    for datapoint, retrieval, system_prompt, user_content in tqdm(prepared, desc="generate"):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "repetition_penalty": 1.08,
                "no_repeat_ngram_size": 8,
            }
            if args.constrained:
                schema = build_json_schema(source_ids_for_retrieval(retrieval))
                parser = json_parser_cls(schema)
                generation_kwargs["prefix_allowed_tokens_fn"] = prefix_builder(
                    processor.tokenizer, parser
                )

            with torch.no_grad():
                generated_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            pipeline_result = PipelineResult(datapoint=datapoint, retrieval_result=retrieval)
            updated = generator.update_pipeline_result(pipeline_result, raw_text, classifier)
            updated_results.append(updated)

            metadata = updated.evidence_generation_result.metadata
            parsed = metadata.get("llm_output", {})
            record = {
                "claim_id": datapoint.claim_id,
                "claim": datapoint.claim,
                "gold": datapoint.label,
                "raw_output": raw_text,
                "parsed_ok": bool(parsed),
                "parse_error": metadata.get("parse_error"),
                "reasoning": metadata.get("reasoning", ""),
                "verdict": str(updated.classification_result)
                if updated.classification_result
                else None,
                "questions": parsed.get("questions", []) if isinstance(parsed, dict) else [],
                "evidence": [e.to_dict() for e in updated.evidence_generation_result],
                "justification": updated.evidence_generation_result.justification,
                "text_sources": len(retrieval.documents),
                "image_source_buckets": [len(bucket) for bucket in retrieval.images],
            }
            records.append(record)
            print(
                "done",
                datapoint.claim_id,
                "parsed",
                record["parsed_ok"],
                "questions",
                len(record["questions"]),
                "evidence",
                len(record["evidence"]),
                "verdict",
                record["verdict"],
                "gold",
                datapoint.label,
                flush=True,
            )
        except Exception as exc:
            traceback.print_exc()
            records.append(
                {
                    "claim_id": datapoint.claim_id,
                    "claim": datapoint.claim,
                    "gold": datapoint.label,
                    "error": repr(exc),
                }
            )
        save_outputs(args.output_dir, records, updated_results)

    print("summary=", json.dumps(build_summary(records), indent=2), flush=True)


if __name__ == "__main__":
    main()
