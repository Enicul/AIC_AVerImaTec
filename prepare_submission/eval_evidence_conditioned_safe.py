import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ref_eval import val_evid_idv, compute_image_scores, textual_val_single
import utils

import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)


def is_placeholder(pred):
    return (not (pred.get('questions') or [])) and (not (pred.get('evidence') or [])) and (not (pred.get('justification') or '').strip())


def zero_result(note):
    return {
        'ques_score': 0.0,
        'evid_score': 0.0,
        'verdict_score': 0.0,
        'justi_score': 0.0,
        'intermediate_info': {
            'note': note,
            'ques_feedback': None,
            'ques_score': [],
            'justi_feedback': None,
            'justi_score': [],
            'evid_feedback': [],
            'evid_image_score': [],
            'evid_text_score': [],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', required=True)
    ap.add_argument('--pred_file_path', required=True)
    ap.add_argument('--llm_name', default='qwen')
    ap.add_argument('--mllm_name', default='qwen')
    ap.add_argument('--save_num', required=True)
    ap.add_argument('--mode', choices=['ev', 'ej'], required=True, help='ev computes E x V; ej computes E x J')
    ap.add_argument('--eval_model', default='google/gemma-3-27b-it')
    ap.add_argument('--cache_dir', default='')
    ap.add_argument('--save_every', type=int, default=5)
    ap.add_argument('--evidence_threshold', type=float, default=0.3)
    args = ap.parse_args()

    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.eval_model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
    processor = AutoProcessor.from_pretrained(args.eval_model, cache_dir=args.cache_dir)
    mllm = {'model': model.eval(), 'processor': processor}
    mllm_name = args.eval_model

    val_path = os.path.join(args.root_dir, 'val.json')
    if not os.path.exists(val_path):
        val_path = os.path.join(args.root_dir, 'data/data_clean/split_data/val.json')
    p2_data = load_json(val_path)
    pred_file = load_json(args.pred_file_path)
    gt_evid_path = os.path.join(args.root_dir, 'prepare_submission/converted_results/gt_evid.json')
    if not os.path.exists(gt_evid_path):
        raise FileNotFoundError(f'Expected precomputed gt evidence at {gt_evid_path}')
    gt_evid_set = load_json(gt_evid_path)

    out_path = os.path.join(
        args.root_dir,
        'prepare_submission/intermediate_eval_results',
        '_'.join([args.llm_name, args.mllm_name, str(args.save_num)]) + '.json',
    )
    if os.path.exists(out_path):
        results = load_json(out_path)
    else:
        results = []

    print('Root dir:', args.root_dir)
    print('Mode:', args.mode, 'resume_rows:', len(results), 'out:', out_path, flush=True)

    for i, row in enumerate(p2_data):
        if i < len(results):
            continue
        if i % args.save_every == 0:
            print(i, 'saving...', flush=True)
            dump_json(results, out_path)

        pred = pred_file[i]
        if is_placeholder(pred):
            print(i, 'placeholder -> zero scores', flush=True)
            results.append(zero_result('placeholder converted row'))
            continue

        pred_evid = pred.get('evidence') or []
        gt_evid = gt_evid_set[i]
        pred_verdict = pred.get('verdict') or ''
        gt_verdict = row['label']
        pred_justi = pred.get('justification') or ''
        gt_justification = row['justification']

        verdict_acc = 1.0 if pred_verdict.lower().strip() == gt_verdict.lower() else 0.0

        if pred_evid:
            detailed_evid_val, evid_val_score = val_evid_idv(mllm, mllm_name, pred_evid, gt_evid, False, True)
            img_scores = compute_image_scores(mllm, mllm_name, pred_evid, gt_evid, evid_val_score)
            _, evid_acc, _ = utils.get_auto_recall(detailed_evid_val, img_scores, len(gt_evid), len(pred_evid))
        else:
            detailed_evid_val, evid_val_score, img_scores, evid_acc = [], [], [], 0.0

        justi_feedback, justi_score, justi_acc = None, [], 0.0
        if args.mode == 'ej' and evid_acc > args.evidence_threshold and pred_justi.strip():
            justi_feedback, justi_score = textual_val_single(
                gt_justification, pred_justi, args.root_dir, mllm_name, mllm, 'justification', False
            )
            justi_acc = utils.justi_recall_compute(justi_feedback, justi_score)

        results.append({
            'ques_score': 0.0,
            'evid_score': evid_acc,
            'verdict_score': verdict_acc,
            'justi_score': justi_acc,
            'intermediate_info': {
                'mode': args.mode,
                'ques_feedback': None,
                'ques_score': [],
                'justi_feedback': justi_feedback,
                'justi_score': justi_score,
                'evid_feedback': detailed_evid_val,
                'evid_image_score': img_scores,
                'evid_text_score': evid_val_score,
            },
        })

    dump_json(results, out_path)
    n = len(results)
    exv = sum(((x.get('verdict_score') or 0) if (x.get('evid_score') or 0) > args.evidence_threshold else 0) for x in results) / n
    exj = sum(((x.get('justi_score') or 0) if (x.get('evid_score') or 0) > args.evidence_threshold else 0) for x in results) / n
    print(f'DONE n={n} E×V={exv:.6f} E×J={exj:.6f}', flush=True)


if __name__ == '__main__':
    main()
