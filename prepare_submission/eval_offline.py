import json
import os
import pickle as pkl
import random
import argparse

import sys

sys.path.append("..")
root_dir = os.path.abspath("/mnt/personal/ullriher/aic_averimatec")
from ref_eval import val_evid_idv, compute_image_scores, textual_val_single
from qa_to_evidence import qa_to_evid
from utils import convert_qa_format
import utils

import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def load_pkl(path):
    data = pkl.load(open(path, "rb"))
    return data


def load_json(path):
    data = json.load(open(path, "r"))
    return data


def is_placeholder_prediction(pred):
    return (
        not (pred.get("questions") or [])
        and not (pred.get("evidence") or [])
        and not (pred.get("justification") or "").strip()
    )


def zero_eval_result(note):
    return {
        "ques_score": 0.0,
        "evid_score": 0.0,
        "verdict_score": 0.0,
        "justi_score": 0.0,
        "intermediate_info": {
            "note": note,
            "ques_feedback": None,
            "ques_score": [],
            "justi_feedback": None,
            "justi_score": [],
            "evid_feedback": [],
            "evid_image_score": [],
            "evid_text_score": [],
        },
    }


import re






class _OpenRouterGenerateContentResponse:
    def __init__(self, payload):
        self.payload = payload
        self.text = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


class _OpenRouterGenerateContentModels:
    def __init__(self, api_key, model_name="google/gemini-2.5-flash", timeout=180):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.referer = os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost")
        self.title = os.environ.get("OPENROUTER_X_TITLE", "AVerImaTeC evaluation")

    def _image_data_url(self, image):
        import base64, io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _message_content(self, contents):
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, str):
                    parts.append({"type": "text", "text": item})
                else:
                    try:
                        from PIL import Image
                        if isinstance(item, Image.Image):
                            parts.append({"type": "image_url", "image_url": {"url": self._image_data_url(item)}})
                            continue
                    except Exception:
                        pass
                    parts.append({"type": "text", "text": str(item)})
            return parts
        return str(contents)

    def generate_content(self, model=None, contents=None):
        import random, time, requests
        payload = {"model": self.model_name, "messages": [{"role": "user", "content": self._message_content(contents)}]}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }
        last_err = None
        for attempt in range(8):
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"openrouter HTTP {resp.status_code}: {resp.text[:500]}")
                retry_after = resp.headers.get("retry-after")
                wait = min(120, (2 ** attempt) + random.random())
                if retry_after:
                    try: wait = max(wait, float(retry_after))
                    except ValueError: pass
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return _OpenRouterGenerateContentResponse(resp.json())
        raise last_err


class OpenRouterGenerateContentClient:
    def __init__(self, api_key, model_name="google/gemini-2.5-flash"):
        self.models = _OpenRouterGenerateContentModels(api_key, model_name)

class _RateLimitedModels:
    def __init__(self, models, min_interval_seconds):
        self._models = models
        self._min_interval_seconds = float(min_interval_seconds or 0)
        self._last_call = 0.0

    def generate_content(self, *args, **kwargs):
        import time
        if self._min_interval_seconds > 0:
            now = time.time()
            wait = self._min_interval_seconds - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
        result = self._models.generate_content(*args, **kwargs)
        self._last_call = time.time()
        return result


class _RateLimitedClient:
    def __init__(self, client, min_interval_seconds):
        self._client = client
        self.models = _RateLimitedModels(client.models, min_interval_seconds)


def apply_genai_rate_limit(client):
    interval = os.environ.get("GENAI_MIN_INTERVAL_SECONDS")
    if interval and hasattr(client, "models"):
        return _RateLimitedClient(client, float(interval))
    return client

class _CustomGenerateContentResponse:
    def __init__(self, payload):
        self.payload = payload
        self.text = self._extract_text(payload)

    @staticmethod
    def _extract_text(payload):
        parts = []
        for cand in payload.get("candidates", []) or []:
            content = cand.get("content", {}) or {}
            for part in content.get("parts", []) or []:
                if "text" in part:
                    parts.append(part.get("text") or "")
        return "\n".join(parts).strip()


class _CustomGenerateContentModels:
    def __init__(self, endpoint, api_key, timeout=120):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def _part(self, item):
        if isinstance(item, str):
            return {"text": item}
        try:
            from PIL import Image
            if isinstance(item, Image.Image):
                import base64, io
                buf = io.BytesIO()
                item.save(buf, format="PNG")
                return {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(buf.getvalue()).decode("ascii")}}
        except Exception:
            pass
        return {"text": str(item)}

    def generate_content(self, model=None, contents=None):
        import os, random, time, requests
        if isinstance(contents, list):
            parts = [self._part(x) for x in contents]
        else:
            parts = [self._part(contents)]
        payload = {"contents": [{"role": "user", "parts": parts}]}
        headers = {"Content-Type": "application/json", "api-key": self.api_key}
        last_err = None
        for attempt in range(8):
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout, verify=(os.environ.get("GENAI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}))
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"custom genai HTTP {resp.status_code}: {resp.text[:500]}")
                wait = min(90, (2 ** attempt) + random.random())
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try: wait = max(wait, float(retry_after))
                    except ValueError: pass
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return _CustomGenerateContentResponse(resp.json())
        raise last_err


class CustomGenerateContentClient:
    def __init__(self, endpoint, api_key):
        self.models = _CustomGenerateContentModels(endpoint, api_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate extra questions based on claims with a prompt. Useful for searching."
    )
    parser.add_argument("--eval_model", default="google/gemma-3-27b-it")
    parser.add_argument("--llm_name", default="gemma")
    parser.add_argument("--mllm_name", default="gemma")
    parser.add_argument("--eval_name", default=None)
    parser.add_argument(
        "--pred_file_path", default=""
    )
    parser.add_argument(
        "--root_dir", default="/mnt/data/factcheck/averimatec"
    )  # this is the absolute path where you put AVerImaTec.
    parser.add_argument(
        "--cache_dir", default=""
    )  # this is the absolute path where you save your huggingface model
    parser.add_argument("--save_num", type=str, default="4")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    if args.eval_name is None:
        args.eval_name = "_".join([args.llm_name, args.mllm_name, str(args.save_num)])
    """
    Potential issues related to Gemma: https://github.com/google-deepmind/gemma/issues/169
    """
    mllm_name = args.eval_model
    if "gemini" in mllm_name:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        custom_endpoint = os.environ.get("GENAI_ENDPOINT")
        custom_key = os.environ.get("GENAI_SUBSCRIPTION_KEY")
        if openrouter_key:
            mllm = apply_genai_rate_limit(OpenRouterGenerateContentClient(openrouter_key, mllm_name))
        elif custom_endpoint and custom_key:
            mllm = apply_genai_rate_limit(CustomGenerateContentClient(custom_endpoint, custom_key))
        else:
            from google import genai
            from google.genai.types import HttpOptions

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY/GOOGLE_API_KEY or GENAI_ENDPOINT+GENAI_SUBSCRIPTION_KEY for Gemini evaluation")
            mllm = apply_genai_rate_limit(genai.Client(http_options=HttpOptions(api_version="v1"), api_key=api_key))
    elif "gemma" in mllm_name:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            mllm_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
            # attn_implementation="eager"
        )
        processor = AutoProcessor.from_pretrained(mllm_name, cache_dir=args.cache_dir)
        mllm = {"model": model.eval(), "processor": processor}
    else:
        raise ValueError(f"Unsupported eval_model: {mllm_name}")

    print("Root dir:", args.root_dir)
    # p2_data = load_json(os.path.join(args.root_dir, "data/data_clean/split_data/test.json"))
    val_path = os.path.join(args.root_dir, "val.json")
    if not os.path.exists(val_path):
        val_path = os.path.join(args.root_dir, "data/data_clean/split_data/val.json")
    p2_data = load_json(val_path)
    if len(args.pred_file_path) > 0:
        pred_file = load_json(args.pred_file_path)
    else:
        pred_file = load_json(
            os.path.join(
                args.root_dir,
                "prepare_submission/converted_results",
                "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
            )
        )

    if os.path.exists(os.path.join(args.root_dir, "prepare_submission/converted_results", "gt_evid.json")):
        gt_evid_flag = True
        gt_evid_set = load_json(
            os.path.join(args.root_dir, "prepare_submission/converted_results", "gt_evid.json")
        )
    else:
        gt_evid_flag = False
        gt_evid_set = []

    if os.path.exists(
        os.path.join(
            args.root_dir,
            "prepare_submission/intermediate_eval_results",
            "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
        )
    ):
        all_eval_results = load_json(
            os.path.join(
                args.root_dir,
                "prepare_submission/intermediate_eval_results",
                "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
            )
        )
    else:
        all_eval_results = []
    for i, row in enumerate(p2_data):
        if i < len(all_eval_results):
            continue
        if args.debug and i > 4:
            break
        if i % 20 == 0:
            print(i, "saving...")
            json.dump(
                all_eval_results,
                open(
                    os.path.join(
                        args.root_dir,
                        "prepare_submission/intermediate_eval_results",
                        "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
                    ),
                    "w",
                ),
            )
        req_id = i
        gt_questions = [info["question"] for info in row["questions"]]
        gt_justification = row["justification"]
        gt_verdict = row["label"]
        if gt_evid_flag:
            gt_evid = gt_evid_set[i]
        else:
            gt_evid = [convert_qa_format(qa, mllm, mllm_name, args.root_dir) for qa in row["questions"]]
            gt_evid_set.append(gt_evid)

        pred_evid = pred_file[i]["evidence"]
        pred_justi = pred_file[i]["justification"]
        pred_questions = pred_file[i]["questions"]
        pred_verdict = pred_file[i]["verdict"]

        if is_placeholder_prediction(pred_file[i]):
            all_eval_results.append(zero_eval_result("empty prediction row"))
            continue

        # verdict prediction
        if pred_verdict.lower().strip() == gt_verdict.lower():
            verdict_acc = 1.0
        else:
            verdict_acc = 0.0
        # evidence evaluation
        if pred_evid:
            detailed_evid_val, evid_val_score = val_evid_idv(mllm, mllm_name, pred_evid, gt_evid, False, True)
            img_scores = compute_image_scores(mllm, mllm_name, pred_evid, gt_evid, evid_val_score)
            _, evid_acc, _ = utils.get_auto_recall(detailed_evid_val, img_scores, len(gt_evid), len(pred_evid))
        else:
            detailed_evid_val, evid_val_score, img_scores, evid_acc = [], [], [], 0.0
        # justification generation
        if pred_justi.strip():
            justi_feedback, justi_score = textual_val_single(
                gt_justification, pred_justi, args.root_dir, mllm_name, mllm, "justification", args.debug
            )
            justi_acc = utils.justi_recall_compute(justi_feedback, justi_score)
        else:
            justi_feedback, justi_score, justi_acc = None, [], 0.0
        # question generation
        if pred_questions:
            ques_feedback, ques_score = textual_val_single(
                gt_questions, pred_questions, args.root_dir, mllm_name, mllm, "question", args.debug
            )
            ques_acc = utils.ques_recall_compute(ques_score, len(gt_questions), len(pred_questions))
        else:
            ques_feedback, ques_score, ques_acc = None, [], 0.0
        if args.debug:
            print("##Question:\n", ques_feedback, "\n", ques_score, "\n\t", ques_acc)
            print("##Verdict:\n", pred_verdict, gt_verdict, verdict_acc)
            print("##Evidence:\n", detailed_evid_val, "\n", img_scores, "\n\t", evid_acc)
            print("##Justification:\n", justi_feedback, "\n", justi_score, "\n\t", justi_acc)

        all_eval_results.append(
            {
                "ques_score": ques_acc,
                "evid_score": evid_acc,
                "verdict_score": verdict_acc,
                "justi_score": justi_acc,
                "intermediate_info": {
                    "ques_feedback": ques_feedback,
                    "ques_score": ques_score,
                    "justi_feedback": justi_feedback,
                    "justi_score": justi_score,
                    "evid_feedback": detailed_evid_val,
                    "evid_image_score": img_scores,
                    "evid_text_score": evid_val_score,
                },
            }
        )
        json.dump(
            all_eval_results,
            open(
                os.path.join(
                    args.root_dir,
                    "prepare_submission/intermediate_eval_results",
                    "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
                ),
                "w",
            ),
        )

    json.dump(
        all_eval_results,
        open(
            os.path.join(
                args.root_dir,
                "prepare_submission/intermediate_eval_results",
                "_".join([args.llm_name, args.mllm_name, str(args.save_num)]) + ".json",
            ),
            "w",
        ),
    )
    if gt_evid_flag == False:
        json.dump(
            gt_evid_set,
            open(os.path.join(args.root_dir, "prepare_submission/converted_results", "gt_evid.json"), "w"),
        )
