import pickle
import json
import os
import argparse

def load_pkl(path):
    return pickle.load(open(path, "rb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    data = load_pkl(args.pkl_path)

    results = []
    for i in sorted(data.keys()):
        info = data[i]
        questions = [qa["question"] for qa in info.get("QA_info", [])]
        evidence = info.get("evidence", info.get("evid_context", []))
        verdict = info.get("verdict", "Not Enough Evidence")
        justification = info.get("justification", "")
        results.append({
            "questions": questions,
            "evidence": evidence,
            "verdict": verdict,
            "justification": justification,
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    json.dump(results, open(args.output_path, "w"), indent=2)
    print(f"Converted {len(results)} claims to {args.output_path}")
