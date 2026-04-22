import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="/home/aied_test/AIC_AVerImaTeC")
parser.add_argument("--llm_name", default="qwen")
parser.add_argument("--mllm_name", default="qwen")
parser.add_argument("--save_num", default="1")
args = parser.parse_args()

path = f"{args.root_dir}/prepare_submission/intermediate_eval_results/{args.llm_name}_{args.mllm_name}_{args.save_num}.json"
r = json.load(open(path))

n = len(r)
verdict = sum(x["verdict_score"] for x in r) / n
evid = [x["evid_score"] for x in r if x["evid_score"] is not None]
ques = sum(x["ques_score"] for x in r) / n
justi = sum(x["justi_score"] for x in r) / n

print(f"Progress: {n} / 152 claims evaluated")
print(f"Verdict accuracy: {verdict:.3f}")
print(f"Evidence score:   {sum(evid)/n:.3f} ({sum(1 for e in evid if e > 0)} non-zero)")
print(f"Question score:   {ques:.3f}")
print(f"Justification:    {justi:.3f}")
