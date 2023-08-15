import json
import os


def keep_train(base_path, out_path):
    jsons = os.listdir(base_path)
    for path in jsons:
        if not path.endswith(".json"):
            continue
        with open(os.path.join(base_path, path)) as json_path:
             with open(os.path.join(out_path, path), "w") as out_json_path:
                 for line in json_path:
                     jd = json.loads(line)
                     if jd["classification"] == "toxic":
                         out_json_path.write(json.dumps(jd))
                         out_json_path.write("\n")


if __name__ == "__main__":
    in_path = "explanations_dicts"
    out_path = "explanations_dicts/predicted_true"
    #in_path = "rule_explanations"
    #out_path = "rule_explanations/predicted_true"
    keep_train(in_path, out_path)
