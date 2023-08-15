import json
import os

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def read_json(path):
    with open(path) as cs:
        classification_scores = json.load(cs)
        print(f'\n{os.path.basename(path)}\n')
        print(f'Toxic F1\t{classification_scores["classification_scores"]["prf"]["toxic"]["f1-score"]}')
        print(f'Sufficiency\t{classification_scores["classification_scores"]["sufficiency"]}')
        print(f'Comprehensiveness\t{classification_scores["classification_scores"]["comprehensiveness"]}')
        print(f'IOU F1\t{classification_scores["iou_scores"][0]["macro"]["f1"]}')
        print(f'Token F1\t{classification_scores["token_prf"]["instance_macro"]["f1"]}')
        print(f'AUPRC\t{classification_scores["token_soft_metrics"]["auprc"]}')


def scores(folders, mixmatch=False):
    majority = {}
    minority = {}
    pure = {}
    all_on_pure = {}
    pure_on_all = {}
    for folder in folders:
        paths = sorted(os.listdir(folder))
        for p in paths:
            if not p.endswith("json"):
                continue
            with open(os.path.join(folder, p)) as explanation:
                if not mixmatch:
                    if "majority" in p and "all" in p:
                        majority[p] = json.load(explanation)
                    elif "minority" in p and "pure" in p:
                        pure[p] = json.load(explanation)
                    elif "minority" in p and "all" in p:
                        minority[p] = json.load(explanation)
                else:
                    if p.index("all") < p.index("pure"):
                        all_on_pure[p] = json.load(explanation)
                    else:
                        pure_on_all[p] = json.load(explanation)
    if not mixmatch:
        return majority, minority, pure
    else:
        return all_on_pure, pure_on_all


def explainability(dictionary, model_type):
    plaus = {}
    faith = {}
    name_replacement = {
        "birnnscrat": "BiRNN-Att true",
        "birnnatt": "BiRNN-Att false",
        "birnn": "BiRNN",
        "cnn_gru": "CNN-GRU",
        "TRUE": "BERT true",
        "FALSE": "BERT false",
        "all_bow": "BOW",
        "all_graph": "Graph",
        "rationale_bow": "Rat. BOW",
        "rationale_graph": "Rat. Graph",
        "feature_graph": "Feat. Graph",
        "hand": "Hand Rules"
    }
    for filename, score in dictionary.items():
        for finding, replacement in name_replacement.items():
            if finding in filename:
                if "lime" in filename:
                    name = f"{replacement} Lime"
                    break
                elif "top5" in filename:
                    name = f"{replacement} Rationale"
                    break
                else:
                    name = replacement
                    break
        if "keep_neutral" in filename:
            faith[name] = {}
            faith[name]["Suff."] = score["classification_scores"]["sufficiency"]
            faith[name]["Comp."] = score["classification_scores"]["comprehensiveness"]
        else:
            plaus[name] = {}
            plaus[name]["IOU F1"] = score["iou_scores"][0]["macro"]["f1"] * 100
            plaus[name]["Token F1"] = score["token_prf"]["instance_macro"]["f1"] * 100
    df_plaus = pd.read_json(json.dumps(plaus)).T
    df_faith = pd.read_json(json.dumps(faith)).T
    df = pd.concat({"Plausibility": df_plaus, "Faithfulness": df_faith}, axis=1)
    return df.to_latex(index=True, float_format="{:.2f}".format,
                       caption=f"Model explainability on {model_type}",
                       label=f"tab:explainability_{model_type}").strip()


def performance(dictionary_majority, dictionary_minority, dictionary_filtered):
    majority = {k: v for (k, v) in dictionary_majority.items() if "keep_neutral" in k}
    minority = {k: v for (k, v) in dictionary_minority.items() if "keep_neutral" in k}
    filtered = {k: v for (k, v) in dictionary_filtered.items() if "keep_neutral" in k}
    name_replacement = {
        "birnnscrat": "BiRNN-Att true",
        "birnnatt": "BiRNN-Att false",
        "birnn": "BiRNN",
        "cnn_gru": "CNN-GRU",
        "TRUE": "BERT true",
        "FALSE": "BERT false",
        "all_bow": "BOW",
        "all_graph": "Graph",
        "rationale_bow": "Rat. BOW",
        "rationale_graph": "Rat. Graph",
        "feature_graph": "Feat. Graph",
        "hand": "Hand Rules"
    }
    quant_majority = {}
    quant_minority = {}
    quant_filtered = {}
    for filename_majority, filename_minority, filename_filtered in zip(majority, minority, filtered):
        for finding, replacement in name_replacement.items():
            if finding in filename_majority:
                name = replacement
                break
        quant_majority[name] = majority[filename_majority]["classification_scores"]["prf"]["toxic"]
        quant_minority[name] = minority[filename_minority]["classification_scores"]["prf"]["toxic"]
        quant_filtered[name] = filtered[filename_filtered]["classification_scores"]["prf"]["toxic"]
    df_majority = pd.read_json(json.dumps(quant_majority)).T.drop("support", axis=1) * 100
    df_minority = pd.read_json(json.dumps(quant_minority)).T.drop("support", axis=1) * 100
    df_filtered = pd.read_json(json.dumps(quant_filtered)).T.drop("support", axis=1) * 100
    df = pd.concat({"Majority": df_majority, "Minority": df_minority, "Filtered": df_filtered}, axis=1)
    df = df.rename(columns={"f1-score": "F1", "precision": "Precision", "recall": "Recall"})
    return df.to_latex(index=True, float_format="{:.2f}".format, caption="Model performance", label="tab:performance").strip()


if __name__ == "__main__":
    folders = ["../models_explain"]
    maj, mino, pur = scores(folders)
    print("\n\\subsection{Performance}\n")
    type_name = ["majority", "minority", "filtered"]
    print(performance(maj, mino, pur))
    print("\n\\subsection{Explainability metrics}\n")
    for i, n in zip([maj, mino, pur], type_name):
        print(f"{explainability(i, n)}\n")

