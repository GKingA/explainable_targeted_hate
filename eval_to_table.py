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


def explainability(dictionary):
    not_names = ["model", "explain", "output", "majority", "minority", "all", "pure"]
    quant = {}
    qual = {}
    name_replacement = {"Bert False": "BERT false", "Bert True": "BERT true",
                        "Birnn": "BiRNN", "Birnnatt": "BiRNN-Att false",
                        "Birnnscrat": "BiRNN-Att true", "Cnn Gru": "CNN-GRU"}
    for filename, score in dictionary.items():
        name = " ".join([e.capitalize() for e in filename.split(".")[0].split("_") if e not in not_names])
        base = " ".join(name.split()[:-1])
        expl = name.split()[-1]
        if base in name_replacement:
            name = " ".join([name_replacement[base], expl])
        quant[name] = {}
        qual[name] = {}
        quant[name]["Model"] = name
        qual[name]["Model"] = name
        quant[name]["IOU F1"] = score["iou_scores"][0]["macro"]["f1"] * 100
        quant[name]["Token F1"] = score["token_prf"]["instance_macro"]["f1"] * 100
        qual[name]["Suff."] = score["classification_scores"]["sufficiency"]
        qual[name]["Comp."] = score["classification_scores"]["comprehensiveness"]
    df = pd.read_json(json.dumps(quant)).T.sort_values('Model')
    df_qual = pd.read_json(json.dumps(qual)).T.sort_values('Model')
    return df.to_latex(index = False, float_format="{:.2f}%".format), df_qual.to_latex(index = False, float_format="{:.2f}".format)


def performance(dictionary):
    not_names = ["model", "explain", "output", "majority", "minority", "all", "pure"]
    name_replacement = {"Bert False": "BERT false", "Bert True": "BERT true",
                        "Birnn": "BiRNN", "Birnnatt": "BiRNN-Att false",
                        "Birnnscrat": "BiRNN-Att true", "Cnn Gru": "CNN-GRU"}
    quant = {}
    for filename, score in dictionary.items():
        name = " ".join([e.capitalize() for e in filename.split(".")[0].split("_") if e not in not_names])
        base = " ".join(name.split()[:-1])
        expl = name.split()[-1]
        if base in name_replacement:
            name = " ".join([name_replacement[base], expl])
        quant[name] = score["classification_scores"]["prf"]["toxic"]
        quant[name]["Model"] = name
    df = pd.read_json(json.dumps(quant)).T.sort_values('Model')
    df = df[["Model", "precision", "recall", "f1-score"]]
    df.precision = df.precision * 100
    df.recall = df.recall * 100
    df["f1-score"] = df["f1-score"] * 100
    return df.to_latex(index=False, float_format="{:.2f}%".format)


if __name__ == "__main__":
    folders = ["fair_eval"]
    maj, mino, pur = scores(folders)
    print("\n\\subsection{Performance}")
    name = ["majority", "minority", "filtered"]
    for i, n in zip([maj, mino, pur], name):
        print("\n\\begin{table}[!ht]")
        print(performance(i))
        print(f"\\caption{{Model performance on {n}}}\n\\label{{tab:{n}_perf}}\n\\end{{table}}")
    print("\n\\subsection{Explainability metrics}\n")
    print("\\subsubsection{Plausibility}")
    for i, n in zip([maj, mino, pur], name):
        qualitative_s = explainability(i)
        print("\n\\begin{table}[!ht]")
        print(qualitative_s[0])
        print(f"\\caption{{Model plausibility on {n}}}\n\\label{{tab:{n}_plaus}}\n\\end{{table}}")
    print("\\subsubsection{Faithfulness}")
    for i, n in zip([maj, mino, pur], name):
        qualitative_s = explainability(i)
        print("\n\\begin{table}[!ht]")
        print(qualitative_s[1])
        print(f"\\caption{{Model faithfulness on {n}}}\n\\label{{tab:{n}_faith}}\n\\end{{table}}")
