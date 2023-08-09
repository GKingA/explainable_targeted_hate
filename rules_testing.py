import os.path

import pandas as pd
import json
import networkx as nx
from ast import literal_eval
from typing import List, Tuple
from argparse import ArgumentParser

from nltk import WordNetLemmatizer

from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.dataset.explainable_dataset import ExplainableDataset
from tuw_nlp.graph.utils import graph_to_pn


def rationale_graph(df: pd.DataFrame):
    graphs = []
    rats = df[df.rationale == 1]
    rat_ids = [
        (literal_eval(r.rationale_lemma), r.graph)
        for (i, r) in rats.iterrows()
        if literal_eval(r.rationale_lemma) != []
    ]
    for ids, graph in rat_ids:
        g = graph.subgraph([i + 1 for i in ids])
        connected_components = list(nx.weakly_connected_components(g))
        graphs += [graph_to_pn(g.subgraph(c)) for c in connected_components]
    return graphs


def classification_score(match: pd.Series, target: str):
    classification = (
        "toxic" if target.capitalize() in match["Predicted label"] else "non-toxic"
    )
    not_class = "non-toxic" if classification == "toxic" else "toxic"
    return classification, not_class


def bow_model(
    potato_path: str,
    json_path: str,
    post_id: str,
    features: str,
    target: str,
    output_file: str,
    test: bool = False,
    delete_negative: bool=False,
) -> None:
    with open(features, "r") as feat_file:
        features = json.load(feat_file)
    with open(post_id, "r") as post_id_file:
        post_id_division = json.load(post_id_file)
    df = ExplainableDataset(
        path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}
    ).to_dataframe()
    og = pd.read_json(json_path).T
    port = "test" if test else "val"
    test_posts = og[og.post_id.isin(post_id_division[port])]
    words = features[target.capitalize()]
    lemmatizer = WordNetLemmatizer()
    with open(output_file, "w") as out:
        for (_, original), (_, potato) in zip(test_posts.iterrows(), df.iterrows()):
            if delete_negative and (potato.label != target.capitalize()):
                continue
            lemmata = [lemmatizer.lemmatize(t) for t in original["post_tokens"]]
            rats = [(lemma in words) * 1 for lemma in lemmata]
            if sum(rats) > 0:
                classification = "toxic"
                classification_scores = {"toxic": 1, "non-toxic": 0}
            else:
                classification = "non-toxic"
                classification_scores = {"toxic": 0, "non-toxic": 1}
            # Just for fairness
            wo = [lemma for lemma in lemmata if lemma not in words]
            wo_rats = [(lemma in words) * 1 for lemma in wo]
            if sum(wo_rats) > 0:
                wo_classification_scores = {"toxic": 1, "non-toxic": 0}
            else:
                wo_classification_scores = {"toxic": 0, "non-toxic": 1}
            just = [lemma for lemma in lemmata if lemma in words]
            just_rats = [(lemma in words) * 1 for lemma in just]
            if sum(just_rats) > 0:
                just_classification_scores = {"toxic": 1, "non-toxic": 0}
            else:
                just_classification_scores = {"toxic": 0, "non-toxic": 1}
            dictionary = {
                "annotation_id": original.post_id,
                "classification": classification,
                "classification_scores": classification_scores,
                "rationales": [
                    {
                        "docid": original.post_id,
                        "hard_rationale_predictions": [
                            {"end_token": index + 1, "start_token": index}
                            for index, i in enumerate(rats) if i == 1
                        ],
                        "soft_rationale_predictions": rats,
                        "truth": 1,
                    }
                ],
                "sufficiency_classification_scores": just_classification_scores,
                "comprehensiveness_classification_scores": wo_classification_scores,
            }
            out.write(f"{json.dumps(dictionary)}\n")


def graph_model(
    potato_path: str,
    json_path: str,
    post_id: str,
    features: str,
    target: str,
    output_file: str,
    test: bool = False,
    delete_negative: bool=False,
) -> Tuple[List[List[int]], List[str]]:
    df = ExplainableDataset(
        path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}
    ).to_dataframe()
    if features.endswith("json"):
        with open(features, "r") as feat_file:
            features = json.load(feat_file)
    else:
        feature_df = pd.read_csv(features, sep="\t", names=["target", "pos", "neg"], header=None)
        features = {feature_df.target[0]: []}
        for _, line in feature_df.iterrows():
            features[feature_df.target[0]].append([[line.pos], [line.neg] if not line.neg else [], line.target])
    with open(post_id, "r") as post_id_file:
        post_id_division = json.load(post_id_file)
    og = pd.read_json(json_path).T
    port = "test" if test else "val"
    test_posts = og[og.post_id.isin(post_id_division[port])]
    print(len(test_posts), len(df))
    evaluator = FeatureEvaluator()
    feat_columns = []
    feats = []
    match = evaluator.match_features(
        df,
        features[target.capitalize()],
        multi=True,
        return_subgraphs=True,
        allow_multi_graph=True,
    )
    joined = df.join(match)
    removed = []
    only_rat = []
    for subs, graph in zip(joined["Matched subgraph"], joined.graph):
        rat = nx.DiGraph()
        s_graph = graph.copy()
        for rule_sub in subs:
            for sub in rule_sub:
                s_graph.remove_nodes_from(sub.nodes)
                try:
                    rat = nx.union(rat, sub)
                except nx.exception.NetworkXError:
                    rat = nx.disjoint_union(rat, sub)
        removed.append(s_graph)
        only_rat.append(rat)
    just_rationale = joined.copy()
    just_rationale.graph = only_rat
    just_match = evaluator.match_features(
        just_rationale,
        features[target.capitalize()],
        multi=True,
        return_subgraphs=True,
        allow_multi_graph=True,
    )  # Sufficiency
    without_rationale = joined.copy()
    without_rationale.graph = removed
    without_match = evaluator.match_features(
        without_rationale,
        features[target.capitalize()],
        multi=True,
        return_subgraphs=True,
        allow_multi_graph=True,
    )  # Comprehension
    with open(output_file, "w") as out:
        for (_, m), (_, just), (_, wo), (_, original), (_, og) in zip(
            match.iterrows(),
            just_match.iterrows(),
            without_match.iterrows(),
            test_posts.iterrows(),
            df.iterrows(),
        ):
            if delete_negative and (og.label != target.capitalize()):
                continue
            classification, not_class = classification_score(m, target)
            just_classification, just_not_class = classification_score(just, target)
            wo_classification, wo_not_class = classification_score(wo, target)
            if classification == "toxic" and wo_classification == "toxic":
                breakpoint()
            rationale_ids = [
                n
                for subgraph in m["Matched subgraph"]
                for rule_graph in subgraph
                for n in rule_graph.nodes()
            ]
            # Correct issues with UD parser, that creates new nodes for < and >
            issues = [
                n[0]
                for n in og.graph.nodes(data=True)
                if n[1]["name"] == "LT" or n[1]["name"] == "GT"
            ]
            corrected_rationale_ids = []
            for rat in rationale_ids:
                n = 0
                for issue in issues:
                    if issue < rat:
                        n += 1
                corrected_rationale_ids.append(rat - n)
            dictionary = {
                "annotation_id": original.post_id,
                "classification": classification,
                "classification_scores": {classification: 1, not_class: 0},
                "rationales": [
                    {
                        "docid": original.post_id,
                        "hard_rationale_predictions": [
                            {"end_token": i, "start_token": i - 1}
                            for i in corrected_rationale_ids
                        ],
                        "soft_rationale_predictions": [
                            1 if i + 1 in corrected_rationale_ids else 0
                            for i, _ in enumerate(original.post_tokens)
                        ],
                        "truth": 1,
                    }
                ],
                "sufficiency_classification_scores": {
                    just_classification: 1,
                    just_not_class: 0,
                },
                "comprehensiveness_classification_scores": {
                    wo_classification: 1,
                    wo_not_class: 0,
                },
            }
            out.write(f"{json.dumps(dictionary)}\n")

    for feature in features[target.capitalize()]:
        match = evaluator.match_features(
            df, [feature], multi=True, return_subgraphs=True, allow_multi_graph=True
        )
        feats.append(str(feature))
        feat_columns.append(
            [
                (r["Predicted label"] == target.capitalize()) * 1
                for (_, r) in match.iterrows()
            ]
        )
    feature_vectors = list(map(list, zip(*feat_columns)))
    return feature_vectors, feats


if __name__ == "__main__":
    parse_args = ArgumentParser()
    parse_args.add_argument("--config", "-c", help="A particular config file or a path to the config files. "
                                                   "If it's a directory, all of the configs")
    args = parse_args.parse_args()
    if os.path.isdir(args.config):
        list_of_configs = [os.path.join(args.config, o) for o in os.listdir(args.config)]
    else:
        list_of_configs = [args.config]

    for conf in list_of_configs:
        with open(conf) as config_file:
            config = json.load(config_file)
        voting = config["voting"]
        filtering = config["filtering"]
        val = "val" if not config["test"] else "test"
        base_path = config["base_path"]
        base_original = config["base_original"]
        method = config["method"]
        in_feats = config["features"]
        post_id_divisions = config["post_id"]
        target = config["target"]
        delete_negative = config["delete_negative"]
        output = f"explanation_dict_{method}_{voting}_{filtering}.json" if "output" not in config else config["output"]
        if os.path.isdir(base_path):
            validation_file = os.path.join(base_path, f"{voting}_{val}_{filtering}.tsv")
        else:
            validation_file = base_path

        if os.path.isdir(base_original):
            original_file = os.path.join(base_original, f"{voting}_{filtering}.json")
        else:
            original_file = base_original

        if "graph" in method:
            feat, names = graph_model(
                validation_file,
                original_file,
                post_id_divisions,
                in_feats,
                target,
                output,
                delete_negative=delete_negative,
            )
        else:
            bow_model(
                validation_file,
                original_file,
                post_id_divisions,
                in_feats,
                target,
                output,
                delete_negative=delete_negative,
            )
