import os.path

import numpy as np
from sklearn.metrics import classification_report

from imodels.rule_list.greedy_rule_list import GreedyRuleListClassifier
from imodels.rule_list.one_r import OneRClassifier
from imodels.rule_list.bayesian_rule_list.bayesian_rule_list import BayesianRuleListClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier
from typing import List, Tuple, Any
import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.lm.vocabulary import Vocabulary
from ast import literal_eval
from matplotlib import pyplot as plt
import networkx as nx
from tqdm import tqdm
from argparse import ArgumentParser

from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.dataset.explainable_dataset import ExplainableDataset
from tuw_nlp.graph.utils import graph_to_pn


def lemmatize(list_of_wordlists: List[List[str]]) -> List[List[str]]:
    stops = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(w) for w in l if w not in stops] for l in list_of_wordlists]


def get_vocab(list_of_wordlists: List[List[str]], cutoff: int) -> Tuple[Vocabulary, List[List[str]]]:
    lemma_list = lemmatize(list_of_wordlists)
    words = [w for l in lemma_list for w in l]
    vocab = Vocabulary(words, unk_cutoff=cutoff)
    return vocab, lemma_list


def bag_of_words_model(potato_path: str, target: str) -> Tuple[List[List[int]], List[str]]:
    df = ExplainableDataset(path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}).to_dataframe()
    vocab, lemma_list = get_vocab([row.text.split(" ") if row.label_id == 1 else [] for (_, row) in df.iterrows()], cutoff=5)
    feature_vectors = [[(word in set(l)) * 1 for word in vocab] for l in lemma_list]
    return feature_vectors, [word for word in vocab]


def bag_of_rationale_words_model(potato_path: str, target: str) -> Tuple[List[List[int]], List[str]]:
    df = ExplainableDataset(path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}).to_dataframe()
    vocab, lemma_list = get_vocab([literal_eval(r) for r in df.rationale_id], cutoff=5)
    feature_vectors = [[(word in l) * 1 for word in vocab] for l in lemma_list]
    return feature_vectors, [word for word in vocab]


def all_graph_model(potato_path: str, target: str) -> Tuple[List[List[int]], List[str]]:
    df = ExplainableDataset(path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}).to_dataframe()
    toxic_df = df[df.label == target.capitalize()]
    feat_list = [[relabel_node_numbers(nx.subgraph(g, e)) for e in g.edges] + [relabel_node_numbers(nx.subgraph(g, n)) for n in g.nodes] for g in toxic_df.graph]
    # flatten it
    feat_list = Vocabulary([j for l in feat_list for j in l], unk_cutoff=5)
    print(len(feat_list))
    all_features = [[[feature], [], target.capitalize()] for feature in feat_list if feature != "<UNK>"]
    evaluator = FeatureEvaluator()
    feat_columns = []
    feats = []
    match = evaluator.match_features(df, all_features, multi=True, allow_multi_graph=True)
    for feature in all_features:
        res = match["Matched rule"].apply(lambda x: feature in x if isinstance(x, list) else False)
        feats.append(str(feature))
        feat_columns.append(res * 1)
    feature_vectors = list(map(list, zip(*feat_columns)))
    return feature_vectors, feats


def graph_model(potato_path: str, target: str, features: str) -> Tuple[List[List[int]], List[str]]:
    df = ExplainableDataset(path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}).to_dataframe()
    with open(features, "r") as feat_file:
        features = json.load(feat_file)
    evaluator = FeatureEvaluator()
    feat_columns = []
    feats = []
    for feature in features[target.capitalize()]:
        match = evaluator.match_features(df, [feature])
        feats.append(str(feature))
        feat_columns.append([(r["Predicted label"] == target.capitalize()) * 1 for (_, r) in match.iterrows()])
    feature_vectors = list(map(list, zip(*feat_columns)))
    return feature_vectors, feats


def relabel_node_numbers(subgraph: nx.DiGraph) -> str:
    relabel_dict = {}
    for n in subgraph.nodes():
        relabel_dict[n] = len(relabel_dict)
    sg = nx.relabel_nodes(subgraph, relabel_dict)
    return graph_to_pn(sg)

def rationale_graph(df: pd.DataFrame):
    graphs = []
    rats = df[df.rationale == 1]
    rat_ids = [(literal_eval(r.rationale_lemma), r.graph) for (i, r) in rats.iterrows() if literal_eval(r.rationale_lemma) != []]
    for ids, graph in rat_ids:
        g = graph.subgraph([i + 1 for i in ids])
        connected_components = list(nx.weakly_connected_components(g))
        graphs += [relabel_node_numbers(g.subgraph(c)) for c in connected_components]
    return list(set(graphs))


def graph_rationale_model(potato_path: str, target: str) -> Tuple[List[List[int]], List[str]]:
    df = ExplainableDataset(path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}).to_dataframe()
    features = rationale_graph(df)
    evaluator = FeatureEvaluator()
    feat_columns = []
    feats = []
    all_features = [[[feature], [], target.capitalize()] for feature in features]
    match = evaluator.match_features(df, all_features, multi=True, allow_multi_graph=True)
    for feature in features:
        match_feature = [[feature], [], target.capitalize()]
        res = match["Matched rule"].apply(lambda x: match_feature in x if isinstance(x, list) else False)
        feats.append(str(match_feature))
        feat_columns.append(res * 1)
    feature_vectors = list(map(list, zip(*feat_columns)))
    return feature_vectors, feats


def get_labels(path: str) -> List[int]:
    df = pd.read_csv(path, sep='\t')
    return df.label_id.to_list()


def gready_model(feature_vectors: List[List[int]], labels: List[int], feature_names: List[str], max_depth=20):
    m = GreedyRuleListClassifier(max_depth=max_depth)
    m.fit(feature_vectors, labels, feature_names=feature_names)
    features = []
    for rule in m.rules_:
        if 'col' in rule:
            if '[' in rule['col']:
                features.append(literal_eval(rule['col']))
            else:
                features.append(rule['col'])
    return features


def oneR_model(feature_vectors: List[List[int]], labels: List[int], feature_names: List[str]):
    m = OneRClassifier()
    m.fit(feature_vectors, labels, feature_names=feature_names)
    print(m)


def bayesian(feature_vectors: List[List[int]], labels: List[int], feature_names: List[str]):
    m = BayesianRuleListClassifier()
    m.fit(feature_vectors, labels, feature_names=feature_names)
    print(m)


def decision_tree(feature_vectors: List[List[int]], labels: List[int], feature_names: List[str]):
    m = DecisionTreeClassifier(max_depth=5)
    m.fit(feature_vectors, labels)
    plot_tree(m)
    plt.show()


def create_features(majmin: str, pureall: str, type_name: str, base_path_train: str, gen_method: Any, target: str) -> None:
    train_file = os.path.join(base_path_train, f"{majmin}_train_{pureall}.tsv")
    with open(f"{type_name}_{majmin}_{pureall}.tsv", "r") as all_bow:
        names = all_bow.readline().strip().split('\t')
        feat = []
        for line in all_bow:
            f = line.strip().split('\t')
            f = [int(i) for i in f]
            feat.append(f)
    labels = get_labels(train_file)
    if not os.path.exists(f"{type_name}_features"):
        os.makedirs(f"{type_name}_features", exist_ok=True)
    for i in tqdm(range(1, len(names) + 1)):
        generated_features = gen_method(feat, labels, names, max_depth=i)
        with open(os.path.join(f"{type_name}_features", f"{type_name}_{majmin}_{pureall}_{i}.json"),
                  "w") as out_features:
            json.dump({target.capitalize(): generated_features}, out_features)


def calculate_graph_validation(majmin: str, pureall: str, type_name: str, base_path_train: str, target: str, length: int) -> None:
    val_file = os.path.join(base_path_train, f"{majmin}_val_{pureall}.tsv")
    df = ExplainableDataset(path=val_file, label_vocab={"None": 0, target.capitalize(): 1}).to_dataframe()
    evaluator = FeatureEvaluator()
    scores = {}
    features_matched = pd.DataFrame()
    for i in tqdm(range(1, length)):
        with open(os.path.join(f"{type_name}_features", f"{type_name}_{majmin}_{pureall}_{i}.json")) as features:
            feats = json.load(features)
            for feat in feats[target.capitalize()]:
                if str(feat) not in features_matched:
                    features_matched[str(feat)] = evaluator.match_features(df, [feat])["Predicted label"] == target.capitalize()
        scores[i] = classification_report(df.label_id,
                                          (features_matched[[str(f) for f in feats[target.capitalize()]]].sum(axis=1) > 0) * 1,
                                          output_dict=True)
    with open(f"{type_name}_{majmin}_{pureall}_results.json", "w") as out_features:
        json.dump(scores, out_features, indent=4)


def calculate_bow_validation(majmin: str, pureall: str, type_name: str, base_path_train: str, target: str, length: int) -> None:
    val_file = os.path.join(base_path_train, f"{majmin}_val_{pureall}.tsv")
    df = ExplainableDataset(path=val_file, label_vocab={"None": 0, target.capitalize(): 1}).to_dataframe()
    lemmatized = lemmatize([text.split(" ") for text in df.text])
    scores = {}
    for i in tqdm(range(1, length)):
        with open(os.path.join(f"{type_name}_features", f"{type_name}_{majmin}_{pureall}_{i}.json")) as features:
            feats = json.load(features)
            feat_words = feats[target.capitalize()]
            feature_vectors = [[(word in set(l)) * 1 for word in feat_words] for l in lemmatized]
            out = pd.DataFrame(data=feature_vectors, columns=feat_words)
        scores[i] = classification_report(df.label_id, (out.sum(axis=1) > 0) * 1, output_dict=True)
    with open(f"{type_name}_{majmin}_{pureall}_results.json", "w") as out_features:
        json.dump(scores, out_features, indent=4)


def best(voting, filtering, type_name):
    f1 = []
    p = []
    r = []
    with open(f"{type_name}_{voting}_{filtering}_results.json") as features:
        feats = json.load(features)
        for i, f in feats.items():
            f1.append(f["1"]["f1-score"])
            p.append(f["1"]["precision"])
            r.append(f["1"]["recall"])
    print(f"\n{type_name} {voting} {filtering}")
    print(f"F1 {np.argmax(f1)+1}: {max(f1)}\nP {np.argmax(p)+1}: {max(p)}\nR {np.argmax(r)+1}: {max(r)}\n"
          f"P {np.argmax(f1)+1}: {p[np.argmax(f1)]}\nR {np.argmax(f1)+1}: {r[np.argmax(f1)]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir", "-t", help="The directory of the training files")
    parser.add_argument("--target", "-tar", help="The target of the hate")
    args = parser.parse_args()

    methods = {"rationale_graph": graph_rationale_model, "feature_graph": graph_model, "all_graph": all_graph_model,
               "rationale_bow": bag_of_rationale_words_model, "all_bow": bag_of_words_model}

    for type_name, method in methods.items():
        for voting in ["majority", "minority"]:
            for filtering in ["all", "pure"]:
                if voting == "majority" and filtering == "pure":
                    continue
                print(type_name, voting, filtering)
                train_file = os.path.join(args.train_dir, f"{voting}_train_{filtering}.tsv")
                if type_name != "feature_graph":
                    feat, names = method(train_file, args.target.capitalize())
                else:
                    feat, names = method(train_file, args.target.capitalize(), f"women_{voting}_train_{filtering}_features.json")
                with open(f"{type_name}_{voting}_{filtering}.tsv", "w") as all_bow:
                    all_bow.write("\t".join(names))
                    all_bow.write("\n")
                    for f in feat:
                        all_bow.write("\t".join([str(feature) for feature in f]))
                        all_bow.write("\n")
    for type_name in methods:
        for voting in ["majority", "minority"]:
            for filtering in ["pure", "all"]:
                if voting == "majority" and filtering == "pure":
                    continue
                print(type_name, voting, filtering)
                create_features(voting, filtering, type_name, args.train_dir, gready_model, args.target.capitalize())
    for type_name in methods:
        for voting in ["majority", "minority"]:
            for filtering in ["all", "pure"]:
                if voting == "majority" and filtering == "pure":
                    continue
                df = pd.read_csv(f"{type_name}_{voting}_{filtering}.tsv", sep="\t")
                if "graph" in type_name:
                    calculate_graph_validation(voting, filtering, type_name, args.train_dir, args.target.capitalize(), len(df.columns))
                else:
                    calculate_bow_validation(voting, filtering, type_name, args.train_dir, args.target.capitalize(), len(df.columns))
                best(voting, filtering, type_name)
