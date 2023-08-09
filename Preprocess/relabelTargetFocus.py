import json
import os
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from argparse import ArgumentParser, ArgumentError
from ast import literal_eval

import pandas as pd
from networkx.readwrite import json_graph
from tuw_nlp.text.preprocess.hatexplain import preprocess_hatexplain
from xpotato.dataset.explainable_dataset import ExplainableDataset
from xpotato.graph_extractor.extract import GraphExtractor
from xpotato.graph_extractor.graph import PotatoGraph
from xpotato.models.trainer import GraphTrainer
from xpotato.dataset.utils import save_dataframe


def read_json(
    file_path: str, split_file: str, keep_disagreement=False, graph_path: str = None
) -> List[Dict[str, List[Dict[str, List[str]]]]]:
    split_ids = json.load(open(split_file))
    data_by_target = []
    with open(file_path) as dataset:
        data = json.load(dataset)
        for post in data.values():
            sentence = " ".join(post["post_tokens"])
            sentence = preprocess_hatexplain(sentence)
            targets = {}
            labels = {}
            pure = False
            one_majority = False
            for annotation in post["annotators"]:
                if annotation["label"] not in labels:
                    labels[annotation["label"]] = 1
                else:
                    labels[annotation["label"]] += 1
                for target_i in annotation["target"]:
                    if target_i not in targets:
                        targets[target_i] = 1
                    else:
                        targets[target_i] += 1
            # Pure is if it targets only one (or none) groups
            if len(targets) == 1 or (len(targets) == 2 and "None" in targets):
                pure = True
            # One majority is if the majority vote would just be one target
            if len([l for l in targets.values() if l > 1]) == 1:
                one_majority = True

            # We don't care about the instances, where each annotator said something different label-wise
            if len(labels) != len(post["annotators"]) or keep_disagreement:
                majority_targets = [t for (t, c) in targets.items() if c >= 2]
                minority_targets = [t for (t, c) in targets.items() if c < 2]
                rationale = defaultdict(list)
                # Get the rationales in an organized manner
                if len(post["rationales"]) > 0:
                    not_none_annotators = [
                        a["target"]
                        for a in post["annotators"]
                        if a["label"] != "normal"
                    ]
                    for annotator, rationales in zip(
                        not_none_annotators, post["rationales"]
                    ):
                        major_intersection = list(
                            set(annotator).intersection(majority_targets)
                        )
                        minor_intersection = list(
                            set(annotator).intersection(minority_targets)
                        )
                        for mi in major_intersection + minor_intersection:
                            rationale[mi].append(rationales)
                    rationale = {
                        key: np.round(np.mean(value, axis=0), decimals=0).tolist()
                        for (key, value) in rationale.items()
                    }
                data_by_target.append(
                    {
                        "id": post["post_id"],
                        "tokens": post["post_tokens"],
                        "sentence": sentence,
                        "pure": pure,
                        "one_majority": one_majority,
                        "rationales": dict(rationale),
                        "majority_labels": majority_targets,
                        "minority_labels": minority_targets,
                        "annotators": post["annotators"],
                        "train": post["post_id"] in split_ids["train"],
                        "val": post["post_id"] in split_ids["val"],
                        "test": post["post_id"] in split_ids["test"],
                        "original_rationale_list": post["rationales"],
                    }
                )
    if graph_path is None or not os.path.exists(graph_path):
        extractor = GraphExtractor(lang="en")
        graphs = list(
            extractor.parse_iterable(
                [data_point["sentence"] for data_point in data_by_target], "ud"
            )
        )
        for graph, data_point in zip(graphs, data_by_target):
            data_point["graph"] = json_graph.adjacency_data(graph)
    else:
        dataframe = pd.read_csv(graph_path, sep="\t")
        for graph, data_point in zip(dataframe["graph"], data_by_target):
            data_point["graph"] = graph
    return data_by_target


def filter_dataframe(
    dataframe: pd.DataFrame, target: str, remove_by_purity: bool = False
) -> Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]]:
    purity_filters = {
        "pure": dataframe.pure,
        "one_majority": dataframe.one_majority,
        "all": True,
    }
    dataframe_fragments = {"pure": {}, "one_majority": {}, "all": {}}
    for purity, purity_filter in purity_filters.items():
        majority_group = dataframe[
            dataframe.majority_labels.apply(lambda x: target.capitalize() in x)
            & purity_filter
        ]
        majority_group["label"] = target.capitalize()
        if remove_by_purity and purity != "all":
            majority_others = dataframe[
                dataframe.majority_labels.apply(lambda x: target.capitalize() not in x)
            ]
        else:
            majority_others = dataframe[~dataframe.id.isin(majority_group.id)]
        majority_others["label"] = "None"
        minority_group = dataframe[
            (
                (dataframe.minority_labels.apply(lambda x: target.capitalize() in x))
                | (dataframe.majority_labels.apply(lambda x: target.capitalize() in x))
            )
            & purity_filter
        ]
        minority_group["label"] = target.capitalize()
        if remove_by_purity and purity != "all":
            minority_other = dataframe[
                (
                    (
                        dataframe.minority_labels.apply(
                            lambda x: target.capitalize() not in x
                        )
                    )
                    & (
                        dataframe.majority_labels.apply(
                            lambda x: target.capitalize() not in x
                        )
                    )
                )
            ]
        else:
            minority_other = dataframe[~dataframe.id.isin(minority_group.id)]
        minority_other["label"] = "None"
        dataframe_fragments[purity]["majority"] = (majority_group, majority_others)
        dataframe_fragments[purity]["minority"] = (minority_group, minority_other)
    return dataframe_fragments


def save_in_original_format(
    filtered_df: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]],
    save_path: str,
    target: str,
) -> None:
    target_path = os.path.join(save_path, target)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for purity, mm_dict in filtered_df.items():
        for maj_min, (group, other) in mm_dict.items():
            df = pd.concat([group, other]).sort_index()
            original_dict = {}
            for idx, line in df.iterrows():
                original_dict[line.id] = {
                    "post_id": line.id,
                    "annotators": [
                        {
                            "label": "normal"
                            if line.label == "None"
                            else "offensive",
                            "annotator_id": a["annotator_id"],
                            "target": a["target"],
                        }
                        for a in line.annotators
                    ],
                    "rationales": [] if len(line.rationales) == 0 else
                    [] if target.capitalize() not in line.rationales else
                    [line.rationales[target.capitalize()] for _ in range(3)],
                    "post_tokens": line.tokens,
                }
            print(purity, len(original_dict))
            with open(
                os.path.join(target_path, f"{maj_min}_{purity}.json"), "w"
            ) as json_file:
                json.dump(original_dict, json_file)


def get_sentences(
    group: pd.DataFrame, other: pd.DataFrame, target: str
) -> List[Tuple[str, str, List[str]]]:
    sentences = {
        index: (
            example.sentence,
            target.capitalize(),
            []
            if target.capitalize() not in literal_eval(example.rationales)
            else [
                tok
                for rat, tok in zip(
                    literal_eval(example.rationales)[target.capitalize()],
                    literal_eval(example.tokens),
                )
                if rat == 1
            ],
            []
            if target.capitalize() not in literal_eval(example.rationales)
            else [
                index
                for index, rat in enumerate(
                    literal_eval(example.rationales)[target.capitalize()]
                )
                if rat == 1
            ],
            []
            if target.capitalize() not in literal_eval(example.rationales)
            else [
                tok["name"]
                for rat, tok in zip(
                    literal_eval(example.rationales)[target.capitalize()],
                    # LT and GT appear only around user or censored as well as an emoji,
                    # but that will not influence this negatively
                    sorted(
                        [
                            node
                            for node in literal_eval(example.graph)["nodes"]
                            if node["name"] not in ["LT", "GT"]
                        ],
                        key=lambda x: x["id"],
                    )[1:],
                )
                if rat == 1
            ],
            PotatoGraph(graph=json_graph.adjacency_graph(literal_eval(example.graph))),
        )
        for (index, example) in group.iterrows()
    }
    sentences.update(
        {
            index: (
                example.sentence,
                "None",
                [],
                [],
                [],
                PotatoGraph(
                    graph=json_graph.adjacency_graph(literal_eval(example.graph))
                ),
            )
            for index, example in other.iterrows()
        }
    )
    return [s[1] for s in sorted(sentences.items())]


def convert_to_potato(
    group: pd.DataFrame, other: pd.DataFrame, target: str
) -> pd.DataFrame:
    sentences = get_sentences(group, other, target)
    potato_dataset = ExplainableDataset(
        sentences,
        label_vocab={"None": 0, f"{target.capitalize()}": 1},
        lang="en",
    )
    return potato_dataset.to_dataframe()


def process(
    data_path: str,
    target: str,
    create_features: bool = False,
    graph_path: str = None,
    remove_by_purity: bool = False,
) -> None:
    df = pd.read_csv(os.path.join(data_path, graph_path), sep="\t")

    save_path = os.path.join(data_path, target)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filtered_df = filter_dataframe(df, target, remove_by_purity)
    for purity, mm_dict in filtered_df.items():
        for maj_min, (group, other) in mm_dict.items():
            train_df = convert_to_potato(group[group.train], other[other.train], target)
            val_df = convert_to_potato(group[group.val], other[other.val], target)
            test_df = convert_to_potato(group[group.test], other[other.test], target)
            save_dataframe(
                train_df, os.path.join(save_path, f"{maj_min}_train_{purity}.tsv")
            )
            save_dataframe(
                val_df, os.path.join(save_path, f"{maj_min}_val_{purity}.tsv")
            )
            save_dataframe(
                test_df, os.path.join(save_path, f"{maj_min}_test_{purity}.tsv")
            )
            if create_features:
                trainer = GraphTrainer(train_df)
                features = trainer.prepare_and_train()

                with open(
                    os.path.join(
                        data_path, f"{target}_{maj_min}_train_{purity}_features.json"
                    ),
                    "w+",
                ) as f:
                    json.dump(features, f)


if __name__ == "__main__":
    target_groups = [
        "african",
        "arab",
        "asian",
        "caucasian",
        "christian",
        "disability",
        "economic",
        "hindu",
        "hispanic",
        "homosexual",
        "indian",
        "islam",
        "jewish",
        "men",
        "other",
        "refugee",
        "women",
    ]
    argparser = ArgumentParser()
    argparser.add_argument(
        "--data_path", "-d", help="Path to the json dataset.", required=True
    )
    argparser.add_argument("--split_path", "-s", help="Path of the official split.")
    argparser.add_argument(
        "--mode",
        "-m",
        help="Mode to start the program. Modes:"
        "\n\t- distinct: "
        "cut the dataset.json into distinct categorical json files"
        "\n\t- process: "
        "load the chosen category as the target and every other one as non-target"
        "\n\t- both: "
        "run the distinct and the process after eachother",
        default="both",
        choices=["distinct", "process", "both"],
    )
    argparser.add_argument(
        "--target",
        "-t",
        help="The target group to set as our category.",
        choices=target_groups,
    )
    argparser.add_argument(
        "--create_features",
        "-cf",
        help="Whether to create train features based on the UD graphs.",
        action="store_true",
    )
    argparser.add_argument(
        "--keep_disagreement",
        "-kd",
        help="Whether to keep the data instances, where the annotators all annotated with different labels.",
        action="store_true",
    )
    argparser.add_argument(
        "--remove_by_purity",
        "-rp",
        help="Whether to remove the instances by purity, shrinking the size of the dataset.",
        action="store_true",
    )
    argparser.add_argument(
        "--graph_path",
        "-gp",
        help="Previously parsed graphs in the same data format as the distinct mode produces",
    )
    args = argparser.parse_args()

    if args.mode != "distinct" and args.target is None:
        raise ArgumentError(
            "Target is not given! If you want to produce a POTATO dataset "
            "(by running this code in process or both mode), you should specify the target."
        )

    graph_file_path = args.graph_path if args.graph_path is not None else "dataset.tsv"
    if args.mode != "process":
        dataset = (
            args.data_path
            if os.path.isfile(args.data_path)
            else os.path.join(args.data_path, "dataset.json")
        )
        if args.split_path is None:
            args.split_path = args.data_path
        split = (
            args.split_path
            if os.path.isfile(args.split_path)
            else os.path.join(args.split_path, "post_id_divisions.json")
        )
        if not os.path.isfile(dataset):
            raise ArgumentError(
                args.data_path,
                "The specified data path is not a file and does not contain a dataset.json file. "
                "If your file has a different name, please specify.",
            )
        dir_path = os.path.dirname(dataset)
        dt_by_target = read_json(
            dataset,
            split,
            keep_disagreement=args.keep_disagreement,
            graph_path=args.graph_path,
        )
        dataf = pd.DataFrame.from_records(dt_by_target)
        dataf.to_csv(os.path.join(dir_path, graph_file_path), sep="\t", index=False)
        filtered = filter_dataframe(dataf, args.target, args.remove_by_purity)
        save_in_original_format(
            filtered, dir_path, args.target
        )

        if args.mode == "both":
            process(
                data_path=dir_path,
                target=args.target,
                create_features=args.create_features,
                graph_path=graph_file_path,
            )

    else:
        dir_path = (
            os.path.dirname(args.data_path)
            if os.path.isfile(args.data_path)
            else args.data_path
        )
        process(
            data_path=dir_path,
            target=args.target,
            create_features=args.create_features,
            graph_path=graph_file_path,
        )
