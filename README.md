# Recreating our results

The following readme is meant to give guidance to recreate 
the results in our paper. The HateXplain readme can be found under
HateXplain_README.md in this repository.

## Preparations

To run the deep learning systems, you will need to create a 
virtual environment using requirements.txt
For the rule based system and the preprocessing you need to install
the [POTATO](https://pypi.org/project/xpotato/) and the 
[imodels](https://pypi.org/project/imodels/) libraries in a different 
environment.


## Preprocessing

For this step, you don't need the virtual environment.

To create the relabeled dataset, run the relabelTargetFocus.py 
script found in the Preprocess folder.

```bash
python3 relabelTargetFocus.py [-h] --data_path DATA_PATH
                              [--split_path SPLIT_PATH]
                              [--mode {distinct,process,both}]
                              [--target {african,arab,asian,caucasian,christian,disability,economic,hindu,hispanic,homosexual,indian,islam,jewish,men,other,refugee,women}]
                              [--create_features] [--keep_disagreement]
                              [--remove_by_purity] [--graph_path GRAPH_PATH]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH, -d DATA_PATH
                        Path to the json dataset.
  --split_path SPLIT_PATH, -s SPLIT_PATH
                        Path of the official split.
  --mode {distinct,process,both}, -m {distinct,process,both}
                        Mode to start the program. Modes: - distinct: cut the
                        dataset.json into distinct categorical json files -
                        process: load the chosen category as the target and
                        every other one as non-target - both: run the distinct
                        and the process after eachother
  --target {african,arab,asian,caucasian,christian,disability,economic,hindu,hispanic,homosexual,indian,islam,jewish,men,other,refugee,women}, -t {african,arab,asian,caucasian,christian,disability,economic,hindu,hispanic,homosexual,indian,islam,jewish,men,other,refugee,women}
                        The target group to set as our category.
  --create_features, -cf
                        Whether to create train features based on the UD
                        graphs.
  --keep_disagreement, -kd
                        Whether to keep the data instances, where the
                        annotators all annotated with different labels.
  --remove_by_purity, -rp
                        Whether to remove the instances by purity, shrinking
                        the size of the dataset.
  --graph_path GRAPH_PATH, -gp GRAPH_PATH
                        Previously parsed graphs in the same data format as
                        the distinct mode produces
```

To create the updated config files, run the reconfig_config_files script.
This only changes the dataset used as base and sets the number of classes
to two.
```bash
python3 reconfig_config_files.py [-h] [--config_dir CONFIG_DIR]
                                 [--target_dir TARGET_DIR]

options:
  -h, --help            show this help message and exit
  --config_dir CONFIG_DIR, -c CONFIG_DIR
                        Directory of the config files
  --target_dir TARGET_DIR, -t TARGET_DIR
                        Name of the target directory, if you wish to train on
                        relabeled data, that only considers text against the
                        target offensive
``` 

## Training

To train the deep learning models you can run the 
manual_training_inference script wih each of the new config files. 

```bash
python3 manual_training_inference.py [-h]
                                    --path_to_json --use_from_file
                                    --attention_lambda

Train a deep-learning model with the given data

positional arguments:
  --path_to_json      The path to json containining the parameters
  --use_from_file     whether use the parameters present here or directly use
                      from file
  --attention_lambda  required to assign the contribution of the atention loss

optional arguments:
  -h, --help          show this help message and exit
```

The rule based systems can be created with the choose_features script.

```bash
python3 choose_features.py [-h] [--train_dir TRAIN_DIR] [--target TARGET]

options:
  -h, --help            show this help message and exit
  --train_dir TRAIN_DIR, -t TRAIN_DIR
                        The directory of the training files
  --target TARGET, -tar TARGET
                        The target of the hate
```

## Evaluation

To create the predictions for the deep learning system, 
run the test_runs.sh script. This will create the predictions over
the ground truth portion of the dataset.
To create predictions over the whole dataset run test_runs_keep.sh

```bash
./test_runs.sh
./test_runs_keep.sh
```

The rules' output can be generated with rules_testing.py

```bash
python3 rules_testing.py [-h] [--config CONFIG]

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        A particular config file or a path to the config
                        files. If it's a directory, all of the configs
```

Then we can generate the final scores with test_eraser 
and test_eraser_rule_no_keep

```bash
./test_eraser.sh
./test_eraser_rule_no_keep.sh
```
