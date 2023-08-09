import os
import json
from argparse import ArgumentParser

def modify_config_file(path:str, target_directory: str=None) -> None:
    with open(path) as config_file:
        config = json.load(config_file)
        config["num_classes"] = 2.0
        if target_directory is not None:
            for filename in os.listdir(target_directory):
                config["data_file"] = os.path.join(target_directory, filename)
                write_path = f"{path[:-5]}_2_class_{os.path.basename(target_directory)}_{filename[:-5]}.json"
                with open(write_path, "w") as config_file_write:
                    json.dump(config, config_file_write, indent=4)
        else:
            write_path = f"{path[:-5]}_2_class.json"
            with open(write_path, "w") as config_file_write:
                json.dump(config, config_file_write, indent=4)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--config_dir", "-c", help="Directory of the config files", default="./best_model_json")
    argparser.add_argument("--target_dir", "-t", help="Name of the target directory, if you wish to train on "
                                                       "relabeled data, that only considers text against the target offensive")
    args = argparser.parse_args()
    model_versions = ["bestModel_bert_base_uncased_Attn_train_FALSE.json",
                      "bestModel_bert_base_uncased_Attn_train_TRUE.json",
                      "bestModel_birnnatt.json",
                      "bestModel_birnn.json",
                      "bestModel_birnnscrat.json",
                      "bestModel_cnn_gru.json"]

    for model in model_versions:
        modify_config_file(os.path.join(args.config_dir, model), args.target_dir)
