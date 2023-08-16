mkdir -p models_explain;
cd eraserbenchmark;

for FILE in ../explanations_dicts/*;
    do
        NEWFILE=$(echo $FILE | cut -d "/" -f 3)
        echo "../models_explain/$NEWFILE"
        if [[ "$FILE" =~ "majority" ]]; then
            voting="majority";
        else
            voting="minority";
        fi
        if [[ "$FILE" =~ "pure" ]]; then
            filtering="pure";
        else
            filtering="all";
        fi
        model="$voting"_"$filtering";
        if [[ "$FILE" =~ "bert" ]]; then
            python3 rationale_benchmark/metrics.py --split test --strict --data_dir ../Data/Evaluation/Model_Eval/bert_"$voting"_"$filtering" --results $FILE --score_file ../models_explain/$NEWFILE
        elif [[ "$FILE" =~ "rule" ]]; then
            python3 rationale_benchmark/metrics.py --split test --strict --data_dir ../Data/Evaluation/Model_Eval/rules/"$voting"_"$filtering" --results $FILE --score_file ../models_explain/$NEWFILE
        else
            python3 rationale_benchmark/metrics.py --split test --strict --data_dir ../Data/Evaluation/Model_Eval/"$voting"_"$filtering" --results $FILE --score_file ../models_explain/$NEWFILE
        fi
    done

