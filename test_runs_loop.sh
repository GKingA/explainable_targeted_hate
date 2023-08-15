for FILE in best_model_json/*;
    do
        if [[ "$FILE" =~ "pure" || "$FILE" =~ "all" ]]; then
            python3 testing_with_lime.py $FILE 100 1 -kn --test;
            python3 testing_with_lime.py $FILE 100 1 --test
            if [[ "$FILE" =~ "bert" || "$FILE" =~ "att" || "$FILE" =~ "scrat" ]]; then
                python3 testing_with_rational.py $FILE 1 -kn --test;
                python3 testing_with_rational.py $FILE 1 --test
            fi
        fi
    done
