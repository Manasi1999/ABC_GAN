i = 0
for S in 100 1000 1000
do
    for F in 10 20 50 
    do 
        for M in 1 0
        do 
            for V in 1 0.1 0.01 
            do 
                papermill Dataset1-Regression.ipynb RegressionOutput/Dataset1-Regression_output_${i}.ipynb -p n_samples ${S} -p n_features ${F} -p mean ${M} -p variance ${V} -k papermill-tutorial
                jupyter nbconvert RegressionOutput/Dataset1-Regression_output_${i}.ipynb --to pdf
                ((i=i+1))
            done 
        done 
    done

done