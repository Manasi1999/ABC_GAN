i=0
for S in 10 100
do
    for F in 10
    do 
        for M in 1 0
        do 
            for V in 1 0.1 0.01 
            do 
                papermill Dataset2_Friedman1.ipynb Dataset2_Friedman1_output_${i}.ipynb -p n_samples ${S} -p n_features ${F} -p mean ${M} -p variance ${V} -k papermill-tutorial
                jupyter nbconvert Dataset2_Friedman1_output_${i}.ipynb --to pdf
                ((i=i+1))
            done 
        done 
    done
done