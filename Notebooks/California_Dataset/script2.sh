for runs in 1 2 3 4 5 6 7 8 9 10
do
    i=0
    for V in 1 0.1 0.01 
    do 
        papermill ABC_GAN-skip_catboost.ipynb ./ABC_GAN_Catboost_Output/ABC-GAN_skip_output_${i}_${runs}.ipynb -p variance ${V} 
        ((i=i+1))
    done 
done