for runs in 1 2
do
    i=0
    for B in 1 0.1 0.01 0
    do 
        for V in 1 0.1 0.01
        do 
            papermill ABC_GAN-Catboost.ipynb ./ABC_GAN_Catboost/ABC-GAN_output_${runs}_${i}.ipynb -p variance ${V} -p bias ${B}
            ((i=i+1))
        done 
    done 
done