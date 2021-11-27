for runs in 7 8 9 10
do
    i=0
    for M in 1 0
    do 
        for V in 1 0.1 0.01 
        do 
            papermill ABC_GAN.ipynb ./Main/ABC-GAN_output_${i}_${runs}.ipynb -p abc_mean ${M} -p variance ${V} -p n_epochs 5000 -k papermill-tutorial
            ((i=i+1))
        done 
    done
done 