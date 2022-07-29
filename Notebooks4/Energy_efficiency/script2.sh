for runs in 1 2
do
    i=6
    for B in 0.01 0
    do 
        for V in 1 0.1 0.01
        do 
            papermill ABC_GAN-TabNet.ipynb ./ABC_GAN_TabNet/ABC-GAN_output_${runs}_${i}.ipynb -p variance ${V} -p bias ${B}
            ((i=i+1))
        done 
    done 
done

# #GAN 
# for runs in 1 2 3 4 5
# do
#     papermill GAN.ipynb ./GAN_Output/GAN_output_${runs}_${i}.ipynb  
# done