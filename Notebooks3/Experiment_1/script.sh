#ABC-GAN - Stats 
# for runs in 1 2 3 4 5 6 7 8 9 10
# do
#     papermill experiment.ipynb ./Output/output_${runs}.ipynb
#     ((i=i+1))
# done

for runs in 1 2 3 4 5 6 7 8 9 10
do
    papermill GAN.ipynb ./GAN_Output/output_${runs}.ipynb
    ((i=i+1))
done


