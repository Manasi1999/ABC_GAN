#ABC-GAN
# for runs in 10
# do
#     papermill experiment_A.ipynb ./Output_A/output_${runs}.ipynb
#     ((i=i+1))
# done

for runs in 9 
do
    papermill experiment_B.ipynb ./Output_B/output_${runs}.ipynb
    ((i=i+1))
done

for runs in 1 2 3 4 5 6 7 8 9 10
do
    papermill GAN.ipynb ./GAN_Output/output_${runs}.ipynb
    ((i=i+1))
done




