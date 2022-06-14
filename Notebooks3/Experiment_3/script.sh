#ABC-GAN - Stats 
for runs in 1 2 3 4 5 6 7 8 9 10
do
    papermill experiment_a.ipynb ./Output_A/output_${runs}.ipynb
    ((i=i+1))
done

for runs in 1 2 3 4 5 6 7 8 9 10
do
    papermill experiment_b.ipynb ./Output_B/output_${runs}.ipynb
    ((i=i+1))
done

for runs in 1 2 3 4 5 6 7 8 9 10
do
    papermill experiment_c.ipynb ./Output_C/output_${runs}.ipynb
    ((i=i+1))
done



