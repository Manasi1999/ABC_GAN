# Run Baseline Models  

# for runs in 1 2 3 4 5 6 7 8 9 10
# do 
#     papermill BaselineModels.ipynb ./BaseLine_Model_Output/baselineModels_output_${runs}.ipynb 

# done 

#Run GAN Model 
<<<<<<< HEAD
for runs in 1 2 3 4 5 6 7 8 9 10
do 
    papermill GAN.ipynb ./Main/GAN_output_${runs}.ipynb  -k papermill-tut

done 

#Run ABC-GAN Model
for runs in 2 3
do
     i=0
     for M in 1 0
     do 
         for V in 1 0.1 0.01 
         do 
             papermill ABC_GAN.ipynb ./Main/ABC-GAN_output_${i}_${runs}.ipynb -p abc_mean ${M} -p variance ${V} -p n_epochs 5000 -k papermill-tut             
             ((i=i+1))
         done 
     done
done 
=======
# for runs in 1 2 3 4 5 6 7 8 9 10
# do 
#    papermill GAN.ipynb ./Main/GAN_output_${runs}.ipynb 

# done 
>>>>>>> 4ba47d993068e5c035a1a5a01e64b2d0692cba81

#Run ABC-GAN Model
for runs in 1 2 3 4 5 6 7 8 9 10
do
    i=0
    for M in 1 0
    do 
        for V in 1 0.1 0.01 
        do 
            papermill ABC_GAN.ipynb ./Main/ABC-GAN_output_${i}_${runs}.ipynb -p abc_mean ${M} -p variance ${V} -p n_epochs 5000 
            ((i=i+1))
        done 
    done
done 


# #Analysis 
# papermill Analysis.ipynb Analysis_Out.ipynb -k papermill-tutorial
# jupyter nbconvert Analysis_Out.ipynb --to pdf

