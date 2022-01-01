# Run Baseline Models  

# for runs in 1 2 3 4 5 6 7 8 9 10
# do 
#     papermill BaselineModels.ipynb ./BaseLine_Model_Output/baselineModels_output_${runs}.ipynb 

# done 

#Run GAN Model 
# for runs in 1 2 3 4 5 6 7 8 9 10
# do 
#     papermill GAN.ipynb ./Main/GAN_output_${runs}.ipynb  -k papermill-tutorial

# done 

#Run ABC-GAN Model
# for runs in 6 7 8 9 10
# do
#      i=0
#      for M in 1 0
#      do 
#          for V in 1 0.1 0.01 
#          do 
#              papermill ABC_GAN.ipynb ./Main/ABC-GAN_output_${i}_${runs}.ipynb -p abc_mean ${M} -p variance ${V} -p n_epochs 5000 -k papermill-tutorial            
#              ((i=i+1))
#          done 
#      done
# done 

#Run ABC-GAN Model
# for runs in 1 2 3 4 5 
# do
#     i=0
#     for M in 1 0
#     do 
#         for V in 1 0.1 0.01 
#         do 
#             papermill ABC_GAN.ipynb ./Main/ABC-GAN_output_${i}_${runs}.ipynb -p abc_mean ${M} -p variance ${V} -p n_epochs 5000 -k papermill-tutorial
#             ((i=i+1))
#         done 
#     done
# done 


# #Analysis 
# papermill Analysis.ipynb Analysis_Out.ipynb -k papermill-tutorial
# jupyter nbconvert Analysis_Out.ipynb --to pdf

for runs in 6 7 8 9 10 11 12 13 14 15  
do
    i=0
    for V in 1 0.1 0.01 
    do 
        papermill ABC_GAN_Model-Catboost_Pre-gen.ipynb ./Test/ABC-GAN_output_${i}_${runs}.ipynb -p variance ${V} -k papermill-tutorial
        ((i=i+1))
    done 
done