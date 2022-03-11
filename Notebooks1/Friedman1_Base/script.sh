#Run Baseline Models  
for runs in 1 2 3 4 5 6 7 8 9 10
do 
    i=0
    for V in 1 0.1 0.01 
    do 
        papermill BaselineModels.ipynb ./BaseLine_Model_Output/BaselineModels_output_${runs}_${i}.ipynb -p variance ${V}
        ((i=i+1))
    done 
done 

#Analysis 
papermill Analysis.ipynb Friedman1_Base.ipynb 
jupyter nbconvert Friedman1_Base.ipynb --to pdf

