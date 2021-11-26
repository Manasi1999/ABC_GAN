# # Simple Regression Dataset 
# i=0
# for S in 10 100
# do
#     for F in 10
#     do 
#         for M in 1 0
#         do 
#             for V in 1 0.1 0.01 
#             do 
#                 papermill Dataset1-Regression.ipynb ../Regression/Dataset1-Regression_output_${i}.ipynb -p n_samples ${S} -p n_features ${F} -p mean ${M} -p variance ${V} -k papermill-tutorial
#                 jupyter nbconvert ../Regression/Dataset1-Regression_output_${i}.ipynb --to pdf
#                 ((i=i+1))
#             done 
#         done 
#     done
# done

#Diabetes Dataset 
# i=0
# for M in 1 0
# do 
#     for V in 1 0.1 0.01 
#     do 
#         papermill Dataset2-Diabetes.ipynb ../Diabetes_Output/Dataset2-Diabetes_output_${i}.ipynb -p mean ${M} -p variance ${V} -k papermill-tutorial
#         jupyter nbconvert ../Diabetes_Output/Dataset2-Diabetes_output_${i}.ipynb --to pdf
#         ((i=i+1))
#     done 
# done

#Boston Housing 
i=0
for runs in range(10):
do
    for E in 5000 10000 
    do 
        for M in 1 0
        do 
            for V in 1 0.1 0.01 
            do 
                papermill Dataset3-Boston.ipynb ../Boston_Output/Dataset3-Boston_output_${i}.ipynb -p mean ${M} -p variance ${V} -p n_epochs ${E} -k papermill-tutorial
                jupyter nbconvert ../Boston_Output/Dataset3-Boston_output_${i}.ipynb --to pdf
                ((i=i+1))
            done 
        done
    done
done 

#California Housing 
# i=0
# for M in 1 0
# do 
#     for V in 1 0.1 0.01 
#     do 
#         papermill Dataset4-California_Housing.ipynb ../Boston_Output/Dataset4-California_Housing_output_${i}.ipynb -p mean ${M} -p variance ${V} -k papermill-tutorial
#         jupyter nbconvert ../California_Output/Dataset4-California_Housing_output_${i}.ipynb --to pdf
#         ((i=i+1))
#     done 
# done