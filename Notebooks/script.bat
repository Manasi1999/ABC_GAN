@echo off 

SET /A i = 0 

FOR /L %%R IN (0,1,10) 
    FOR  %%E IN (5000,10000) 
        FOR %%M IN (1,0)
            FOR %%V IN (1,0.1,0.01)
                papermill Dataset3-Boston.ipynb ../Boston_Output/Dataset3-Boston_output_%%i.ipynb -p mean %%M -p variance %%V -p n_epochs %%E -k papermill-tutorial
                jupyter nbconvert ../Boston_Output/Dataset3-Boston_output_%%i.ipynb --to pdf
                SET /A i = %i% + 1