# Meta_scaler
This repository is home to the code, data and results related to the paper "Meta-scaler: a meta-learning framework for the selection of scaling techniques" (https://doi.org/10.1109/TNNLS.2024.3366615). 
The files are arranged in the directory structure in a very intuitive way, but some explanation is required for the 'code' directory:
- The code is distributed in 10 IPython Notebooks (.ipynb) and one R notebook (.rmd).
- If one wants to reproduce the whole experiment, all the statistical analysis and reproduce all figures and tables in the paper, this is the order of execution:
1. 'code/ST_performances/01_data_preparation_for_ST_perf.ipynb'
2. 'code/ST_performances/02_experiment_ST_perf.ipynb'
3. 'code/Meta_features_extraction/03_Extract_PyMFE.ipynb'
4. 'code/Meta_features_extraction/04_Extract_ImbCoL.Rmd'
5. 'code/05_merge_metafeatures_and_ST_perfs.ipynb'
6. 'code/06_experiment_meta_scaler.ipynb'
7. 'code/07_analysis.ipynb'
8. 'code/Reproducing_Jain_et_al/Reproducing_Jain_for_SOTA_comp.ipynb'
9. 'code/Reproducing_Zagatti_et_al/01_ST_perf_to_include_Normalizer.ipynb'
10. 'code/Reproducing_Zagatti_et_al/02_Reproducing_Zagatti_SOTA_comp.ipynb'
11. 'code/07B_analysis_SOTA_comparison.ipynb'

Note that the file 'code/environment.yml' presents our best effort to record the necessary software environment requiriments to execute the project, but some mismatches may happen as this was executed in several different machines.
