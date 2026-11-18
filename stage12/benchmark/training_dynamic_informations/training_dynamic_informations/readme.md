# A Closer Look at Deep Learning on Tabular Data - Supplemental Datasets

This directory contains supplemental datasets for the paper **"A Closer Look at Deep Learning on Tabular Data"**. The datasets include training dynamic information recorded during the training of various deep learning methods on benchmark tabular datasets.

## Folder Structure

### 1. `time`

This folder contains files named with the pattern `seed_time_costs`, which record the time costs for different seeds during the training process. Each file provides insights into the computational efficiency and time required for training the models on the benchmark datasets, recorded at each epoch.

### 2. `loss`

This folder contains files named with the pattern `seed_val_losses`, which record the validation losses for different seeds during the training process. Each file helps evaluate the performance of the models in terms of validation loss, offering a perspective on model accuracy and generalization, recorded at each epoch.

### 3. `val_res`

This folder contains files named with the pattern `seed_results`, which record the Accuracy/RMSE on the validation set for different seeds during the training process, recorded at each epoch. 

## Additional Information

These datasets are essential for understanding the training dynamics and performance characteristics of different deep learning methods applied to tabular data. The information can be used for in-depth analysis and comparison of various models, aiding in the development and refinement of deep learning techniques for tabular data.