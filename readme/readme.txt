1. The format of input data for training purpose should be exactly similar to format of 'features.csv' ('key' column and features columns) and 'labels.csv'. ('key' column and labels column)
   File names should be same too.

2. Additional 'features.txt' file should be given to choose which features to take for training.
   Format should be similar to one provided (each feature name on each line)
   Current file takes all features by default.

3. There are folders for each model. Each folder contains .py training script.
   The output of training scripts are model.sav, featureImportance.png and ROC.png files which are saved in same folder.

4. prediction-script.py is the prediction script.
   The data to be input for prediction should be of format 'predict.csv' ('key' column and features columns) and with same file name.
   It also uses 'features.txt' file which defines features to use for predicting. Again format and file name should be similar (each feature name on each line).

5. You can additionally give input in prediction script to decide which model to use for prediction.
   To do that, enter additional parameter (model name) in predictions function in prediction-script.py (line 68)
   By default it will output predictions from all models.
   Predictions output are in form of probabilities.
   
   