#!/usr/bin/env python3
import pandas as pd
import pickle

data_frames = []  
file_names = ["train_lr.pkl", "train_rf.pkl", "train_gb.pkl"]  

for file_name in file_names:
    with open(file_name, "rb") as f:
        model_data = pickle.load(f)  
        df = pd.DataFrame.from_dict(model_data, orient="index")  
        df.reset_index(inplace=True) 
        df.columns = ["Model Name", "Accuracy", "F1-Score", "ROC-AUC"] 
        data_frames.append(df)


final_df = pd.concat(data_frames, ignore_index=True)
print(final_df)
final_df.to_csv('Evaluation_summary.csv')

with open("summary.pkl", "wb") as f:
    pickle.dump(final_df, f)
print("saved to summary.pkl")
