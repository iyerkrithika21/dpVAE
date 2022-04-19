import subprocess
import os
import glob
import pandas as pd

model_dir_list = glob.glob("outputs/*")

model_names = [m.split("/")[-1] for m in model_dir_list]
result_df = pd.DataFrame()
result_array = []
dataset_dir = model_dir_list[0] + "/dataset/"
for dirs in model_dir_list:

	generated_dir = dirs + "/generated/"
	

	command = ["python","-m", "pytorch_fid", dataset_dir,generated_dir]
	result = subprocess.run(command,stdout=subprocess.PIPE)
	result = str(result)
	result = result.split("FID:")[-1].split("\n')")[0]
	result_array.append(result)

result_df["model"] = model_names
result_df["result"] = result_array
result_df.to_csv("result_3.csv",index=False)
