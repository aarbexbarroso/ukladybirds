import os
import pandas as pd

for RUN in range(1,6):
	MAXENT_DIR = f'/path/maxent_v3/Batch_{RUN}/species_predictions_val.csv'
	RESNET_DIR = f'/path/csv_10/resnet50_128/Batch{RUN}/val_predictions_batch_{RUN}.csv'
	# Read the two CSVs
	df_maxent = pd.read_csv(MAXENT_DIR)  # contains column 'id' (numbers)
	df_resnet = pd.read_csv(RESNET_DIR)  # contains column 'image_id' (e.g. '123.jpg')
	maxent_columns = [
	    'id','Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	df_maxent = df_maxent[maxent_columns]
	# Convert 'image_id' to number, removing '.jpg'
	df_resnet['id'] = df_resnet['image_id'].str.replace('.jpg', '', regex=False).astype(int)
	resnet_columns = [
	    'id', 'real_class', 'Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	df_resnet = df_resnet[resnet_columns]
	print(f'Maxent: {len(df_maxent)}')
	print(f'Resnet: {len(df_resnet)}')
	print(df_maxent.columns.tolist())
	print(df_resnet.columns.tolist())
	# Perform the left join based on the 'id' column
	df_merged = pd.merge(df_resnet, df_maxent, on='id', how='left')
	ordered_classes = [
	    'Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	class_map = {name: i for i, name in enumerate(ordered_classes)}
	df_merged['real_class'] = df_merged['real_class'].map(class_map)
	print(f'Merged: {len(df_merged)}')
	print(df_merged.columns.tolist())
	print(df_merged.head())
	# File path
	output_file_path = f'/path/predictions_for_ga/maxent_v3_resnet50_128/Batch_{RUN}/val_predictions.csv'
	# Extract directory from path
	output_directory = os.path.dirname(output_file_path)
	# Create directory if it does not exist
	os.makedirs(output_directory, exist_ok=True)
	# Save result
	df_merged.to_csv(output_file_path, index=False)









import os
import pandas as pd

for RUN in range(1,6):
	MAXENT_DIR = f'/path/maxent_v3/Batch_{RUN}/species_predictions_test.csv'
	RESNET_DIR = f'/path/csv_10/resnet50_128/Batch{RUN}/test_predictions_batch_{RUN}.csv'
	# Read the two CSVs
	df_maxent = pd.read_csv(MAXENT_DIR)  # contains column 'id' (numbers)
	df_resnet = pd.read_csv(RESNET_DIR)  # contains column 'image_id' (e.g. '123.jpg')
	maxent_columns = [
	    'id','Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripunctatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	df_maxent = df_maxent[maxent_columns]
	# Convert 'image_id' to number, removing '.jpg'
	df_resnet['id'] = df_resnet['image_id'].str.replace('.jpg', '', regex=False).astype(int)
	resnet_columns = [
	    'id', 'real_class', 'Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripunctatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	df_resnet = df_resnet[resnet_columns]
	print(f'Maxent: {len(df_maxent)}')
	print(f'Resnet: {len(df_resnet)}')
	print(df_maxent.columns.tolist())
	print(df_resnet.columns.tolist())
	# Perform the left join based on the 'id' column
	df_merged = pd.merge(df_resnet, df_maxent, on='id', how='left')
	ordered_classes = [
	    'Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripunctatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	class_map = {name: i for i, name in enumerate(ordered_classes)}
	df_merged['real_class'] = df_merged['real_class'].map(class_map)
	print(f'Merged: {len(df_merged)}')
	print(df_merged.columns.tolist())
	print(df_merged.head())
	# File path
	output_file_path = f'/path/predictions_for_ga/maxent_v3_resnet50_128/Batch_{RUN}/test_predictions.csv'
	# Extract directory from path
	output_directory = os.path.dirname(output_file_path)
	# Create directory if it does not exist
	os.makedirs(output_directory, exist_ok=True)
	# Save result
	df_merged.to_csv(output_file_path, index=False)