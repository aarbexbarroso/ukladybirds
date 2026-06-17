import os
import pandas as pd

for RUN in range(1,6):
	MAXENT_DIR = f'/path/maxent_v3/Batch_{RUN}/predicoes_todas_as_especies_val.csv'
	RESNET_DIR = f'/path/csv_10/resnet50_128/Batch{RUN}/val_predictions_batch_{RUN}.csv'
	# Ler os dois CSVs
	df_maxent = pd.read_csv(MAXENT_DIR)  # contém coluna 'id' (números)
	df_resnet = pd.read_csv(RESNET_DIR)  # contém coluna 'image_id' (ex: '123.jpg')
	colunas_maxent = [
	    'id','Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	df_maxent = df_maxent[colunas_maxent]
	# Converter 'image_id' para número, removendo '.jpg'
	df_resnet['id'] = df_resnet['image_id'].str.replace('.jpg', '', regex=False).astype(int)
	colunas_resnet = [
	    'id', 'real_class', 'Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	df_resnet = df_resnet[colunas_resnet]
	print(f'Maxent: {len(df_maxent)}')
	print(f'Resnet: {len(df_resnet)}')
	print(df_maxent.columns.tolist())
	print(df_resnet.columns.tolist())
	# Fazer o left join com base na coluna 'id'
	df_merged = pd.merge(df_resnet, df_maxent, on='id', how='left')
	classes_ordenadas = [
	    'Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	mapa_classes = {nome: i for i, nome in enumerate(classes_ordenadas)}
	df_merged['real_class'] = df_merged['real_class'].map(mapa_classes)
	print(f'Merged: {len(df_merged)}')
	print(df_merged.columns.tolist())
	print(df_merged.head())
	# Caminho do arquivo
	caminho_arquivo = f'/path/predictions_for_ga/maxent_v3_resnet50_128/Batch_{RUN}/val_predictions.csv'
	# Extrai o diretório do caminho
	diretorio = os.path.dirname(caminho_arquivo)
	# Cria o diretório se não existir
	os.makedirs(diretorio, exist_ok=True)
	# Salvar o resultado
	df_merged.to_csv(caminho_arquivo, index=False)









import os
import pandas as pd

for RUN in range(1,6):
	MAXENT_DIR = f'/home/antonio/modelos_maxent_v3/Batch_{RUN}/predicoes_todas_as_especies.csv'
	RESNET_DIR = f'/home/antonio/csv_10/CNN_Mateus/Batch{RUN}/test_predictions_batch_{RUN}.csv'
	# Ler os dois CSVs
	df_maxent = pd.read_csv(MAXENT_DIR)  # contém coluna 'id' (números)
	df_resnet = pd.read_csv(RESNET_DIR)  # contém coluna 'image_id' (ex: '123.jpg')
	colunas_maxent = [
	    'id','Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	df_maxent = df_maxent[colunas_maxent]
	# Converter 'image_id' para número, removendo '.jpg'
	df_resnet['id'] = df_resnet['image_id'].str.replace('.jpg', '', regex=False).astype(int)
	colunas_resnet = [
	    'id', 'real_class', 'Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	df_resnet = df_resnet[colunas_resnet]
	print(f'Maxent: {len(df_maxent)}')
	print(f'Resnet: {len(df_resnet)}')
	print(df_maxent.columns.tolist())
	print(df_resnet.columns.tolist())
	# Fazer o left join com base na coluna 'id'
	df_merged = pd.merge(df_resnet, df_maxent, on='id', how='left')
	classes_ordenadas = [
	    'Adalia bipunctata', 'Adalia decempunctata', 'Anatis ocellata',
	    'Aphidecta obliterata', 'Calvia quattuordecimguttata', 'Chilocorus renipustulatus',
	    'Coccinella septempunctata', 'Coccinella undecimpunctata', 'Exochomus quadripustulatus',
	    'Halyzia sedecimguttata', 'Harmonia axyridis', 'Harmonia quadripunctata',
	    'Hippodamia variegata', 'Propylea quattuordecimpunctata',
	    'Psyllobora vigintiduopunctata', 'Scymnus interruptus',
	    'Subcoccinella vigintiquattuorpunctata', 'Tytthaspis sedecimpunctata']
	mapa_classes = {nome: i for i, nome in enumerate(classes_ordenadas)}
	df_merged['real_class'] = df_merged['real_class'].map(mapa_classes)
	print(f'Merged: {len(df_merged)}')
	print(df_merged.columns.tolist())
	print(df_merged.head())
	# Caminho do arquivo
	caminho_arquivo = f'/home/antonio/predictions_for_ga/maxent_v3_cnn_mateus/Batch_{RUN}/test_predictions.csv'
	# Extrai o diretório do caminho
	diretorio = os.path.dirname(caminho_arquivo)
	# Cria o diretório se não existir
	os.makedirs(diretorio, exist_ok=True)
	# Salvar o resultado
	df_merged.to_csv(caminho_arquivo, index=False)

