# PLTNUM
環境構築
conda create --name pltnum python=3.11.8
conda activate pltnum
conda install anaconda::requests
conda install conda-forge::biopython
conda install anaconda::pandas
conda install anaconda::numpy
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install anaconda::scikit-learn
conda install conda-forge::transformers

1. 
https://static-content.springer.com/esm/art%3A10.1038%2Fnature10098/MediaObjects/41586_2011_BFnature10098_MOESM304_ESM.xls
をダウンロードする

pdb中からaaの情報は取れるから、必要ない
<!-- 2. 
python scripts/get_aa_from_uniprot_accession.py \
    --file_path="data/41586_2011_BFnature10098_MOESM304_ESM.xls" \
    --sheet_name="Sheet1" \
    --uniprotids_column="Uniprot IDs" \
    --num_processes=30
でaccessionidからaaを取得する -->

3. 
mouseのpdbファイルダウンロード
https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000589_10090_MOUSE_v4.tar

4. 
foldseekを適用
https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/viewからfoldseekをダウンロードし、bin下に置く
chmod 777 bin/foldseek

### Run Foldseek
```
python scripts/use_foldseek_for_uniprot.py  \
    --file_path="data/41586_2011_BFnature10098_MOESM304_ESM.xls" \
    --sheet_name="Sheet1" \
    --pdb_dir="/home2/sagawa/protein-half-life-prediction/PHLprediction/UP000000589_10090_MOUSE_v4" \
    --uniprotids_column="Uniprot IDs" \
    --num_processes=4
```

### Train PLTNUM regression
```
python scripts/train.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="westlake-repl/SaProt_650M_AF2" \
    --architecture="SaProt" \
    --lr=2e-5 \
    --epochs=3 \
    --batch_size=4 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --mask_ratio=0.05 \
    --mask_prob=0.1 \
    --n_folds=10 \
    --output_dir="./regression/" \
    --task="regression" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa_foldseek"
```
### Train PLTNUM regression
```
python scripts/train.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="facebook/esm2_t33_650M_UR50D" \
    --architecture="ESM2" \
    --lr=2e-5 \
    --epochs=3 \
    --batch_size=4 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --mask_ratio=0.05 \
    --mask_prob=0.1 \
    --n_folds=10 \
    --output_dir="./regression_ESM2/" \
    --task="regression" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa"
```
### Train LSTM regression
```
python scripts/train.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="LSTM" \
    --architecture="LSTM" \
    --lr=2e-5 \
    --epochs=10 \
    --batch_size=32 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --n_folds=10 \
    --output_dir="./regression_LSTM/" \
    --task="regression" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa"
```


### Train PLTNUM classification
```
python scripts/train.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="westlake-repl/SaProt_650M_AF2" \
    --architecture="SaProt" \
    --lr=2e-5 \
    --epochs=3 \
    --batch_size=4 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --mask_ratio=0.05 \
    --mask_prob=0.2 \
    --n_folds=10 \
    --output_dir="./classification/" \
    --task="classification" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa_foldseek"
```
### Train PLTNUM classification
```
python scripts/train.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="facebook/esm2_t33_650M_UR50D" \
    --architecture="ESM2" \
    --lr=2e-5 \
    --epochs=3 \
    --batch_size=4 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --mask_ratio=0.05 \
    --mask_prob=0.2 \
    --n_folds=10 \
    --output_dir="./classification_ESM2/" \
    --task="classification" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa"
```
### Train LSTM classification
```
python scripts/train.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="LSTM" \
    --architecture="LSTM" \
    --lr=2e-4 \
    --epochs=30 \
    --batch_size=32 \
    --use_amp \
    --num_workers=4 \
    --max_length=128 \
    --used_sequence="left" \
    --mask_prob=0 \
    --random_delete_ratio=0 \
    --n_folds=10 \
    --output_dir="./classification_LSTM/" \
    --task="classification" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa"
```




### Prediction
```
python scripts/predict.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="westlake-repl/SaProt_650M_AF2" \
    --model_path="/home2/sagawa/protein-half-life-prediction/ver20_56/model_ver20_56_fold0.pth" \
    --architecture="SaProt" \
    --batch_size=64 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --output_dir="./Peptide_Level_Turnover_Measurements_Enable_the_Study_of_Proteoform_Dynamics_prediction_result/" \
    --task="classification" \
    --sequence_col="aa_foldseek"
```
```
python scripts/predict.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="facebook/esm2_t33_650M_UR50D" \
    --model_path="/home2/sagawa/PLTNUM/classification/model_fold0.pth" \
    --architecture="ESM2" \
    --batch_size=64 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --output_dir="data" \
    --task="classification" \
    --sequence_col="aa"
```
```
python scripts/predict.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="westlake-repl/SaProt_650M_AF2" \
    --model_path="/home2/sagawa/PLTNUM/regression/model_fold0.pth" \
    --architecture="SaProt" \
    --batch_size=64 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --output_dir="data" \
    --task="regression" \
    --sequence_col="aa_foldseek"
```
























### Run Foldseek
```
python scripts/use_foldseek_for_uniprot.py  \
    --file_path="data/41586_2011_BFnature10098_MOESM304_ESM.xls" \
    --sheet_name="Sheet1" \
    --pdb_dir="/home2/sagawa/protein-half-life-prediction/PHLprediction/UP000000589_10090_MOUSE_v4" \
    --uniprotids_column="Uniprot IDs" \
    --num_processes=4
```

### Train PLTNUM regression
```
python scripts/train.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="westlake-repl/SaProt_650M_AF2" \
    --architecture="SaProt" \
    --lr=2e-5 \
    --epochs=3 \
    --batch_size=4 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --mask_ratio=0.05 \
    --mask_prob=0.1 \
    --n_folds=10 \
    --output_dir="./regression/" \
    --task="regression" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa_foldseek"
```

### Train PLTNUM classification
```
python scripts/train.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="westlake-repl/SaProt_650M_AF2" \
    --architecture="SaProt" \
    --lr=2e-5 \
    --epochs=10 \
    --batch_size=4 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --mask_ratio=0.05 \
    --mask_prob=0.2 \
    --n_folds=10 \
    --output_dir="./classification/" \
    --task="classification" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa_foldseek"
```

### Train LSTM classification
```
CUDA_VISIBLE_DEVICES=2 python scripts/train_LSTM.py \
    --data_path="data/41586_2011_BFnature10098_MOESM304_ESM_foldseek.csv" \
    --model="LSTM" \
    --architecture="LSTM" \
    --lr=2e-5 \
    --epochs=10 \
    --batch_size=32 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --n_folds=10 \
    --output_dir="./classification/" \
    --task="classification" \
    --target_col="Protein half-life average [h]" \
    --sequence_col="aa"
```

### Prediction
```
python scripts/predict.py \
    --data_path="/home2/sagawa/protein-half-life-prediction/Peptide_Level_Turnover_Measurements_Enable_the_Study_of_Proteoform_Dynamics/train_foldseek.csv" \
    --model="westlake-repl/SaProt_650M_AF2" \
    --model_path="/home2/sagawa/protein-half-life-prediction/ver20_56/model_ver20_56_fold0.pth" \
    --architecture="SaProt" \
    --batch_size=64 \
    --use_amp \
    --num_workers=4 \
    --max_length=512 \
    --used_sequence="left" \
    --output_dir="./Peptide_Level_Turnover_Measurements_Enable_the_Study_of_Proteoform_Dynamics_prediction_result/" \
    --task="classification" \
    --sequence_col="aa_foldseek"
```