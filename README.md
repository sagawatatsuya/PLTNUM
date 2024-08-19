# PLTNUM: Protein LifeTime Neural Model
PLTNUM is a protein language model designed to predict protein half-life from amino acid sequences. It is built upon the [SaProt](https://huggingface.co/westlake-repl/SaProt_650M_AF2) model and trained using data from [Schwanhäusser, B et al. Nature. 2011](https://www.nature.com/articles/nature10098) dataset, which is a protein half-life dataset obtained using mass spectrometry and SILAC.  
PLTNUM not only achieves high accuracy in predicting protein half-life but also leverages SHAP analysis to pinpoint specific amino acid residues that significantly influence the prediction. 

![model image](https://github.com/sagawatatsuya/PLTNUM/blob/main/model-image.png)

- [PLTNUM](#pltnum)  
  - [Installation](#installation)  
  - [Usage](#usage)  
  - [Train](#train)  
  - [Structure](#structure) 
  - [Authors](#authors)
  - [Citation](#citation)  


## Installation
You can create a new conda environment with the required dependencies using the following commands:
```bash
git clone https://github.com/sagawatatsuya/PLTNUM.git
cd PLTNUM
conda env create -f environment.yml
```
Or you can install the required dependencies using conda:
```bash
conda create --name pltnum python=3.11.8
conda activate pltnum
conda install anaconda::requests
conda install conda-forge::biopython
conda install anaconda::pandas
conda install anaconda::numpy
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install anaconda::scikit-learn
conda install conda-forge::transformers
conda install -c conda-forge shap
```

## Usage
### 1. Create a structure-aware amino acid sequence using Foldseek
We provides two different PLTNUM models based on different architectures: ESM2 and SaProt. If you want to use the SaProt-based PLTNUM, you have to create a structure aware aa sequence using Foldseek.  
The following command will create a structure aware aa sequence from the PDB files in the `pdb_files` directory and creates an output csv file that contais the information of pdb file path and the structure aware aa sequence.
```
python scripts/apply_foldseek_to_pdb.py  \
    --pdb_dir="pdb_files" \
    --num_processes=4 \
    --output_dir="./data"
```
### 2. Half-life prediction using PLTNUM
You can use the PLTNUM model to predict protein half-life from amino acid sequences. The following command will predict protein half-life using the ESM2-based PLTNUM model.
```
python scripts/predict_with_PreTrainedModel.py \
    --data_path="data/demo_input.csv" \
    --model_path="sagawa/PLTNUM-SaProt-NIH3T3" \
    --architecture="SaProt" \
    --batch_size=32 \
    --use_amp \
    --num_workers=4 \
    --output_dir="output" \
    --sequence_col="aa_foldseek"
```

### 3. 作成した配列を基にSHAP計算
```
python scripts/calculate_shap.py \
    --data_path="data/demo_input.csv" \
    --model="facebook/esm2_t33_650M_UR50D" \
    --model_path="sagawa/PLTNUM-SaProt-NIH3T3" \
    --architecture="SaProt" \
    --batch_size=512 \
    --output_dir="output" \
    --sequence_col="aa_foldseek" \
    --max_evals=2
```

## Train
```
python scripts/train.py \
    --data_path="data/demo_input.csv" \
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
    --output_dir="./classification_PLTNUM_mouse/" \
    --task="classification" \
    --target_col="T1/2 [h]" \
    --sequence_col="aa_foldseek"
```

## Structure
```
PLTNUM/  
├── bin/            # Foldseek's binary file  
├── data/           # Datasets  
├── pdb_files/      # PDB files  
├── scripts/        # Scripts for training and prediction, etc.
└── README.md       # This README file  
```

## Authors
Tatsuya Sagawa, Eisuke Kanao, and Yasushi Ishihama  

## Citation