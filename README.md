# PLTNUM: Protein LifeTime Neural Model
PLTNUM is a protein language model designed to predict protein half-life from amino acid sequences. It is built upon the [SaProt](https://huggingface.co/westlake-repl/SaProt_650M_AF2) model and is trained on the dataset from [Schwanhäusser, B et al. Nature. 2011](https://www.nature.com/articles/nature10098), which was derived using mass spectrometry and SILAC.  
PLTNUM achieves not only highly accurate protein half-life prediction but also identification of amino acid residues that significantly influence these predictions utilizing SHAP analysis.  
The [models](https://huggingface.co/sagawa?search_models=PLTNUM) and [demo](https://huggingface.co/spaces/sagawa/PLTNUM) are available on the Hugging Face.

![model image](https://github.com/sagawatatsuya/PLTNUM/blob/main/model-image.png)

- [PLTNUM](#pltnum)  
  - [Installation](#installation)  
  - [Usage](#usage)  
  - [Train](#train)  
  - [Structure](#structure) 
  - [Authors](#authors)
  - [Citation](#citation)  


## Installation
To set up PLTNUM, you can either clone the repository and create a conda environment or directly install the dependencies via conda:

**Clone and Setup Environment:**  
```bash
git clone https://github.com/sagawatatsuya/PLTNUM.git
cd PLTNUM
conda env create -f environment.yml
```
**Direct installation:**  
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
**1. Create Structure-Aware Sequences**  
To use the SaProt-based PLTNUM model, generate structure-aware amino acid sequences with Foldseek:
```
python scripts/apply_foldseek_to_pdb.py  \
    --pdb_dir="pdb_files" \
    --num_processes=4 \
    --output_dir="./data"
```
**2. Half-life prediction using PLTNUM**  
Predict protein half-life using the PLTNUM model:
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

**3. SHAP analysis**  
Interpret predictions with SHAP analysis and search for the sequence that significantly influences the prediction:
```
python scripts/calculate_shap.py \
    --data_path="data/demo_input.csv" \
    --model="facebook/esm2_t33_650M_UR50D" \
    --model_path="sagawa/PLTNUM-SaProt-NIH3T3" \
    --architecture="SaProt" \
    --batch_size=32 \
    --output_dir="output" \
    --sequence_col="aa_foldseek" \
    --max_evals=100
```

## Train  
You can train PLTNUM with your own dataset. 
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
    --output_dir="./model_output/" \
    --task="classification" \
    --target_col="T1/2 [h]" \
    --sequence_col="aa_foldseek"
```

## Structure  
```
PLTNUM/  
├── bin/              # Foldseek's binary file  
├── data/             # Datasets  
├── pdb_files/        # PDB files  
├── scripts/          # Scripts for training and prediction, etc.
├── environment.yml   # Conda environment file
└── README.md         # This README file  
```

## Author
Tatsuya Sagawa  

## Citation
```
@article {Sagawa2024.09.10.612367,
	author = {Sagawa, Tatsuya and Kanao, Eisuke and Ogata, Kosuke and Imami, Koshi and Ishihama, Yasushi},
	title = {Prediction of Protein Half-lives from Amino Acid Sequences by Protein Language Models},
	elocation-id = {2024.09.10.612367},
	year = {2024},
	doi = {10.1101/2024.09.10.612367},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {We developed a protein half-life prediction model, PLTNUM, based on a protein language model using an extensive dataset of protein sequences and protein half-lives from the NIH3T3 mouse embryo fibroblast cell line as a training set. PLTNUM achieved an accuracy of 71\% on validation data and showed robust performance with an ROC of 0.73 when applied to a human cell line dataset. By incorporating Shapley Additive Explanations (SHAP) into PLTNUM, we identified key factors contributing to shorter protein half-lives, such as cysteine-containing domains and intrinsically disordered regions. Using SHAP values, PLTNUM can also predict potential degron sequences that shorten protein half-lives. This model provides a platform for elucidating the sequence dependency of protein half-lives, while the uncertainty in predictions underscores the importance of biological context in influencing protein half-lives.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/09/14/2024.09.10.612367},
	eprint = {https://www.biorxiv.org/content/early/2024/09/14/2024.09.10.612367.full.pdf},
	journal = {bioRxiv}
}
```
