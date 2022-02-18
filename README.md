# BBBNLP
nlp models for blood brain barrier permeability
one bidirectional LSTM model, one bert and one roberta model for bbb prediction
if you want to use the dataset [please cite the group that made it](https://www.nature.com/articles/s41597-021-01069-5).
the lstm model is uploaded in it's folder and for using the both bert and roberta models you can use [huggingface website](https://huggingface.co/Parsa).
roberta model for smiles prediction was fine-tuned on [deepchem's 77mil model](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM).

all the sklearn models and lightgbm model also are ready to use in their folder(sklearn_lgb_models).
