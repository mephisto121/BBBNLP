# BBBNLP
nlp models for blood brain barrier permeability
one bidirectional LSTM model, one bert and one roberta model for bbb prediction
if you want to use the dataset [please cite the group that made it](https://www.nature.com/articles/s41597-021-01069-5).
the lstm model is uploaded in it's folder(models) and for using the both bert and roberta models you can use [huggingface website](https://huggingface.co/Parsa).
roberta model for smiles prediction was fine-tuned on [deepchem's 77mil model](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM).

All the models include the BiLSTM models are available by running the main.py file or running the colab notebook down below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jGYf3sq93yO4EbgVaEl3nlClrVatVaXS#scrollTo=AMEdQItmilAw)
