import tensorflow as tf
from tensorflow.keras.metrics import Precision, AUC, binary_accuracy, TrueNegatives, TruePositives, FalseNegatives, FalsePositives
from helper_functions import mcc_metric
from transformers import AutoTokenizer
from simpletransformers.classification import ClassificationModel
from rdkit import Chem
import argparse
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description='Run BiLSTM-SMILES-IUPAC and roberta and bert predictions from the command line')
    parser.add_argument('-sl', '--smiles_lstm', default=None, type = str, 
                        help='SMILES input for BBB predictions with BiLSTM model')
    parser.add_argument('-it', '--iupac_lstm', default = None,
                         help = 'IUPAC input for BBB prediction with BiLSTM model', type=str)
    parser.add_argument('-st', '--smiles_roberta', default = None, type = str, 
                        help = "SMILES input for predictions with roberta model" )
    parser.add_argument('-ib', '--iupac_bert', default = None, type = str, 
                    help = "IUPAC input for BBB prediction with bert model ")
    parser.add_argument('-n', '--name', default='test_mol', help='The name of the molecule')
    parser.add_argument('-sm', '--smiles_model', default='model/model_33.tf',help='Path to the model')
    parser.add_argument('-im', '--iupac_model', default='model/model_34iupac.tf', help='Path to the model')   

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if bool(args.smiles_lstm) == True:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(args.smiles_lstm), isomericSmiles=True)

        model = tf.keras.models.load_model(args.smiles_model, custom_objects ={'Precision': Precision, 'AUC':AUC, 'binary_accuracy':binary_accuracy,
                                                    'TrueNegatives':TrueNegatives, 'TruePositives':TruePositives, 'FalseNegatives':FalseNegatives,
                                                    'FalsePositives':FalsePositives, 'mcc_metric':mcc_metric} )
        tokenizer = AutoTokenizer.from_pretrained('Parsa/BBB_prediction_classification_SMILES')

        smiles = [smiles]
        smile_tokenized = tokenizer(list(smiles), truncation=True, padding=True, max_length = 244)
        smiles_tokenized_ready = smile_tokenized['input_ids']
        n = []
        for i in smiles_tokenized_ready:
            n = i
        c = np.zeros(244)
        n = np.array(n)
        c[:n.shape[0]] = n
        c = list(c)
        c = [c]
        print(np.round(model.predict(c)))

    elif bool(args.iupac_lstm)==True:

        model = tf.keras.models.load_model(args.iupac_model, custom_objects ={'Precision': Precision, 'AUC':AUC, 'binary_accuracy':binary_accuracy,
                                            'TrueNegatives':TrueNegatives, 'TruePositives':TruePositives, 'FalseNegatives':FalseNegatives,
                                            'FalsePositives':FalsePositives, 'mcc_metric':mcc_metric} )
        tokenizer = AutoTokenizer.from_pretrained('Parsa/BBB_prediction_classification_IUPAC')

        iupac_name = [args.iupac_lstm]
        iupac_tokenized = tokenizer(list(iupac_name), truncation = True, padding = True, max_length = 256)
        iupac_tokenized = iupac_tokenized['input_ids']
        n = []
        for i in iupac_tokenized:
            n = i
        c = np.zeros(256)
        n = np.array(n)
        c[:n.shape[0]] = n
        c = list(c)
        c = [c]
        print(np.round(model.predict(c)))

    elif bool(args.iupac_bert) == True:
        model = ClassificationModel('bert', 'Parsa/BBB_prediction_classification_IUPAC',use_cuda=False)
        pred, _ = model.predict([args.iupac_bert])
        print(pred)

    elif bool(args.smiles_roberta) == True:
        model = ClassificationModel('roberta', "Parsa/BBB_prediction_classification_SMILES",use_cuda=False)
        pred, _ = model.predict([args.smiles_roberta])
        print(pred)
    else:
        print('found no inputs')
