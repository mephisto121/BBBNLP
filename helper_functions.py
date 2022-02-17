import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, precision_score, confusion_matrix

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def calculate_metrics(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return f'R²: {r2_score(y_true, y_pred):.3f}\n' \
           f'MAE: {mean_absolute_error(y_true, y_pred):.3f}\n' \
           f'RMSE: {rmse(y_true, y_pred):.3f}\n' \
           f'Precision: {precision_score(y_true, y_pred):.3f}'

def confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted molecule Category')
    ax.set_ylabel('Actual molecule Category ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Acid','Amphoteric', 'Base'])
    ax.yaxis.set_ticklabels(['Acid','Amphoteric', 'Base'])

    ## Display the visualization of the Confusion Matrix.
    return plt.show()

def plot_rsl(y_true, y_pred):
    sns.set(color_codes=True)
    plt.figure(figsize=(6, 6), dpi=100)
    plt.title(f'Model Performance')
    plt.xlim((-7, 30))
    plt.ylim((-7, 30))
    plt.xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
    plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
    p = sns.regplot(x=y_pred, y=y_true)
    p.plot([-4, 18], [-4, 18])
    p.text(min(y_true), max(y_true), calculate_metrics(y_true, y_pred))
    plt.ylabel('Actual pKₐ')
    plt.xlabel('Predicted pKₐ')
    plt.show()


def coeff_determination_tf(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def mcc_metric(y_true, y_pred):
    threshold = 0.5  
    predicted = tf.cast(tf.greater(y_pred, threshold), tf.float32)
    true_pos = tf.math.count_nonzero(predicted * y_true)
    true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
    false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
    false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
    x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) 
        * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
    return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)




def rmse_tf(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))

def img_standard(img):
    return(tf.image.per_image_standardization(img))

def standardize(smiles):
    # this code is from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
    mol = Chem.MolFromSmiles(smiles)
     
    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol) 
     
    # if many fragments, get the "parent" (the actual mol we are interested in) 
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
         
    # try to neutralize molecule
    uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
     
    # note that no attempt is made at reionization at this step
    # nor at ionization at some pH (rdkit has no pKa caculator)
    # the main aim to to represent all molecules from different sources
    # in a (single) standard way, for use in ML, catalogue, etc.
     
    te = rdMolStandardize.TautomerEnumerator() # idem
    taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
     
    return taut_uncharged_parent_clean_mol

def calculate_data(smiles):
    #calculating morgan fingerprint, MACCS and descriptors from ROMol format

    descriptors = []
    morgan_fp = []
    maccs = []
    
    desc_calc = MolecularDescriptorCalculator([x for x in [x[0] for x in Descriptors.descList]])
    for mol in smiles:
        descriptors.append(desc_calc.CalcDescriptors(mol))
        morgan_fp.append(AllChem.GetMorganFingerprintAsBitVect(mol, 4, 4096, useFeatures=True))
        maccs.append(AllChem.GetMACCSKeysFingerprint(mol))
    # in this part all the descriptors that get any NaN as outcome will remove from dataset
    descriptors = pd.DataFrame(descriptors)
    descriptors = descriptors.dropna(axis = 1)
    descriptors = np.array(descriptors)
    morgan_fp = np.array(morgan_fp)
    maccs = np.array(maccs)

    return descriptors, morgan_fp, maccs, np.concatenate([descriptors, morgan_fp, maccs], axis = 1), np.concatenate([descriptors, morgan_fp], axis = 1)

def standard_data(data):
    standard = StandardScaler()
    scaled_data = standard.fit_transform(data)
    return scaled_data

def calculate_data_single(mol):
    descriptors = []
    morgan_fp = []
    maccs = []
    
    desc_calc = MolecularDescriptorCalculator([x for x in [x[0] for x in Descriptors.descList]])
    descriptors.append(desc_calc.CalcDescriptors(mol))
    morgan_fp.append(AllChem.GetMorganFingerprintAsBitVect(mol, 4, 4096, useFeatures=True))
    maccs.append(AllChem.GetMACCSKeysFingerprint(mol))
    descriptors = pd.DataFrame(descriptors)
    descriptors = descriptors.dropna(axis = 1)
    descriptors = np.array(descriptors)
    morgan_fp = np.array(morgan_fp)
    maccs = np.array(maccs)
    return descriptors, morgan_fp, maccs, np.concatenate([descriptors, morgan_fp, maccs], axis = 1), np.concatenate([descriptors, morgan_fp], axis = 1)

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)