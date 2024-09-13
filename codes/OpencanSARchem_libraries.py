import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None, 'display.max_rows',None)

import rdkit
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFMCS
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem import rdCIPLabeler

rdDepictor.SetPreferCoordGen(True)
IPythonConsole.drawOptions.minFontSize=20

from rdkit.Chem import DataStructs, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit.Chem import QED
from rdkit.Chem.Descriptors import qed

import multiprocessing as mp
from datetime import datetime
from datetime import date
import tqdm
import time

import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
from IPython import get_ipython
from IPython.display import display

import scipy
from scipy.stats import shapiro, skew, kurtosis, norm
from scipy.stats import f_oneway, ttest_rel, ttest_ind
