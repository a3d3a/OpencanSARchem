import sys, os								
import os.path								
									
import pandas as pd							
import numpy as np							
import scipy as sp							
									
import rdkit								
from rdkit import RDLogger						
RDLogger.DisableLog('rdApp.*')						
from rdkit import rdBase						
									
from rdkit import Chem, DataStructs							
from rdkit.Chem import DataStructs, AllChem, Draw															
									
from rdkit.Chem.Draw import IPythonConsole				
from rdkit.Chem.Draw import rdDepictor						
rdDepictor.SetPreferCoordGen(True)					
IPythonConsole.drawOptions.minFontSize=20				

from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams									
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint			
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker		
from rdkit.Chem.MolStandardize import rdMolStandardize			
from rdkit.Chem.Scaffolds import MurckoScaffold				
from rdkit.Chem import rdMolDescriptors, Descriptors, rdFMCS, QED														
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Lipinski import NumHeteroatoms
					
import multiprocessing as mp						
import tqdm									
									
import time								
from datetime import date						
from datetime import datetime							

def disconnect_metals(mol,metal_bond_patt):
    
    """Copied from canSARchem_RDKit (Python Script:Break Metal Bond)
    Break covalent bonds between metals and organic atoms under certain conditions.

    :param mol: Input molecule.
    :return: Molecule with metal disconnected."""
    
    for smarts in metal_bond_patt:
        pairs = mol.GetSubstructMatches(smarts)
        rwmol = Chem.RWMol(mol)
        orders = list()
        for i, j in pairs:
            bond = mol.GetBondBetweenAtoms(i, j)
            bond_type = bond.GetBondTypeAsDouble()
            orders.append(int(bond_type))
            rwmol.RemoveBond(i, j)
        mol = rwmol.GetMol()
        for n, (i, j) in enumerate(pairs):
            charge = orders[n]
            atom_1 = mol.GetAtomWithIdx(i)
            atom_2 = mol.GetAtomWithIdx(j)
            print("Before: {}->{} {} {}->{} {}".format(atom_1.GetSymbol(), atom_1.GetFormalCharge(), atom_1.GetImplicitValence(),
										               atom_2.GetSymbol(), atom_2.GetFormalCharge(), atom_2.GetImplicitValence()))
			
            atom_1.SetFormalCharge(atom_1.GetFormalCharge() + charge)
            atom_2.SetFormalCharge(atom_2.GetFormalCharge() - charge)
            print("After: {}->{} {} {}->{} {}".format(atom_1.GetSymbol(), atom_1.GetFormalCharge(), atom_1.GetImplicitValence(),
										 atom_2.GetSymbol(), atom_2.GetFormalCharge(), atom_2.GetImplicitValence()))
	
    Chem.SanitizeMol(mol)
    return mol

uc=rdMolStandardize.Uncharger()

def mol_from_smiles(smiles):
    try:
        mol=Chem.MolFromSmiles(smiles, True)
        if mol:
           for atom in mol.GetAtoms():
               if atom.GetAtomicNum() == 0:  # Check for dummy atom
                  return np.nan
           return mol
        if not mol:
           mol = Chem.MolFromSmiles(smiles, sanitize=False)
           if not mol:
              mol=np.nan
              return mol 
           problems = Chem.DetectChemistryProblems(mol)
           if len(problems) > 1:
               mol=np.nan
               return mol
           if len(problems) == 1:
            if problems[0].GetType() == 'AtomValenceException':
                edit_mol = Chem.RWMol(mol)
                edit_mol.UpdatePropertyCache(strict=False)
                for at in edit_mol.GetAtoms():
                    if at.GetAtomicNum() == 7 and at.GetExplicitValence()==4 and at.GetFormalCharge()==0:
                       at.SetFormalCharge(1)
                modified_smiles = Chem.MolToSmiles(edit_mol)
                print(modified_smiles)
                mol=Chem.MolFromSmiles(modified_smiles, True)
                return mol
            elif problems[0].GetType() == 'KekulizeException':
                unkekulized_atoms = list(problems[0].GetAtomIndices())
                for idx in unkekulized_atoms:
                   atom = mol.GetAtomWithIdx(idx)
                   if atom.GetSymbol().lower() == 'n':
                      atidx=idx
                      break        
                if atidx:
                  edit_mol = Chem.RWMol(mol)
                  edit_mol.GetAtomWithIdx(atidx).SetNumExplicitHs(1)
                  modified_smiles = Chem.MolToSmiles(edit_mol)
                  mol=Chem.MolFromSmiles(modified_smiles, True)
                  print(modified_smiles)
                  return mol
                
           #elif problems[0].GetType() == 'AtomKekulizeException':
               #pass
    
    except Exception as e:
       print(f"An exception occured: {type(e).__name__}, {e}")
       mol=np.nan
       return mol
    

def FIuTS(mol):
    # rdMolStandardize.Cleanup() "equivalent to"  molvs.standardize.Standardizer().standardize()
    # https://molvs.readthedocs.io/en/latest/guide/standardize.html
    # RemoveHs, RDKit SanitizeMol,MetalDisconnector, Normalizer, Reionizer, RDKit AssignStereochemistry
    try:
        mol = rdMolStandardize.Cleanup(mol)
  
        AM=Chem.MolFromSmarts('[Li,Na,K,Rb,Cs,Fr:2]~[N,O,S,F:1]')
        AEM=Chem.MolFromSmarts('[Be,Mg,Ca,Sr,Ba,Ra:2]~[N,O,S,F:1]')
        metal_bond_patt = [AM,AEM]

        for smarts in metal_bond_patt:
            result = mol.HasSubstructMatch(smarts)
            if result == True:
               mol = disconnect_metals(mol, metal_bond_patt)
       
        remove_hs_params = Chem.RemoveHsParameters()
        remove_hs_params.removeAndTrackIsotopes = True
        mol_no_h = Chem.RemoveHs(mol, remove_hs_params)
        mol_no_c = uc.uncharge(mol_no_h)

              # adding all Hs (including isotopic ones), then removing only non isotopic Hs
        mol = Chem.RemoveHs(Chem.AddHs(mol_no_c))
 
        smiles=Chem.MolToSmiles(mol)
        mol=Chem.MolFromSmiles(smiles)
    except:
        mol=np.nan
    return mol

def uIuTS(mol):
    remover = SaltRemover(defnFilename=os.path.join(os.getcwd(), 'codes', 'OpencanSARchem_salt_strip_list.txt'))
    salts_nonStandardInchiKey=pd.read_csv(os.path.join(os.getcwd(),'codes','OpencanSARchem_salt_inchikey_list.csv'))
    salt_inchikeys=salts_nonStandardInchiKey['inchiKeys'].tolist()
    try:
        if len(Chem.GetMolFrags(mol)) == 1:
           NumFrags = len(Chem.GetMolFrags(mol))
           return mol

        N_atoms = mol.GetNumAtoms()
        N_hetero = NumHeteroatoms(mol)
        N_heavy = mol.GetNumHeavyAtoms()

        if (N_heavy - N_hetero >= 1) and (N_atoms >= 2):

           mol_removed=remover.StripMol(mol, dontRemoveEverything=True, sanitize=False) 
           
           remove_hs_params = Chem.RemoveHsParameters()
           remove_hs_params.removeAndTrackIsotopes = True
           mol_no_h = Chem.RemoveHs(mol_removed, remove_hs_params)
           mol_no_c = uc.uncharge(mol_no_h)

           mol = Chem.RemoveHs(Chem.AddHs(mol_no_c))

           if len(Chem.GetMolFrags(mol)) == 1:
              smiles=Chem.MolToSmiles(mol)
              mol=Chem.MolFromSmiles(smiles) 
              return mol
               
           frags = list(Chem.GetMolFrags(mol, asMols=True))
           organic_fragments = []
           for frag in frags:
                if (frag.GetNumHeavyAtoms() - NumHeteroatoms(frag) >= 1) and (frag.GetNumAtoms() >= 1):
                    organic_fragments.append(frag)
           if len(organic_fragments) == 1:
               smiles=Chem.MolToSmiles(organic_fragments[0])
               mol=Chem.MolFromSmiles(smiles)
               return mol


           non_salt_organic_fragments = []
           salt_matches = []

           for organic_frag in organic_fragments:
               frag_inchikey = Chem.inchi.MolToInchiKey(organic_frag)
               is_salt = False
               for salt_inchikey in salt_inchikeys:
                   if frag_inchikey == salt_inchikey:
                       salt_matches.append(organic_frag)
                       is_salt = True
                       break
               if not is_salt:
                  non_salt_organic_fragments.append(organic_frag)
           if not non_salt_organic_fragments:
               if salt_matches:
                  largest_match = max(salt_matches, key=lambda x: len(Chem.inchi.MolToInchiKey(x)))
                  non_salt_organic_fragments.append(largest_match)
           
           if len(non_salt_organic_fragments) == 1:
                smiles=Chem.MolToSmiles(non_salt_organic_fragments[0])
                mol=Chem.MolFromSmiles(smiles)
                return mol

           non_salt_organic_fragments.sort(reverse=True, key = lambda m: m.GetNumHeavyAtoms()) 
           smiles=Chem.MolToSmiles(non_salt_organic_fragments[0])
           mol=Chem.MolFromSmiles(smiles)
           return mol

    except Exception as e:
        
        print(f"An exception occured: {type(e).__name__}, {e}")
        mol=np.nan
    return mol

def uIuuS(mol):
    if mol.GetNumHeavyAtoms() < 100:
       try:
           remove_hs_params = Chem.RemoveHsParameters()
           remove_hs_params.removeAndTrackIsotopes = True
           mol_no_h = Chem.RemoveHs(mol, remove_hs_params)

           tautomer_params = rdMolStandardize.CleanupParameters()
           tautomer_params.tautomerRemoveIsotopicHs = False
           tautomer_params.tautomerReassignStereo = True
           tautomer_params.tautomerRemoveSp3Stereo = False
           tautomer_params.tautomerRemoveBondStereo = False

           mol=rdMolStandardize.CanonicalTautomer(mol_no_h, tautomer_params)
        
                               # adding all Hs (including isotopic ones), then removing only non isotopic Hs
           smiles=Chem.MolToSmiles(Chem.RemoveHs(Chem.AddHs(mol)))
           mol=Chem.MolFromSmiles(smiles)
       except:
           mol=np.nan
       return mol

    else:
       return mol

def uuuuu(mol):
   
    try:
        mol=rdMolStandardize.IsotopeParent(mol)
        Chem.RemoveStereochemistry(mol)

        smiles=Chem.MolToSmiles(mol)
        mol=Chem.MolFromSmiles(smiles)
    except:
        mol=np.nan
    return mol

def get_murcko(mol):
    try:
        mol = rdMolStandardize.IsotopeParent(mol)
        murcko = MurckoScaffold.GetScaffoldForMol(mol)
        if Chem.MolToSmiles(murcko) == '':  # in case of compounds without rings
            murcko = np.nan
    except Exception as e:
        print(f"Error processing molecule: {e}")
        murcko=np.nan 
    return murcko

def representations(mol):
    try:
        smiles=Chem.MolToSmiles(mol)
        inchi=Chem.inchi.MolToInchi(mol)
        if not inchi:
           inchi=Chem.inchi.MolToInchi(mol,treatWarningAsError=True)
        inchikey=Chem.inchi.MolToInchiKey(mol)

        nsinchi=Chem.inchi.MolToInchi(mol,options='-FixedH')
        if not nsinchi:
           nsinchi=Chem.inchi.MolToInchi(mol,options='-FixedH',treatWarningAsError=True)
        nsinchikey=Chem.inchi.MolToInchiKey(mol,options='-FixedH')
    except:
        smiles='FAILED'
        inchi='FAILED'
        inchikey='FAILED'
        nsinchi='FAILED'
        nsinchikey='FAILED'
    return smiles,inchi,inchikey,nsinchi,nsinchikey

def pains(mol):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    first_match = catalog.GetFirstMatch(mol)
    
    if first_match is not None:
        pains_description = first_match.GetDescription()
        pains_references='doi:10.1021/jm901137j'
        pains = [1, pains_description, pains_references]
        return pains
    pains = [0,np.nan,np.nan]
    return pains

rules = pd.read_csv(os.path.join(os.getcwd(), 'codes', 'OpencanSARchem_toxicophore_rules.txt'), sep="\t")
smarts=rules['SMARTS'].tolist()

def toxicophore(mol):
    for smarts_pattern in smarts:
        rule=Chem.MolFromSmarts(smarts_pattern)
        has_toxicophore = mol.HasSubstructMatch(rule)
        if has_toxicophore:
            return 1
    return 0

prop_list_names=['SLogP','LabuteASA','TPSA','AMW','ExactMW','NumLipinskiHBA','NumLipinskiHBD','NumRotatableBonds','NumHBD','NumHBA','NumAmideBonds','NumHeteroAtoms','NumHeavyAtoms','NumAtoms','NumStereocenters','NumUnspecifiedStereocenters','NumRings','NumAromaticRings','NumSaturatedRings','NumAliphaticRings','NumAromaticHeterocycles','NumSaturatedHeterocycles','NumAliphaticHeterocycles','NumAromaticCarbocycles','NumSaturatedCarbocycles','NumAliphaticCarbocycles','NumBonds','MolFormula','IsChiral','NumRotBonds','qed']
def mol_props(smiles): #passing the mol directly seems to fail on stereochemistry.  Not sure why.  Will pass smiles and then conver to mol in the function
    #SLogP,LabuteASA,TPSA,AMW,ExactMW,NumLipinskiHBA,NumLipinskiHBD,NumRotatableBonds,NumHBD,NumHBA,NumAmideBonds,NumHeteroAtoms,NumHeavyAtoms,NumAtoms,NumStereocenters,NumUnspecifiedStereocenters,NumRings,NumAromaticRings,NumSaturatedRings,NumAliphaticRings,NumAromaticHeterocycles,NumSaturatedHeterocycles,NumAliphaticHeterocycles,NumAromaticCarbocycles,NumSaturatedCarbocycles,NumAliphaticCarbocycles,NumBonds,MolFormula,IsChiral,NumRotBonds
    mol=Chem.MolFromSmiles(smiles)
    if not mol:
       print('mol is none')
    SLogP=Descriptors.MolLogP(mol)
    LabuteASA=rdMolDescriptors.CalcLabuteASA(mol)
    TPSA=Descriptors.TPSA(mol)
    AMW=Descriptors.MolWt(mol)
    ExactMW=Descriptors.ExactMolWt(mol)
    NumLipinskiHBA=rdMolDescriptors.CalcNumLipinskiHBA(mol)
    NumLipinskiHBD=rdMolDescriptors.CalcNumLipinskiHBD(mol)
    NumRotatableBonds=rdMolDescriptors.CalcNumRotatableBonds(mol)
    NumHBD=rdMolDescriptors.CalcNumHBD(mol)
    NumHBA=rdMolDescriptors.CalcNumHBA(mol)
    NumAmideBonds=rdMolDescriptors.CalcNumAmideBonds(mol)
    NumHeteroAtoms=rdMolDescriptors.CalcNumHeteroatoms(mol)
    NumHeavyAtoms=rdMolDescriptors.CalcNumHeavyAtoms(mol)
    NumAtoms=rdMolDescriptors.CalcNumAtoms(mol)
    NumStereocenters=rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    NumUnspecifiedStereocenters=rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    NumRings=rdMolDescriptors.CalcNumRings(mol)
    NumAromaticRings=rdMolDescriptors.CalcNumAromaticRings(mol)
    NumSaturatedRings=rdMolDescriptors.CalcNumSaturatedRings(mol)
    NumAliphaticRings=rdMolDescriptors.CalcNumAliphaticRings(mol)
    NumAromaticHeterocycles=rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    NumSaturatedHeterocycles=rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
    NumAliphaticHeterocycles=rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
    NumAromaticCarbocycles=rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
    NumSaturatedCarbocycles=rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
    NumAliphaticCarbocycles=rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
    NumBonds=mol.GetNumBonds()
    MolFormula=rdMolDescriptors.CalcMolFormula(mol)
    #IsChiral needs better logic than this - we can use the canonical smiles, invert @ symbols, recanonicalize, and compare
    if NumStereocenters >= 1:
        IsChiral=1
    else:
        IsChiral=0
    NumRotBonds=rdMolDescriptors.CalcNumRotatableBonds(mol)
    qed=QED.default(mol)
    properties=[SLogP,LabuteASA,TPSA,AMW,ExactMW,NumLipinskiHBA,NumLipinskiHBD,NumRotatableBonds,NumHBD,NumHBA,NumAmideBonds,NumHeteroAtoms,NumHeavyAtoms,NumAtoms,NumStereocenters,NumUnspecifiedStereocenters,NumRings,NumAromaticRings,NumSaturatedRings,NumAliphaticRings,NumAromaticHeterocycles,NumSaturatedHeterocycles,NumAliphaticHeterocycles,NumAromaticCarbocycles,NumSaturatedCarbocycles,NumAliphaticCarbocycles,NumBonds,MolFormula,IsChiral,NumRotBonds,qed]
    return properties
