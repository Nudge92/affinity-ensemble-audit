#!/usr/bin/env python3
"""
validate_with_controls.py — 대조군으로 모델 검증
=====================================================
알려진 EGFR 리간드(양성 대조군)와 비바인더(음성 대조군)를 사용하여
PIGNet2, OnionNet2, DeepDTA 모델의 예측 정확도를 평가합니다.

사용법:
  cd ~/Project/PLI
  python scripts/validate_with_controls.py

필요 조건:
  - SMINA 설치: ~/bin/smina
  - conda 환경: pignet2, onionnet2, deepdta
  - 대조군 CSV: data/control_ligands_egfr.csv
"""
import subprocess
import os
import sys
import json
import tempfile
import csv

import pandas as pd
import numpy as np

# ─── 설정 ───
PROJECT_DIR = os.path.expanduser("~/Project/PLI")
PROTEIN_PDB = os.path.join(PROJECT_DIR, "data", "egfr_clean.pdb")
CONTROL_CSV = os.path.join(PROJECT_DIR, "data", "control_ligands_egfr.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "06_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDA_PYTHON = {
    "PIGNet2":   os.path.expanduser("~/miniforge3/envs/pignet2/bin/python"),
    "OnionNet2": os.path.expanduser("~/miniforge3/envs/onionnet2/bin/python"),
    "DeepDTA":   os.path.expanduser("~/miniforge3/envs/deepdta/bin/python"),
    "GraphDTA":  os.path.expanduser("~/miniforge3/envs/pignet2/bin/python"),
    "MGraphDTA": os.path.expanduser("~/miniforge3/envs/pignet2/bin/python"),
    "GIGN":      os.path.expanduser("~/miniforge3/envs/pignet2/bin/python"),
    "PSICHIC":   os.path.expanduser("~/miniforge3/envs/psichic_fp/bin/python"),
    "NMDN":      os.path.expanduser("~/miniforge3/envs/nmdn/bin/python"),
}

SMINA_BIN = os.path.expanduser("~/bin/smina")
POCKET_CENTER = (1.5, 194.2, 20.4)
POCKET_BOX_SIZE = 20.0

# EGFR 서열 (Davis 데이터셋에서 직접 추출, proteins.txt key="EGFR")
# 길이: 1210 aa (전체 수용체 서열, 막관통 도메인 포함)
TARGET_SEQUENCE = (
    "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEV"
    "VLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALA"
    "VLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDF"
    "QNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGC"
    "TGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYV"
    "VTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFK"
    "NCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAF"
    "ENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKL"
    "FGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCN"
    "LLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVM"
    "GENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVV"
    "ALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGS"
    "GAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGI"
    "CLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAA"
    "RNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSY"
    "GVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPK"
    "FRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQ"
    "QGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTED"
    "SIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLN"
    "TVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRV"
    "APQSSEFIGA"
)


def smina_dock(smi, mol_id, protein_pdb, tmpdir, python_path):
    """SMILES → SMINA docking → docked SDF 경로 반환"""
    input_sdf = os.path.join(tmpdir, f"{mol_id}_input.sdf")
    docked_sdf = os.path.join(tmpdir, f"{mol_id}_docked.sdf")

    convert_script = f"""
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.MolFromSmiles("{smi}")
if mol is None:
    sys.exit(1)
mol = Chem.AddHs(mol)
res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
if res == -1:
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)
mol.SetProp("_Name", "{mol_id}")
writer = Chem.SDWriter("{input_sdf}")
writer.write(mol)
writer.close()
"""
    r = subprocess.run([python_path, "-c", convert_script],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None

    cmd = [
        SMINA_BIN,
        "--receptor", protein_pdb,
        "--ligand", input_sdf,
        "--out", docked_sdf,
        "--center_x", str(POCKET_CENTER[0]),
        "--center_y", str(POCKET_CENTER[1]),
        "--center_z", str(POCKET_CENTER[2]),
        "--size_x", str(POCKET_BOX_SIZE),
        "--size_y", str(POCKET_BOX_SIZE),
        "--size_z", str(POCKET_BOX_SIZE),
        "--exhaustiveness", "8",
        "--num_modes", "1",
        "--quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(docked_sdf):
        return None
    return docked_sdf


def run_pignet2(df, tmpdir):
    """PIGNet2 예측"""
    print("\n[PIGNet2] 예측 시작...")
    python_path = CONDA_PYTHON["PIGNet2"]
    pignet2_dir = os.path.abspath(os.path.join(PROJECT_DIR, "external_models/PIGNet2"))

    results = {}
    docked_sdfs = {}

    # 개별 docking
    for _, row in df.iterrows():
        mol_id = row["ID"]
        smi = row["SMILES"]
        print(f"  Docking {mol_id} ({row.get('Drug_Name', '')})...", end=" ")
        docked = smina_dock(smi, mol_id, PROTEIN_PDB, tmpdir, python_path)
        if docked:
            docked_sdfs[mol_id] = docked
            print("OK")
        else:
            print("FAIL")

    if not docked_sdfs:
        return {}

    # 병합 SDF
    merged_sdf = os.path.join(tmpdir, "pignet2_merged.sdf")
    id_order = []
    with open(merged_sdf, "w") as out_f:
        for mol_id in sorted(docked_sdfs.keys()):
            with open(docked_sdfs[mol_id]) as in_f:
                content = in_f.read()
            if content.strip():
                out_f.write(content)
                if not content.endswith("\n"):
                    out_f.write("\n")
                id_order.append(mol_id)

    # PIGNet2 실행
    out_path = os.path.join(tmpdir, "pignet2_predict.txt")
    ckpt_path = os.path.join(pignet2_dir, "src", "ckpt", "pda_0.pt")
    cmd = [
        python_path,
        os.path.join(pignet2_dir, "src", "exe", "predict.py"),
        ckpt_path,
        "-p", PROTEIN_PDB,
        "-l", merged_sdf,
        "-o", out_path,
        "--no-prot-pdb",
        "--no-prot-sdf",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  PIGNet2 실행 실패: {r.stderr[:300]}")
        return {}

    # 파싱
    with open(out_path) as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 3 and line_num < len(id_order):
                dG = float(parts[2])
                pKd = dG / -1.364
                results[id_order[line_num]] = round(pKd, 3)

    print(f"  PIGNet2 완료: {len(results)}개")
    return results


def run_onionnet2(df, tmpdir):
    """OnionNet2 예측"""
    print("\n[OnionNet2] 예측 시작...")
    python_path = CONDA_PYTHON["OnionNet2"]
    onionnet_dir = os.path.abspath(os.path.join(PROJECT_DIR, "external_models/OnionNet-2"))
    predict_script = os.path.join(onionnet_dir, "scoring", "predict.py")
    model_path = os.path.join(onionnet_dir, "models", "62shell_saved-model.h5")
    scaler_path = os.path.join(onionnet_dir, "models", "train_scaler.scaler")

    results = {}

    for _, row in df.iterrows():
        mol_id = row["ID"]
        smi = row["SMILES"]
        print(f"  {mol_id} ({row.get('Drug_Name', '')})...", end=" ")

        # Docking
        docked_sdf = smina_dock(smi, mol_id, PROTEIN_PDB, tmpdir, python_path)
        if not docked_sdf:
            print("dock FAIL")
            continue

        # SDF → PDB
        lig_pdb = os.path.join(tmpdir, f"{mol_id}_on2.pdb")
        conv = f"""
from rdkit import Chem
mol = Chem.MolFromMolFile("{docked_sdf}", removeHs=False)
if mol: Chem.MolToPDBFile(mol, "{lig_pdb}")
"""
        subprocess.run([python_path, "-c", conv], capture_output=True, text=True)
        if not os.path.exists(lig_pdb):
            print("conv FAIL")
            continue

        # 예측
        out_csv = os.path.join(tmpdir, f"{mol_id}_on2_pred.csv")
        cmd = [
            python_path, predict_script,
            "-rec_fpath", PROTEIN_PDB,
            "-lig_fpath", lig_pdb,
            "-model", model_path,
            "-scaler", scaler_path,
            "-shells", "62",
            "-shape", "84,124,1",
            "-out_fpath", out_csv,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("pred FAIL")
            continue

        try:
            pred_df = pd.read_csv(out_csv)
            score = float(pred_df["pred_pKd"].iloc[0])
            results[mol_id] = round(score, 3)
            print(f"pKd={score:.3f}")
        except Exception:
            print("parse FAIL")

    print(f"  OnionNet2 완료: {len(results)}개")
    return results


def run_deepdta(df, tmpdir):
    """DeepDTA 예측"""
    print("\n[DeepDTA] 예측 시작...")
    python_path = CONDA_PYTHON["DeepDTA"]
    deepdta_dir = os.path.abspath(os.path.join(PROJECT_DIR, "external_models/DeepDTA"))

    input_json = os.path.join(tmpdir, "deepdta_input.json")
    output_json = os.path.join(tmpdir, "deepdta_output.json")
    pred_script_path = os.path.join(tmpdir, "predict_deepdta.py")

    input_data = {
        "protein_sequence": TARGET_SEQUENCE,
        "smiles_list": df["SMILES"].tolist(),
    }
    with open(input_json, "w") as f:
        json.dump(input_data, f)

    pred_script_content = r'''
import json, sys, os, pickle
import numpy as np

deepdta_dir = sys.argv[1]
input_json = sys.argv[2]
output_json = sys.argv[3]

sys.path.insert(0, os.path.join(deepdta_dir, "source"))

with open(input_json) as f:
    data = json.load(f)

protein_seq = data["protein_sequence"]
smiles_list = data["smiles_list"]

charset_dir = os.path.join(deepdta_dir, "data", "davis")
try:
    with open(os.path.join(charset_dir, "charsmiset.pkl"), "rb") as f:
        charsmiset = pickle.load(f)
    with open(os.path.join(charset_dir, "charseqset.pkl"), "rb") as f:
        charseqset = pickle.load(f)
except FileNotFoundError:
    SMILES_CHARS = " #%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVYZ[\\]abcdefghiklmnoprstuvyz"
    PROTEIN_CHARS = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    charsmiset = {c: i for i, c in enumerate(SMILES_CHARS)}
    charseqset = {c: i for i, c in enumerate(PROTEIN_CHARS)}

def encode_smiles(smi, charset, max_len=100):
    enc = [charset.get(c, 0) + 1 for c in smi[:max_len]]
    return enc + [0] * (max_len - len(enc))

def encode_protein(seq, charset, max_len=1000):
    enc = [charset.get(c, 0) + 1 for c in seq[:max_len]]
    return enc + [0] * (max_len - len(enc))

prot_enc = encode_protein(protein_seq, charseqset)
smi_encs = [encode_smiles(smi, charsmiset) for smi in smiles_list]

try:
    import tensorflow as tf
    def cindex_score(y_true, y_pred):
        import tensorflow.keras.backend as K
        g = y_true - K.expand_dims(y_true, -1)
        f = y_pred - K.expand_dims(y_pred, -1)
        g = K.cast(g > 0, K.floatx())
        f = K.cast(f > 0, K.floatx())
        return K.sum(g * f) / (K.sum(g) + K.epsilon())

    model_path = os.path.join(deepdta_dir, "source", "saved_models", "deepdta_fold0_p0_00.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={"cindex_score": cindex_score})

    X_drug = np.array(smi_encs)
    X_prot = np.tile(np.array(prot_enc), (len(smi_encs), 1))

    preds = model.predict([X_drug, X_prot]).flatten()
    results = [{"smiles": smi, "score": float(p)} for smi, p in zip(smiles_list, preds)]
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    results = []

with open(output_json, "w") as f:
    json.dump(results, f)
'''
    with open(pred_script_path, "w") as f:
        f.write(pred_script_content)

    r = subprocess.run(
        [python_path, pred_script_path, deepdta_dir, input_json, output_json],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  DeepDTA 실행 실패: {r.stderr[:300]}")
        return {}

    with open(output_json) as f:
        preds = json.load(f)

    results = {}
    for i, pred in enumerate(preds):
        mol_id = df.iloc[i]["ID"]
        results[mol_id] = round(pred["score"], 3)

    print(f"  DeepDTA 완료: {len(results)}개")
    return results


def run_graphdta(df, tmpdir):
    """GraphDTA (GINConvNet) 예측"""
    print("\n[GraphDTA] 예측 시작...")
    python_path = CONDA_PYTHON["GraphDTA"]
    graphdta_dir = os.path.abspath(os.path.join(PROJECT_DIR, "external_models/GraphDTA"))

    input_json = os.path.join(tmpdir, "graphdta_input.json")
    output_json = os.path.join(tmpdir, "graphdta_output.json")
    pred_script_path = os.path.join(tmpdir, "predict_graphdta.py")

    input_data = {
        "protein_sequence": TARGET_SEQUENCE,
        "smiles_list": df["SMILES"].tolist(),
    }
    with open(input_json, "w") as f:
        json.dump(input_data, f)

    pred_script_content = r'''
import json, sys, os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import networkx as nx
from rdkit import Chem

graphdta_dir = sys.argv[1]
input_json = sys.argv[2]
output_json = sys.argv[3]

sys.path.insert(0, graphdta_dir)
from models.ginconv import GINConvNet

with open(input_json) as f:
    data = json.load(f)
protein_seq = data["protein_sequence"]
smiles_list = data["smiles_list"]

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i+1) for i, v in enumerate(seq_voc)}

def seq_cat(prot, max_len=1000):
    x = np.zeros(max_len)
    for i, ch in enumerate(prot[:max_len]):
        x[i] = seq_dict.get(ch, 0)
    return x

def one_of_k_encoding(x, s):
    if x not in s: raise Exception(f"{x} not in {s}")
    return list(map(lambda a: x==a, s))
def one_of_k_encoding_unk(x, s):
    if x not in s: x = s[-1]
    return list(map(lambda a: x==a, s))
def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al',
             'I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn',
             'H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) +
        [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None
    features = []
    for atom in mol.GetAtoms():
        f = atom_features(atom); features.append(f / sum(f))
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    if not edges:
        return mol.GetNumAtoms(), torch.FloatTensor(features), torch.LongTensor([[0],[0]])
    g = nx.Graph(edges).to_directed()
    ei = torch.LongTensor([[e1,e2] for e1,e2 in g.edges]).transpose(1,0)
    return mol.GetNumAtoms(), torch.FloatTensor(features), ei

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GINConvNet().to(device)
model.load_state_dict(torch.load(os.path.join(graphdta_dir, "model_GINConvNet_davis.model"), map_location=device))
model.eval()
target_tensor = torch.LongTensor([seq_cat(protein_seq)])

results = []
for smi in smiles_list:
    try:
        gd = smile_to_graph(smi)
        if gd is None: results.append({"score": None}); continue
        c_size, feat, ei = gd
        d = Data(x=feat, edge_index=ei, y=torch.FloatTensor([0]), target=target_tensor)
        d.__setitem__('c_size', torch.LongTensor([c_size]))
        for batch in DataLoader([d], batch_size=1):
            batch = batch.to(device)
            with torch.no_grad(): pred = model(batch)
            results.append({"score": float(pred.cpu().numpy().flatten()[0])})
    except Exception as e:
        results.append({"score": None})

with open(output_json, "w") as f:
    json.dump(results, f)
'''
    with open(pred_script_path, "w") as f:
        f.write(pred_script_content)

    r = subprocess.run([python_path, pred_script_path, graphdta_dir, input_json, output_json],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  GraphDTA 실행 실패: {r.stderr[:300]}")
        return {}

    with open(output_json) as f:
        preds = json.load(f)

    results = {}
    for i, pred in enumerate(preds):
        if pred.get("score") is not None:
            results[df.iloc[i]["ID"]] = round(pred["score"], 3)

    print(f"  GraphDTA 완료: {len(results)}개")
    return results


def run_mgraphdta(df, tmpdir):
    """MGraphDTA 예측"""
    print("\n[MGraphDTA] 예측 시작...")
    python_path = CONDA_PYTHON["MGraphDTA"]
    mgraphdta_dir = os.path.abspath(os.path.join(PROJECT_DIR, "external_models/MGraphDTA/regression"))

    input_json = os.path.join(tmpdir, "mgraphdta_input.json")
    output_json = os.path.join(tmpdir, "mgraphdta_output.json")
    pred_script_path = os.path.join(tmpdir, "predict_mgraphdta.py")

    input_data = {
        "protein_sequence": TARGET_SEQUENCE,
        "smiles_list": df["SMILES"].tolist(),
    }
    with open(input_json, "w") as f:
        json.dump(input_data, f)

    pred_script_content = r'''
import json, sys, os, glob
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

mgraphdta_dir = sys.argv[1]
input_json = sys.argv[2]
output_json = sys.argv[3]

sys.path.insert(0, mgraphdta_dir)
from model import MGraphDTA

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

with open(input_json) as f:
    data = json.load(f)
protein_seq = data["protein_sequence"]
smiles_list = data["smiles_list"]

VOCAB_PROTEIN = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,
                 "M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,
                 "U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
target_len = 1200
target_enc = [VOCAB_PROTEIN.get(s,0) for s in protein_seq]
target_enc = (target_enc + [0]*target_len)[:target_len]
target_tensor = torch.LongTensor([target_enc])

def mol2graph(mol):
    if mol is None: return None
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    g = nx.DiGraph()
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        g.add_node(i, a_type=a.GetSymbol(), a_num=a.GetAtomicNum(),
            acceptor=0, donor=0, aromatic=a.GetIsAromatic(),
            hybridization=a.GetHybridization(), num_h=a.GetTotalNumHs(),
            ExplicitValence=a.GetExplicitValence(), FormalCharge=a.GetFormalCharge(),
            ImplicitValence=a.GetImplicitValence(), NumExplicitHs=a.GetNumExplicitHs(),
            NumRadicalElectrons=a.GetNumRadicalElectrons())
    for f in feats:
        if f.GetFamily()=='Donor':
            for n in f.GetAtomIds(): g.nodes[n]['donor']=1
        elif f.GetFamily()=='Acceptor':
            for n in f.GetAtomIds(): g.nodes[n]['acceptor']=1
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e = mol.GetBondBetweenAtoms(i,j)
            if e: g.add_edge(i,j,b_type=e.GetBondType(),IsConjugated=int(e.GetIsConjugated()))
    feat=[]
    for n,d in g.nodes(data=True):
        h=[]
        h+=[int(d['a_type']==x) for x in ['H','C','N','O','F','Cl','S','Br','I']]
        h+=[d['a_num'],d['acceptor'],d['donor'],int(d['aromatic'])]
        h+=[int(d['hybridization']==x) for x in (Chem.rdchem.HybridizationType.SP,
             Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3)]
        h+=[d['num_h'],d['ExplicitValence'],d['FormalCharge'],d['ImplicitValence'],
            d['NumExplicitHs'],d['NumRadicalElectrons']]
        feat.append((n,h))
    feat.sort(key=lambda x:x[0])
    na = torch.FloatTensor([x[1] for x in feat])
    if na.max()!=na.min(): na=(na-na.min())/(na.max()-na.min())
    edges={}
    for n1,n2,d in g.edges(data=True):
        et=[int(d['b_type']==x) for x in (Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC)]
        et+=[int(not d['IsConjugated']),int(d['IsConjugated'])]
        edges[(n1,n2)]=et
    if not edges:
        return na, torch.LongTensor([[0],[0]]), torch.FloatTensor([[0,0,0,0,0,0]])
    ei=torch.LongTensor(list(edges.keys())).transpose(0,1)
    ea=torch.FloatTensor(list(edges.values()))
    return na,ei,ea

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MGraphDTA(3, 26, embedding_size=128, filter_num=32, out_dim=1).to(device)
mc = glob.glob(os.path.join(mgraphdta_dir,"save","**","model","*.pt"),recursive=True)
if not mc:
    mc = glob.glob(os.path.join(os.path.dirname(mgraphdta_dir),"visualization","pretrained_model","*.pt"))
if not mc: print("ERROR: No model found",file=sys.stderr); sys.exit(1)
mp = sorted(mc)[-1]
print(f"Loading: {mp}",file=sys.stderr)
model.load_state_dict(torch.load(mp,map_location=device))
model.eval()

results=[]
for smi in smiles_list:
    try:
        mol=Chem.MolFromSmiles(smi)
        gd=mol2graph(mol)
        if gd is None: results.append({"score":None}); continue
        na,ei,ea=gd
        d=Data(x=na,edge_index=ei,edge_attr=ea,y=torch.FloatTensor([0]),target=target_tensor)
        for batch in DataLoader([d],batch_size=1):
            batch=batch.to(device)
            with torch.no_grad(): pred=model(batch)
            results.append({"score":float(pred.cpu().numpy().flatten()[0])})
    except: results.append({"score":None})

with open(output_json,"w") as f: json.dump(results,f)
'''
    with open(pred_script_path, "w") as f:
        f.write(pred_script_content)

    r = subprocess.run([python_path, pred_script_path, mgraphdta_dir, input_json, output_json],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  MGraphDTA 실행 실패: {r.stderr[:300]}")
        return {}

    with open(output_json) as f:
        preds = json.load(f)

    results = {}
    for i, pred in enumerate(preds):
        if pred.get("score") is not None:
            results[df.iloc[i]["ID"]] = round(pred["score"], 3)

    print(f"  MGraphDTA 완료: {len(results)}개")
    return results


def run_gign(df, tmpdir):
    """GIGN 예측 (SMINA docking + pocket 추출 + 3D interaction graph)"""
    print("\n[GIGN] 예측 시작...")
    python_path = CONDA_PYTHON["GIGN"]
    gign_dir = os.path.abspath(os.path.join(PROJECT_DIR, "external_models/GIGN/GIGN"))

    input_json = os.path.join(tmpdir, "gign_input.json")
    output_json = os.path.join(tmpdir, "gign_output.json")
    pred_script_path = os.path.join(tmpdir, "predict_gign.py")

    # docking 먼저
    docked_sdfs = {}
    for _, row in df.iterrows():
        mol_id = row["ID"]
        smi = row["SMILES"]
        print(f"  Docking {mol_id} ({row.get('Drug_Name','')})...", end=" ")
        docked = smina_dock(smi, mol_id, PROTEIN_PDB, tmpdir, python_path)
        if docked:
            docked_sdfs[mol_id] = docked
            print("OK")
        else:
            print("FAIL")

    if not docked_sdfs:
        print("  GIGN: 모든 docking 실패!")
        return {}

    docked_info = []
    for _, row in df.iterrows():
        mol_id = row["ID"]
        if mol_id in docked_sdfs:
            docked_info.append({
                "mol_id": mol_id,
                "smiles": row["SMILES"],
                "docked_sdf": docked_sdfs[mol_id],
            })

    input_data = {
        "protein_pdb": PROTEIN_PDB,
        "pocket_distance": 5,
        "docked_info": docked_info,
        "gign_dir": gign_dir,
    }
    with open(input_json, "w") as f:
        json.dump(input_data, f)

    pred_script_content = r'''
import json, sys, os, subprocess, glob
import numpy as np
import torch
import networkx as nx
from scipy.spatial import distance_matrix
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from torch_geometric.data import Data, Batch

input_json = sys.argv[1]
output_json = sys.argv[2]

with open(input_json) as f:
    data = json.load(f)

protein_pdb = data["protein_pdb"]
pocket_distance = data["pocket_distance"]
docked_info = data["docked_info"]
gign_dir = data["gign_dir"]

sys.path.insert(0, gign_dir)
from GIGN import GIGN

def one_of_k_encoding_unk(x, s):
    if x not in s: x = s[-1]
    return list(map(lambda a: x==a, s))

def atom_feats(mol, graph):
    syms = ['C','N','O','S','F','P','Cl','Br','I']
    for atom in mol.GetAtoms():
        r = one_of_k_encoding_unk(atom.GetSymbol(), syms+['Unknown']) + \
            one_of_k_encoding_unk(atom.GetDegree(),[0,1,2,3,4,5,6]) + \
            one_of_k_encoding_unk(atom.GetImplicitValence(),[0,1,2,3,4,5,6]) + \
            one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2]) + \
            [atom.GetIsAromatic()] + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0,1,2,3,4])
        graph.add_node(atom.GetIdx(), feats=torch.FloatTensor(r))

def mol2graph(mol):
    g = nx.Graph(); atom_feats(mol, g)
    for b in mol.GetBonds(): g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    g = g.to_directed()
    x = torch.stack([f['feats'] for _,f in g.nodes(data=True)])
    ei = torch.stack([torch.LongTensor((u,v)) for u,v in g.edges()]).T if g.number_of_edges()>0 else torch.zeros((2,0),dtype=torch.long)
    return x, ei

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GIGN(35, 256).to(device)
mc = glob.glob(os.path.join(gign_dir,"model","**","model","*.pt"),recursive=True)
best = [m for m in mc if "repeat1" in m]
mp = best[0] if best else sorted(mc)[-1]
model.load_state_dict(torch.load(mp, map_location=device, weights_only=False))
model.eval()

tmpdir = os.path.dirname(input_json)
results = []
for info in docked_info:
    mid = info["mol_id"]
    sdf = info["docked_sdf"]
    try:
        lig_pdb = os.path.join(tmpdir, f"gign_lig_{mid}.pdb")
        subprocess.run(["obabel", sdf, "-O", lig_pdb, "-d"], capture_output=True, timeout=60)
        if not os.path.exists(lig_pdb):
            results.append({"mol_id":mid,"score":None}); continue
        pocket_pdb = os.path.join(tmpdir, f"gign_pocket_{mid}.pdb")
        pyscript = os.path.join(tmpdir, f"gign_pymol_{mid}.py")
        with open(pyscript,'w') as f:
            f.write(f"import pymol\nfrom pymol import cmd\ncmd.load('{protein_pdb}','protein')\ncmd.remove('resn HOH')\ncmd.load('{lig_pdb}','ligand')\ncmd.remove('hydrogens')\ncmd.select('pocket','byres ligand around {pocket_distance}')\ncmd.save('{pocket_pdb}','pocket')\ncmd.delete('all')\n")
        subprocess.run([sys.executable, pyscript], capture_output=True, timeout=120)
        if not os.path.exists(pocket_pdb) or os.path.getsize(pocket_pdb)==0:
            results.append({"mol_id":mid,"score":None}); continue
        ligand = Chem.MolFromPDBFile(lig_pdb, removeHs=True)
        pocket = Chem.MolFromPDBFile(pocket_pdb, removeHs=True)
        if not ligand or not pocket or ligand.GetNumConformers()==0 or pocket.GetNumConformers()==0:
            results.append({"mol_id":mid,"score":None}); continue
        anl = ligand.GetNumAtoms()
        pl = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
        pp = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
        xl, eil = mol2graph(ligand); xp, eip = mol2graph(pocket)
        x = torch.cat([xl,xp]); ei_intra = torch.cat([eil, eip+anl], dim=-1)
        dm = distance_matrix(pl.numpy(), pp.numpy())
        idx = np.where(dm < 5.)
        g = nx.Graph()
        for i,j in zip(idx[0],idx[1]): g.add_edge(i,j+anl)
        g = g.to_directed()
        ei_inter = torch.stack([torch.LongTensor((u,v)) for u,v in g.edges()]).T if g.number_of_edges()>0 else torch.zeros((2,0),dtype=torch.long)
        if ei_inter.shape[1]==0:
            results.append({"mol_id":mid,"score":None}); continue
        pos = torch.cat([pl,pp])
        d = Data(x=x,edge_index_intra=ei_intra,edge_index_inter=ei_inter,y=torch.FloatTensor([0]),pos=pos)
        batch = Batch.from_data_list([d]).to(device)
        with torch.no_grad(): pred = model(batch)
        results.append({"mol_id":mid,"score":float(pred.cpu().numpy().flatten()[0])})
    except Exception as e:
        results.append({"mol_id":mid,"score":None})
with open(output_json,"w") as f: json.dump(results,f)
'''
    with open(pred_script_path, "w") as f:
        f.write(pred_script_content)

    r = subprocess.run([python_path, pred_script_path, input_json, output_json],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  GIGN 실행 실패: {r.stderr[:300]}")
        return {}

    with open(output_json) as f:
        preds = json.load(f)

    results = {}
    for pred in preds:
        if pred.get("score") is not None:
            results[pred["mol_id"]] = round(pred["score"], 3)

    print(f"  GIGN 완료: {len(results)}개")
    return results


def run_psichic(df, tmpdir):
    """PSICHIC 예측"""
    print("\n[PSICHIC] 예측 시작...")
    python_path = CONDA_PYTHON["PSICHIC"]
    psichic_dir = os.path.abspath(os.path.join(PROJECT_DIR, "external_models/PSICHIC"))

    screen_csv = os.path.join(tmpdir, "psichic_screen.csv")
    import pandas as _pd
    screen_df = _pd.DataFrame({
        "Protein": [TARGET_SEQUENCE] * len(df),
        "Ligand": df["SMILES"].tolist(),
    })
    screen_df.to_csv(screen_csv, index=False)

    result_dir = os.path.join(tmpdir, "psichic_result")

    r = subprocess.run(
        [python_path, os.path.join(psichic_dir, "screening.py"),
         "--screenfile", screen_csv,
         "--trained_model_path", os.path.join(psichic_dir, "trained_weights", "PDBv2020_PSICHIC"),
         "--result_path", result_dir,
         "--device", "cuda:0",
         "--save_interpret", "False"],
        capture_output=True, text=True,
        cwd=psichic_dir,
    )
    if r.returncode != 0:
        print(f"  PSICHIC 실행 실패: {r.stderr[:300]}")
        return {}

    result_csv = os.path.join(result_dir, "screening.csv")
    if not os.path.exists(result_csv):
        print("  PSICHIC 결과 파일 없음")
        return {}

    pred_df = pd.read_csv(result_csv)
    results = {}
    for i, row in pred_df.iterrows():
        if i < len(df) and "predicted_binding_affinity" in pred_df.columns:
            mol_id = df.iloc[i]["ID"]
            results[mol_id] = round(float(row["predicted_binding_affinity"]), 3)

    print(f"  PSICHIC 완료: {len(results)}개")
    return results


def run_nmdn(df, tmpdir):
    """DiffDock-NMDN 예측 (SMINA docking + NMDN scoring)"""
    print("\n[NMDN] 예측 시작...")
    python_path = CONDA_PYTHON["NMDN"]
    nmdn_dir = os.path.abspath(os.path.join(PROJECT_DIR, "external_models/DiffDock-NMDN"))
    pignet2_python = CONDA_PYTHON["PIGNet2"]

    # docking 먼저
    docked_sdfs = {}
    for _, row in df.iterrows():
        mol_id = row["ID"]
        smi = row["SMILES"]
        print(f"  Docking {mol_id} ({row.get('Drug_Name','')})...", end=" ")
        docked = smina_dock(smi, mol_id, PROTEIN_PDB, tmpdir, pignet2_python)
        if docked:
            docked_sdfs[mol_id] = docked
            print("OK")
        else:
            print("FAIL")

    if not docked_sdfs:
        print("  NMDN: 모든 docking 실패!")
        return {}

    results = {}
    for mol_id, docked_sdf in docked_sdfs.items():
        output_csv = os.path.join(tmpdir, f"nmdn_{mol_id}.csv")
        cmd = [
            python_path, os.path.join(nmdn_dir, "predict.py"),
            "--prot", PROTEIN_PDB,
            "--ligs", docked_sdf,
            "--save_csv", output_csv,
        ]
        print(f"  NMDN scoring {mol_id}...", end=" ")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=nmdn_dir)
        if r.returncode != 0:
            print("FAIL")
            continue

        try:
            pred_df = pd.read_csv(output_csv)
            score = float(pred_df["pKd-Score"].iloc[0])
            results[mol_id] = round(score, 3)
            print(f"NMDN={score:.3f}")
        except Exception:
            print("parse FAIL")

    print(f"  NMDN 완료: {len(results)}개")
    return results


def compute_metrics(df_results):
    """Spearman 상관 등 계산 — 모든 *_pKd 컬럼 자동 탐색"""
    from scipy.stats import spearmanr, pearsonr

    metrics = {}
    exp_pkd = df_results["Exp_pKd"].values

    # 모든 *_pKd 컬럼을 자동으로 탐색
    model_cols = [c for c in df_results.columns if c.endswith("_pKd") and c != "Exp_pKd"]
    for col in model_cols:
        model = col.replace("_pKd", "")
        pred = df_results[col].values
        valid = ~np.isnan(pred) & ~np.isnan(exp_pkd)
        if valid.sum() < 3:
            continue

        sp_corr, sp_p = spearmanr(exp_pkd[valid], pred[valid])
        pe_corr, pe_p = pearsonr(exp_pkd[valid], pred[valid])
        rmse = np.sqrt(np.mean((exp_pkd[valid] - pred[valid])**2))
        mae = np.mean(np.abs(exp_pkd[valid] - pred[valid]))

        metrics[model] = {
            "Spearman_r": round(sp_corr, 3),
            "Spearman_p": round(sp_p, 4),
            "Pearson_r": round(pe_corr, 3),
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "N": int(valid.sum()),
        }

    return metrics


def main():
    print("=" * 70)
    print("  EGFR 대조군 검증 (Control Validation)")
    print("=" * 70)

    # 대조군 로드
    df = pd.read_csv(CONTROL_CSV)
    print(f"\n대조군: {len(df)}개 분자")
    print(f"  양성 (알려진 바인더): {(df['Type']=='positive').sum()}개")
    print(f"  음성 (비바인더):      {(df['Type']=='negative').sum()}개")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 각 모델 실행
        pig_results = run_pignet2(df, tmpdir)
        on2_results = run_onionnet2(df, tmpdir)
        dta_results = run_deepdta(df, tmpdir)
        gdta_results = run_graphdta(df, tmpdir)
        mgdta_results = run_mgraphdta(df, tmpdir)
        gign_results = run_gign(df, tmpdir)
        psichic_results = run_psichic(df, tmpdir)
        nmdn_results = run_nmdn(df, tmpdir)

    # 결과 통합
    df["PIGNet2_pKd"] = df["ID"].map(pig_results).astype(float)
    df["OnionNet2_pKd"] = df["ID"].map(on2_results).astype(float)
    df["DeepDTA_pKd"] = df["ID"].map(dta_results).astype(float)
    df["GraphDTA_pKd"] = df["ID"].map(gdta_results).astype(float)
    df["MGraphDTA_pKd"] = df["ID"].map(mgdta_results).astype(float)
    df["GIGN_pKd"] = df["ID"].map(gign_results).astype(float)
    df["PSICHIC_pKd"] = df["ID"].map(psichic_results).astype(float)
    df["NMDN_pKd"] = df["ID"].map(nmdn_results).astype(float)
    df["Exp_pKd"] = df["pKd"].astype(float)

    ALL_MODELS = ["PIGNet2", "OnionNet2", "DeepDTA", "GraphDTA", "MGraphDTA", "GIGN", "PSICHIC", "NMDN"]

    # 결과 출력
    print("\n" + "=" * 150)
    print("  대조군 예측 결과 비교")
    print("=" * 150)
    header = f"{'ID':<14} {'Type':<9} {'Drug':<16} {'Exp_pKd':>8}"
    for m in ALL_MODELS:
        header += f" {m:>10}"
    print(header)
    print("-" * 150)

    for _, row in df.iterrows():
        exp = row["Exp_pKd"]
        line = f"{row['ID']:<14} {row['Type']:<9} {row.get('Drug_Name',''):<16} {exp:>8.2f}"
        for m in ALL_MODELS:
            col = f"{m}_pKd"
            val = row.get(col, np.nan)
            line += f" {val:>10.3f}" if not np.isnan(val) else f" {'N/A':>10}"
        print(line)

    # 모델별 성능 지표
    metrics = compute_metrics(df)
    print("\n" + "=" * 70)
    print("  모델 성능 지표 (전체 대조군)")
    print("=" * 70)
    print(f"{'Model':<12} {'Spearman':>10} {'Pearson':>10} {'RMSE':>8} {'MAE':>8} {'N':>4}")
    print("-" * 60)
    for model, m in metrics.items():
        print(f"{model:<12} {m['Spearman_r']:>10.3f} {m['Pearson_r']:>10.3f} "
              f"{m['RMSE']:>8.3f} {m['MAE']:>8.3f} {m['N']:>4}")

    # 양성 대조군만
    df_pos = df[df["Type"] == "positive"].copy()
    metrics_pos = compute_metrics(df_pos)
    print("\n" + "=" * 70)
    print("  모델 성능 지표 (양성 대조군만)")
    print("=" * 70)
    print(f"{'Model':<12} {'Spearman':>10} {'Pearson':>10} {'RMSE':>8} {'MAE':>8} {'N':>4}")
    print("-" * 60)
    for model, m in metrics_pos.items():
        print(f"{model:<12} {m['Spearman_r']:>10.3f} {m['Pearson_r']:>10.3f} "
              f"{m['RMSE']:>8.3f} {m['MAE']:>8.3f} {m['N']:>4}")

    # 양성 vs 음성 분리 능력
    print("\n" + "=" * 70)
    print("  양성/음성 분리 능력 (평균 예측 pKd)")
    print("=" * 70)
    for model in ALL_MODELS:
        col = f"{model}_pKd"
        if col not in df.columns:
            continue
        pos_mean = df[df["Type"]=="positive"][col].mean()
        neg_mean = df[df["Type"]=="negative"][col].mean()
        gap = pos_mean - neg_mean
        print(f"  {model:<12}: 양성 평균={pos_mean:.3f}, 음성 평균={neg_mean:.3f}, "
              f"차이={gap:+.3f} {'✅ OK' if gap > 0.5 else '⚠️ 약함'}")

    # 앙상블 (평균 percentile)
    print("\n" + "=" * 70)
    print("  앙상블 순위 (Mean Percentile)")
    print("=" * 70)

    for model in ALL_MODELS:
        col = f"{model}_pKd"
        if col in df.columns:
            # 높은 pKd가 좋으므로 ascending=False
            df[f"{model}_rank"] = df[col].rank(ascending=False, method="min")
            df[f"{model}_pctile"] = df[f"{model}_rank"] / len(df)

    pctile_cols = [c for c in df.columns if c.endswith("_pctile")]
    if pctile_cols:
        df["Mean_Pctile"] = df[pctile_cols].mean(axis=1)
        df["Ensemble_Rank"] = df["Mean_Pctile"].rank(method="min")
        df_sorted = df.sort_values("Ensemble_Rank")

        print(f"{'Rank':>4} {'ID':<14} {'Type':<9} {'Drug':<16} {'Exp_pKd':>8} {'Mean_Pctile':>12}")
        print("-" * 70)
        for _, row in df_sorted.iterrows():
            print(f"{int(row['Ensemble_Rank']):>4} {row['ID']:<14} {row['Type']:<9} "
                  f"{row.get('Drug_Name',''):<16} {row['Exp_pKd']:>8.2f} {row['Mean_Pctile']:>12.3f}")

    # 저장
    output_path = os.path.join(OUTPUT_DIR, "control_validation_egfr.csv")
    df.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")
    print("\n" + "=" * 70)
    print("  검증 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()