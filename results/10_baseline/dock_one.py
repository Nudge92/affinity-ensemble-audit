import sys, os, subprocess, tempfile, re
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
SMINA='<path-to>/smina'; PROT='./data/cdk2_clean.pdb'
C=(0.5,27.1,9.0); B=20.0
idx, smi, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
dock=os.path.join(outdir, f'docked_{idx}.sdf')
if os.path.exists(dock) and os.path.getsize(dock)>0: sys.exit(0)
m=Chem.MolFromSmiles(smi)
if m is None: print(f"{idx}\tNA\tparse_fail"); sys.exit(0)
m=Chem.AddHs(m)
if AllChem.EmbedMolecule(m, AllChem.ETKDGv3())!=0:
    if AllChem.EmbedMolecule(m, AllChem.ETKDG())!=0: print(f"{idx}\tNA\tembed_fail"); sys.exit(0)
try: AllChem.MMFFOptimizeMolecule(m)
except Exception: pass
inp=os.path.join(outdir, f'in_{idx}.sdf')
w=Chem.SDWriter(inp); w.write(m); w.close()
cmd=[SMINA,'--receptor',PROT,'--ligand',inp,'--out',dock,
     '--center_x',str(C[0]),'--center_y',str(C[1]),'--center_z',str(C[2]),
     '--size_x',str(B),'--size_y',str(B),'--size_z',str(B),
     '--exhaustiveness','8','--num_modes','1','--quiet','--cpu','1']
try:
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=900)
except subprocess.TimeoutExpired:
    print(f"{idx}\tNA\ttimeout"); sys.exit(0)
if r.returncode!=0 or not os.path.exists(dock) or os.path.getsize(dock)==0:
    print(f"{idx}\tNA\tsmina_fail"); sys.exit(0)
txt=open(dock).read()
mm=re.search(r'minimizedAffinity>\s*\n\s*([-\d.]+)', txt)
print(f"{idx}\t{mm.group(1) if mm else 'NA'}\tok")
os.remove(inp)
