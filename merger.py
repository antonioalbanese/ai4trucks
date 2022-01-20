from tqdm import tqdm
import pandas as pd
from pathlib import Path

"""
Questo script salva i file di ogni fornitore (nel formato 'FORNITORE/SOTTOCARTELLA/file.json') in files
CSV nella cartella 'dataset'.
"""

ignore_list = [#"VISIRUN", "SCANIA", "MOVIMATICA"
               'CGTISAT_INFOold', 'TIMBRATURE_ingestioneventhub', 'IP' # different folder structure
              ]
dataset = Path("dataset2")
# DL_path = Path("SEA Data Lake")
DL_path = Path("SDL_2")

dataset.mkdir(exist_ok=True, parents=True)

def merge_json(path):
    df = pd.DataFrame([])
    for f in tqdm(sorted(path.glob('*json')),
                        desc=f"{path.parent.stem}/{path.stem}"
                 ):
        try:
            tmp = pd.read_json(f)
            tmp["filename"] = f.stem.rsplit('-', 1)[0]
            df = df.append(tmp) #TODO: da fare meglio, questo permette di aggregare dati come VISIRUN
        except:
            tmp = pd.read_json(f, typ='series')
            tmp["filename"] = f.stem.rsplit('-', 1)[0]
            df = df.append(tmp, ignore_index=True)
        
    return df

for fornitore in DL_path.iterdir():
    if fornitore.name != "SCANIA": continue
    if fornitore.name in ignore_list: continue

    for tab in fornitore.iterdir():
#         print(tab)
        if "old" in tab.stem.lower(): continue
        if not tab.is_dir(): continue
        output = dataset / f"{fornitore.stem}_{tab.stem}.csv"
        if output.exists(): continue
            
        merge_json(tab).to_csv(output)