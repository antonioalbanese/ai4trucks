from tqdm import tqdm
import pandas as pd
from pathlib import Path

"""
Questo file salva i file di ogni fornitore (nel formato 'FORNITORE/SOTTOCARTELLA/file.json') in files
CSV nella cartella 'dataset'.
"""

ignore_list = [#"VISIRUN", "SCANIA",
               'CGTISAT_INFOold', 'TIMBRATURE_ingestioneventhub', 'IP' # different folder structure
              ]
dataset = Path("dataset")
dataset.mkdir(exist_ok=True, parents=True)

def merge_json(path):
    df = pd.DataFrame([])
    for f in tqdm(sorted(path.glob('*json')),
                        desc=f"{path.parent.stem}/{path.stem}"
                 ):
        try:
            df = df.append(pd.read_json(f)) #TODO: da fare meglio, questo permette di aggregare dati come VISIRUN
        except:
            df = df.append(pd.read_json(f, typ='series'), ignore_index=True)
        
    return df

for fornitore in Path("SEA Data Lake").iterdir():
    if fornitore.name in ignore_list: continue

    for tab in fornitore.iterdir():
        if not tab.is_dir(): continue
        output = dataset / f"{fornitore.stem}_{tab.stem}.csv"
        if output.exists(): continue
            
        merge_json(tab).to_csv(output)