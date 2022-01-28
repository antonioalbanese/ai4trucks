from inspect_tables import read_data
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

idx = pd.IndexSlice


def binder_mov(apply_offset=True, debug=False):
    recover_date_start = pd.to_datetime('2021-09-01')
    recover_date_end = pd.to_datetime('2021-09-09')
    
    def set_offset(df):
        d = df.copy()
        recover_date_start = pd.to_datetime('2021-09-01')
        recover_date_end = pd.to_datetime('2021-09-09')

        x = (d[d.index<recover_date_start].index - d.index.min()).days.values
        y = d[d.index<recover_date_start].engineHours.values

        linear = np.poly1d(np.polyfit(x, y, 1))
        real = d.loc[recover_date_end + pd.Timedelta(days=1), "engineHours"]
        real = real[0] if type(real) == pd.Series else real
        offset = linear((recover_date_end - d.index.min()).days) - real

        d.loc[d.index>=recover_date_end, "engineHours"] += offset
        return d
    
    data_mov = read_data("dataset/MOVIMATICA_vehicles.csv",
                         cut_range=True,
                         low_memory=False
                        ).set_index(["date"])[["plate", "odometer",
                                               "position_speed", "engineHours", 
                                               "timestamp"
                                              ]].drop_duplicates()
    
    if debug:
        import seaborn as sns
        from matplotlib import pyplot as plt
        
        print(data_mov.head())
        print(f"| Movimatica: {data_mov.shape} |")
        
        fig, ax = plt.subplots(1, 2, figsize= (16,4))
        
    # risolve il problema di Movimatica su EngineHours, non è necessario sugli altri
    if apply_offset:
        for pl in data_mov.plate.unique():
            data_mov[data_mov.plate==pl] = set_offset(data_mov[data_mov.plate==pl])
    
    df_interpol = data_mov.groupby("plate")\
                    .resample('D')\
                    .agg({
        "odometer": agg_funs,
        "position_speed": agg_funs,
        "engineHours": agg_funs,
    })

    df_interpol.loc[:, ('odometer', 'mean')] = df_interpol.odometer['mean'].interpolate()
    df_interpol.loc[:, ('engineHours', 'mean')] = df_interpol.engineHours['mean'].interpolate()
    df_interpol.loc[:, ('position_speed', 'mean')] = df_interpol.position_speed['mean'].interpolate()

    if debug:
        for pl in data_mov.plate.unique():
            data = df_interpol.xs(pl)
            ax[0].plot(data.odometer["mean"])
            if not data.engineHours["mean"].is_monotonic:
                ax[1].plot(data.engineHours["mean"], label=pl)
            else:
                ax[1].plot(data.engineHours["mean"])
        ax[0].set_ylabel("km")
        ax[1].set_ylabel("hours")

        ax[0].set_title("Odometer")
        ax[1].set_title("engineHours")

        ax[0].tick_params(axis='x', labelrotation=90)
        ax[1].tick_params(axis='x', labelrotation=90)

        plt.legend(title="⚠️ Not monotonic")
        plt.show()
        
    df_interpol.columns = ['_'.join(c) for c in df_interpol.columns]
    
    return df_interpol
    
# -------------------- #
# Parameter definition #
dt = 30
fatt_distance = 7
which_data = {
    "movimatica": binder_mov,
}
agg_funs = ["mean", "std", "min", "max", "count", "median"]
# -------------------- #

def main():
    cat_fatture = pd.read_excel("excels/eventi_manutenzioni_esterne (da fatture).xlsx", sheet_name="Categorie", usecols="A:H").rename(columns={"Categoria componente": "Categoria"})

    cat_fatture.Data = pd.to_datetime(cat_fatture.Data)
    cat_fatture.Categoria = cat_fatture.Categoria.astype("category").apply(lambda x: x if x != "Impianto di lubrificazione motore" else "Impianto lubrificazione motore")
    cat_fatture = cat_fatture[~cat_fatture.Categoria.isin(["?", "Generale"])&(cat_fatture.Tagliando == "No")]

    # Removing points closer than $fatt_distance days
    df_fatture =  cat_fatture.set_index(["Categoria", "Targa"]).sort_values(by="Data").sort_index()
    df_fatture['delta'] = np.inf

    for ix in df_fatture.index:
        df_fatture.loc[ idx[ix], 'delta'] = df_fatture.loc[idx[ix], 'Data' ].diff().dt.days

    df_fatture = df_fatture[df_fatture.delta.isna()|(df_fatture.delta > fatt_distance)]

    print(f"Numero di fatture considerate: {df_fatture.shape[0]}")

    # Removing points not preceded by at least $dt days without failures
    df_fatture = df_fatture.reset_index().sort_values(by=["Targa", "Data"])
    less_thr = df_fatture.groupby("Targa").apply(lambda x: x.Data.diff().dt.days.between(0, dt, inclusive='right'))
    df_fatture = df_fatture.drop(less_thr.index[less_thr].get_level_values(1))

    for fornitore, binder in which_data.items():
        data_mov = binder_mov()


