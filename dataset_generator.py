from inspect_tables import read_data
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


idx = pd.IndexSlice


def binder_mov(agg_funs, apply_offset=True, debug=False):
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

    for a_f in agg_funs:
        if a_f in ["std", "count"]:
            df_interpol.loc[:, ('odometer', a_f)] = df_interpol.odometer[a_f].fillna(0)
            df_interpol.loc[:, ('engineHours', a_f)] = df_interpol.engineHours[a_f].fillna(0)
            df_interpol.loc[:, ('position_speed', a_f)] = df_interpol.position_speed[a_f].fillna(0)
        else:
            df_interpol.loc[:, ('odometer', a_f)] = df_interpol.odometer[a_f].interpolate()
            df_interpol.loc[:, ('engineHours', a_f)] = df_interpol.engineHours[a_f].interpolate()
            df_interpol.loc[:, ('position_speed', a_f)] = df_interpol.position_speed[a_f].interpolate()
        

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

def failure_list(dt, care_category=False, include_eurom=False):
    cat_fatture = pd.read_excel("excels/eventi_manutenzioni_esterne (da fatture).xlsx",
                                sheet_name="Categorie", usecols="A:H").rename(columns={"Categoria componente": "Categoria"})

    cat_fatture.Data = pd.to_datetime(cat_fatture.Data)
    cat_fatture.Categoria = cat_fatture.Categoria.astype("category").apply(lambda x: x if x != "Impianto di lubrificazione motore" else "Impianto lubrificazione motore")
    cat_fatture = cat_fatture[~cat_fatture.Categoria.isin(["?", "Generale"])&(cat_fatture.Tagliando == "No")&(cat_fatture.Revisione == "No")]
    cat_fatture.Categoria = cat_fatture.Categoria.astype(str)

    if include_eurom:
        cat_fatture2 = read_data("dataset/EUROMASTER_GetDossiers.csv")[["dossierId", "date", "plate", "visitReason"]]
        cat_fatture2 = cat_fatture2.rename(columns={"date": "Data", 
                                                    "plate": "Targa",
                                                    "dossierId": "ID",
                                                    "visitReason": "Manutenzione"
                                              })

        cat_fatture2.Manutenzione = cat_fatture2.Manutenzione.str.title()
        cat_fatture2 = cat_fatture2[~cat_fatture2.Manutenzione.isin(["COSTI TRASPORTO COME DA ACCORDI"])]
        cat_fatture2 =  cat_fatture2.assign(Revisione="No",
                                            Tagliando="No",
                                            Categoria="Ruote",
                                           )
        cat_fatture2["Componente"] = cat_fatture2.Manutenzione.apply(lambda x: "Assi" if "ass" in x.lower() else "Pneumatici")
        
        cat_fatture = cat_fatture.append(cat_fatture2)
    

    cat_fatture['delta'] = np.inf
    # Removing points closer than $dt days
    if care_category:
        cat_fatture = cat_fatture.set_index(["Targa", "Data", "Categoria"]).sort_index()
        for ix in cat_fatture.index:
            cat_fatture.loc[ idx[ix], 'delta'] = cat_fatture.loc[idx[ix], 'Data' ].diff().dt.days
    else:
        cat_fatture = cat_fatture.set_index(["Targa", "Data"]).sort_index()
        for plate, gp in cat_fatture.groupby("Targa"):
            cat_fatture.loc[idx[plate], "delta"] = gp.assign(date=gp.index.get_level_values("Data")).date.diff().dt.days

    kept_fatture = cat_fatture[cat_fatture.delta.isna()|(cat_fatture.delta > dt)]

    print(f"Numero di fatture considerate: {kept_fatture.shape[0]}")

    return kept_fatture.reset_index()
    

def get_timeseries(dt=20, hot_period=7, int_idx=True, single_class=True, verbose=False):
    '''
    time_window of the Dataset obj should be the same of dt, but in this way we would miss too many failures
    '''
    which_data = {
        "movimatica": binder_mov,
    }
    agg_funs = ["mean", "std", "min", "max", "count", "median"]
    # -------------------- #
    
    cat_fatture = failure_list(dt)
    dataset = pd.DataFrame()
    for fornitore, binder in which_data.items():
        data = binder(agg_funs, debug=verbose)
        data = data.assign(**{cat : 0. for cat in cat_fatture.Categoria.unique()})
        
        for i, row in cat_fatture.iterrows():
            if (row.Targa, row.Data) in data.index:
                data.loc[(row.Targa, row.Data), row.Categoria] = 1.
        
        data["any_failure"] = data[np.array(cat_fatture.Categoria.unique())].sum(axis=1).gt(0).astype(float)
        
        dataset = dataset.append(data)

    if single_class: dataset = dataset.drop(['Impianto frenante', 'Impianto lubrificazione motore',
       'Impianto elettrico', 'Meccanica', "Impianto d'alimentazione",
       'Impianto di scarico', 'Sensoristica', 'Idraulica'], axis=1)
        
    if verbose:
        print(f"Dataset has {len(dataset)} time series")
    
    dataset = dataset.reset_index()
    
    # Todo: customize for specific failure category
    dataset["attended_failure"] = 0.
    for i in dataset[dataset.any_failure == 1].index:
        dataset.loc[idx[i-hot_period+1:i], "attended_failure"] = 1.
        
    dataset.date = dataset.date.factorize()[0]
#     dataset.plate = dataset.plate.factorize()[0]
    
    # RUL
    dataset = dataset.assign(RUL = dataset[dataset.any_failure == 1].date)
    dataset["RUL"] = dataset.RUL.fillna(method="bfill") - dataset.date
    dataset = dataset.drop(dataset[(dataset.RUL < 0)|dataset.RUL.isna()].index) # removing samples with no failure after
    
    return dataset


def get_timeseries_old(dt=30, verbose=False):
    # -------------------- #
    # Parameter definition #
    which_data = {
        "movimatica": binder_mov,
    }
    agg_funs = ["mean", "std", "min", "max", "count", "median"]
    # -------------------- #
    
    cat_fatture = failure_list(dt)

    dataset = []
    for fornitore, binder in which_data.items():
        data_mov = binder_mov(agg_funs, debug=verbose)
        
        for i, row in cat_fatture.iterrows():
            if row.Targa in data_mov.index.get_level_values(0):
                series = data_mov.xs(row.Targa, level=0).truncate(row.Data - pd.Timedelta(days=dt-1), row.Data)
                dataset.append({
                    "time_series": series.assign(failure=(series.index == row.Data).astype(int)),
                    "Categoria": row.Categoria,
                    "Targa": row.Targa,
                    "ID": row.ID,
                    "Revisione": row.Revisione,
                    "Tagliando": row.Tagliando,
                    "Componente": row.Componente,
                    "Manutenzione": row.Manutenzione,
                    "Fornitore": fornitore,
                })

    if verbose:
        print(f"Dataset has {len(dataset)} time series")
    
    return dataset
        
        # Feature engineering
     
    
class FailureDataset(Dataset):
    def __init__(self, data, sequence_length=30, label_col="attended_failure", removed_cols=[None]):
        """
        Takes as input a dataframe with the last column as label, X features are un-normalized.
        """
        self.sequence_length = sequence_length
        sequence_cols = data.columns.difference(['plate', 'date', 'any_failure', 'RUL'] + [label_col] + removed_cols)
        self.X_values = []
        self.y_values = []
        for pl in data.plate.unique():
            data_array = MinMaxScaler().fit_transform(data[data.plate == pl][sequence_cols].values)
            label_array = data[data.plate == pl][label_col].values
            n_el = data_array.shape[0]
            for i, f in zip(range(0, n_el-sequence_length), range(sequence_length, n_el)):
                self.X_values += [data_array[i:f, :]]
                self.y_values += [label_array[f]]

        self.X_values = torch.Tensor(self.X_values)
        self.y_values = torch.Tensor(self.y_values).unsqueeze(1)

        
    def __len__(self):
        return len(self.y_values)
    
    @staticmethod
    def gen_sequence(truck_series, seq_length, seq_cols):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        data_array = truck_series[seq_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]
    
    def __getitem__(self, index):
        return (self.X_values[index],
                self.y_values[index]
               )

# # Data Labeling - generate column RUL (Remaining Useful Life)
# rul = dataset.groupby('id')['cycle'].max().reset_index()
# rul.columns = ['id', 'max']
# train_df = train_df.merge(rul, on=['id'], how='left')
# train_df['RUL'] = train_df['max'] - train_df['cycle']
# train_df.drop('max', axis=1, inplace=True)
# train_df.head()


