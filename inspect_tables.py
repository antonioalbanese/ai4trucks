# from autoviz.AutoViz_Class import AutoViz_Class
# AV = AutoViz_Class()

import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from easydict import EasyDict as ed

pd.options.display.max_columns = 100

targhe = [
"FY293YC", "FY295YC", "FY298YC", "FY294YC", "FY296YC", "FV903SK", "FV904SK", "FV906SK", "FV907SK", "FV908SK", "FZ330SC", "FV913SK", "FV914SK", "FY402YC", "FY403YC", "ZB477AN", "ZB473AN", "ZB474AN", "ZB476AN", "ZB475AN", "ZB478AN", "ZB137AR", "ZB139AR", "ZB150AR", "ZB127AR", "ZB132AR", "ZB128AR", "ZB131AR", "ZB130AR", "FY400YC", "ZB135AR", "ZB136AR", "ZB134AR", "ZB373AN", "FY401YC", "CW363HC", "CW367HC", "FP698BP", "FP699BP", "CW365HC", "CW368HC", "CN433CA", "FV989FV", "FV990FV", "FV991FV", "FV995FV", "FV996FV", "FV997FV", "FV992FV", "FV985FV", "FV987FV", "FV988FV", "FV993FV", "FV994FV", "EG181YE", "FV986FV", "EN971TN", "FY299YC", 
]
sns.set_style("whitegrid")

def get_args():
    parser = argparse.ArgumentParser(description='Produce analisi di base su una tabella di dati di un fornitore.')
    parser.add_argument("-d", "--data", type=Path, help="Percorso del file (in CSV).", required=True)
    parser.add_argument("--timestamp_name", type=str, default=None, help="Nome della variabile di timestamp (optional).")
    parser.add_argument("--plate_name", type=str, default=None, help="Nome della variabile Targa (optional).")
    parser.add_argument("--cut_date", nargs='?', default=False, const=True, help="Elimina tutti i dati precedenti a maggio 2021.")

    
    return parser.parse_args()

def read_data(path):
    args = ed({"cut_range": True})
    df = pd.read_csv(path, index_col=0)
    df = df[df.plate.isin(targhe)]
    
    time_cols = [c for c in df.columns if "time".lower() in c or "date" in c.lower()]
    for c in time_cols:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%dT%H:%M:%S%z", utc=True)
#         df[c] = pd.to_datetime(df[c])
#     if args.timestamp_name:
#         df = df.rename(columns={args.timestamp_name: "timestamp"})
#     else:
    inferred_ts = {"timestamp", "datestamp", "PositionDate", "transaction_date", "PositionDateTime"}.intersection(df.columns)
    assert len(inferred_ts) == 1, "Impossibile determinare il nome del timestamp"
    df = df.rename(columns={inferred_ts[0]: "timestamp"})
    df.timestamp = pd.to_datetime(df.timestamp)
    
#     if args.plate_name:
#         df = df.rename(columns={args.plate_name: "plate"})
#     else:
    inferred_pl = [c for c in df.columns if "plate" in c.lower()]
    assert len(inferred_pl) == 1, "Impossibile determinare informazioni di targa"
    df = df.rename(columns={inferred_pl[0]: "plate"})
        
    if args.cut_range:
        anomalies = df[df.timestamp < "2021-01-01"]
        print(f"Eliminati {len(anomalies)} record anomali antecedenti al 2021 (in date {' '.join(anomalies.timestamp.dt.strftime('%d/%m/%Y').unique())})")
        df = df.drop(anomalies.index)
    
    return df
    
def overview(df, timestamp="timestamp", plate="plate", fatture=None):
    date_range = (df[timestamp].max() - df[df[timestamp].dt.year==2021][timestamp].min()).days
    targhe_here = df[plate].unique()

    daterange = f"Dati raccolti tra {df[df[timestamp].dt.year==2021][timestamp].min().strftime('%m/%Y')} e {df[timestamp].max().strftime('%m/%Y')}"
    if (df[timestamp].dt.year!=2021).any(): 
        daterange += " (con alcune eccezioni)\n"
    
    print(f"{daterange}\n\
    {df.shape[1]} parametri totali monitorati\n\
    {df.shape[0]} record nel datalake\n\
    {(df.drop('filename', axis=1) if 'filename' in df.columns else df).drop_duplicates().shape[0]} record non ripetuti\n\
    {len(targhe_here)} truck monitorati\n\
    In media {df.shape[0]/len(targhe_here)} misurazioni per ogni mezzo su 5 mesi\n\
    In media {df.shape[0]/len(targhe_here)/date_range} misurazioni/giorno/mezzo (dettaglio successivamente)\n\
    {len(fatture[fatture.Targa.isin(df[plate].unique())])} fatture associate.")
    
    useless_cols = [c for c in df.columns if len(df[c].unique()) <= 1]
    print("Misurazioni con valore singolo:")
    print(df[useless_cols].iloc[0,:])
    df = df.drop(useless_cols, axis=1)
    print("--> Colonne eliminate")

def plot_date_relplot(df, timestamp="timestamp", plate="Plate"):
    df["Date"] = df[timestamp].dt.date
    key = df.columns[0]
    df = df.sort_values(by="Date").groupby([plate, "Date"], as_index=False)[key].count()
    df['c'] = df[key] / df[key].max()
    
    g = sns.relplot(
        data=df,
        x="Date", y=plate, hue="c", size="c",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=10, sizes=(50, 250), size_norm=(-.2, .8), aspect=1.5
    )

    for t in g._legend.get_texts():
        t.set_text(int(float(t.get_text())*df[key].max()))
    g._legend.set_title("Samples")
    return g

    
def draw_report(df, timestamp="timestamp", plate="plate", fatture=None, per_day=True):
    report = df.groupby(plate).agg({
    timestamp: ["min", "max", 
                "count", pd.Series.nunique
               ],
    }).droplevel(0, axis=1)

    report["fatture"] = report.apply(lambda x: fatture[(fatture.Targa==x.name)&(fatture.Apertura_commessa.dt.date>=x['min'].date())&(fatture.Apertura_commessa.dt.date<=x['max'].date())].ID.count(), axis=1)
    report = report.reset_index()
    
    m1 = "misure"
    m2 = "misure[distinte]"
    
    if per_day:
        report["misure/giorno"] = report["count"] / (report['max'] - report['min']).dt.days
        report["misure[distinte]/giorno"] = report["nunique"] / (report['max'] - report['min']).dt.days
        m1 += "/giorno"
        m2 += "/giorno"
    else:
        report = report.rename(columns={"count": m1, "nunique": m2})
        
    fig, ax = plt.subplots(1,4, figsize=(22,9), sharey=True)
    palette = None
    sns.scatterplot(data=df, x=timestamp, y=plate, palette=palette, ax=ax[0], zorder=3, s=10)
    sns.scatterplot(data=report, x="min", y=plate, palette=palette, ax=ax[0], zorder=4)
    sns.scatterplot(data=report, x="max", y=plate, palette=palette, ax=ax[0], zorder=4)
    ax[0].hlines(data=report, y=plate, xmin="min", xmax="max", linewidth=5, color='lightblue')
    sns.barplot(data=report, x=m1, y=plate, palette=palette, ax=ax[1])
    sns.barplot(data=report, x=m2, y=plate, palette=palette, ax=ax[2])
    sns.barplot(data=report, x="fatture", y=plate, palette=palette, ax=ax[3])

    ax[0].tick_params(axis='x', rotation=90)
    ax[0].set_xlabel("Date range")
    for a in ax:
        a.set_ylabel("")
        a.set_xlabel(a.get_xlabel(), size=17)
    #     a.grid(False)
        a.yaxis.grid(False)
        sns.despine(trim=True)

    plt.tight_layout()
    plt.close()
    return fig
#     fig.savefig(outdir / "report.png")

if __name__ == '__main__':
    args = get_args()
    outdir = Path(args.data.stem)
    outdir.mkdir(esists_ok=True, parents=True)
    
    fatture = ...
    df = read_data(args.data)
    overview(df)
    
    draw_report(df)
    
    
    
    