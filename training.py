from comet_ml import Experiment, get_config
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

import argparse
from easydict import EasyDict as ed

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything

from dataset_generator import get_timeseries
from lstm_model import truckLSTM

from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plates", type=str, default=None, help="List of plates to consider, separated by comma.")
    parser.add_argument("-d", "--provider", type=str, default='Movimatica', help="Data provider to use. Default is 'Movimatica'")
    parser.add_argument("-c", "--categories", type=str, default=None, help="List of categories to consider, separated by comma")
    parser.add_argument("-t", "--task", type=str, default='classification', help="Predictive task. Can be 'regression' (RUL prediction) or 'classification' (failure range binary detection).")
    return parser.parse_args()


hparams = ed(
    batch_size = 4, 
    max_epochs = 50,
    hidden_size = 100,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.001,
    dt=10,
    seg_len = 30,
    hot_period = 7,
    provider = 'Movimatica',
    task = 'classification', #/regression
    test_set_plates = ["FY402YC", "FY293YC", "ZB132AR", "ZB131AR", "FY401YC", "ZB150AR", "ZB475AN", "ZB477AN"]
)


def train(hparams):
    seed_everything(1)
    mod_name = 'lstm_reg'
    for f in (hparams.plates, hparams.categories):
        if f is not None:
            mod_name = f"{mod_name}_{f}"
#     csv_logger = CSVLogger('./logs', name=mod_name, version='v0')
    
    experiment = Experiment(
        api_key="guqtwioseJmdXw2iMRtTuxaIn",
        project_name="ai4trucks",
    )
    comet_logger = CometLogger(
        api_key="guqtwioseJmdXw2iMRtTuxaIn",
        project_name="ai4trucks",
    )
    
    dataset = get_timeseries(dt=hparams.dt, hot_period=hparams.hot_period,
                             limit_plate=hparams.plates,
                             limit_cat=hparams.categories, 
                             use_rul=True if args.task=="regression" else False,
                             limit_provider=hparams.provider
                            )

    
    model = truckLSTM(hparams, dataset)
    experiment.set_model_graph(str(model))

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        logger=comet_logger,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
#         log_every_n_steps=5,
        default_root_dir='./logs'
    )
    trainer.fit(model)

    
if __name__ == '__main__':
    args = parse_args()
    plates = args.plates.split(',') if args.plates is not None else None
    categories = args.categories
    
    if args.categories:
        categories = categories.split(',')
    if categories is None and plates is not None:
        categories = len(plates)*[None]
                
    hparams.task = args.task
    hparams.provider = args.provider
  
    if any((plates, categories)):
        for pl, cat in zip(plates, categories):
            hparams.update({"plates": pl, "categories": cat})
            train(hparams)
    else:
        hparams.update({"plates": None, "categories": None})
        train(hparams)