import argparse
from easydict import EasyDict as ed

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

from dataset_generator import get_timeseries, FailureDataset
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plates", type=str, default=None, help="List of plates to consider, separated by comma.")
    parser.add_argument("-d", "--provider", type=str, default='Movimatica', help="Data provider to use. Default is 'Movimatica'")
    parser.add_argument("-c", "--categories", type=str, default=None, help="List of categories to consider, separated by comma")
    parser.add_argument("-t", "--task", type=str, default='classification', help="Predictive task. Can be 'regression' (RUL prediction) or 'classification' (failure range binary detection).")
    return parser.parse_args()


class truckLSTM(LightningModule):
    def __init__(self, hparams, dataset):
        super().__init__()
        for h, val in hparams.items():
            setattr(self, h, val)
            
        self.dataset = dataset
        
        if self.task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.lstm = nn.LSTM(input_size=dataset.shape[1]-1, 
                            hidden_size=hparams.hidden_size,
                            num_layers=hparams.num_layers, 
                            dropout=hparams.dropout, 
                            batch_first=True
                           )
#         self.relu = nn.ReLU()
        self.linear = nn.Linear(hparams.hidden_size, 1)

        self.train_samples = None
        self.test_samples = None


    def setup(self, stage=None):   
#         self.train_samples = dataset[~dataset.plate.isin(self.test_set_plates)]
#         self.test_samples = dataset[dataset.plate.isin(self.test_set_plates)]
        
        lim = 32
        in_train = self.dataset.groupby('plate').apply(lambda x: x.date < x.date.max() - lim).droplevel(0)
        self.train_samples = self.dataset[in_train]
        self.test_samples = self.dataset[~in_train]
    
    def train_dataloader(self):
        train_dataset = FailureDataset(self.train_samples, 
                                       label_col="RUL" if args.task == "regression" else "attended_failure"
                                      )
        return DataLoader(train_dataset,
                          batch_size = self.batch_size, 
                          shuffle = False, 
                          num_workers = 4,
                          drop_last=True
                         )
    
    def val_dataloader(self):
        val_dataset = FailureDataset(self.test_samples,
                                     label_col="RUL" if args.task == "regression" else "attended_failure"
                                    )
        return DataLoader(val_dataset,
                          batch_size = 1, 
                          shuffle = False, 
                          drop_last=True
                         )

    def forward(self, x):
        # [lstm_out]: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
#         lstm_out = self.relu(lstm_out)
        y_pred = self.linear(lstm_out[:, -1, :])
#         print(lstm_out.shape, y_pred.shape)
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    # Try: https://pytorch-lightning.readthedocs.io/en/stable/advanced/sequences.html?
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        assert not y_hat.isnan().any()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        
        if args.task == 'classification':
            accuracy = (((torch.sigmoid(y_hat) > .5) * y).sum(1) / y.shape[1]).detach().mean()
            self.log('train_acc', accuracy, on_epoch=True, on_step=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        assert not x.isnan().any()
        y_hat = self.forward(x)
        assert not y_hat.isnan().any(), (x.shape, x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, on_step=True)
    
        if args.task == 'classification':
            accuracy = (((torch.sigmoid(y_hat) > .5) * y).sum(1) / y.shape[1]).detach().mean()
            self.log('val_acc', accuracy, on_epoch=True, on_step=True)
            
        return 
    
hparams = ed(
    batch_size = 4, 
    max_epochs = 30,
    hidden_size = 100,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.0001,
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
    csv_logger = CSVLogger('./logs', name=mod_name, version='v0'),

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        logger=csv_logger,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=5,
        default_root_dir='./logs'
    )
    
    dataset = get_timeseries(dt=hparams.dt, hot_period=hparams.hot_period,
                             limit_plate=hparams.plates,
                             limit_cat=hparams.categories, 
                             use_rul=True if args.task=="regression" else False,
                             limit_provider=hparams.provider
                            )

    model = truckLSTM(hparams, dataset)
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

#     Parallel(n_jobs=4)(delayed(train)(pl, cat) for pl, cat in zip(plates, categories))    
    if any((plates, categories)):
        for pl, cat in zip(plates, categories):
            hparams.update({"plates": pl, "categories": cat})
            train(hparams)
    else:
        hparams.update({"plates": None, "categories": None})
        train(hparams)
