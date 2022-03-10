import torch
import argparse
from easydict import EasyDict as ed
from torch import nn
import pytorch_lightning as pl
from dataset_generator import get_timeseries, FailureDataset
from torch.utils.data import DataLoader
from dataset_generator import get_timeseries



class truckLSTM(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        for h, val in hparams.items():
            setattr(self, h, val)
            
        self.lstm = nn.LSTM(input_size=hparams.n_features, 
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
        dataset = get_timeseries(dt=10, hot_period=self.hot_period,
                                 limit_plate=self.plates,
                                 limit_cat=self.categories
                                )
        
#         self.train_samples = dataset[~dataset.plate.isin(self.test_set_plates)]
#         self.test_samples = dataset[dataset.plate.isin(self.test_set_plates)]
        
        lim = 32
        self.train_samples = dataset[dataset.date < dataset.date.max() - lim]
        self.test_samples = dataset[dataset.date >= dataset.date.max() - lim]
    
    def train_dataloader(self):
        train_dataset = FailureDataset(self.train_samples)
        return DataLoader(train_dataset,
                          batch_size = self.batch_size, 
                          shuffle = False, 
                          num_workers = 4,
                          drop_last=True
                         )
    
    def val_dataloader(self):
        val_dataset = FailureDataset(self.test_samples)
        return DataLoader(val_dataset,
                          batch_size = 1, 
                          shuffle = False, 
                          num_workers = 4,
                          drop_last=True
                         )

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
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
        
        loss = self.criterion(y_hat, y)
        
        accuracy = (((torch.sigmoid(y_hat) > .5) * y).sum(1) / y.shape[1]).mean()
#         result = pl.TrainResult(minimize=loss, early_stop_on=loss)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        accuracy = (((torch.sigmoid(y_hat) > .5) * y).sum(1) / y.shape[1]).mean()
#         result = pl.EvalResult(checkpoint_on=loss)
        self.log('val_loss', loss, on_epoch=True, on_step=True)
        self.log('val_acc', accuracy, on_epoch=True, on_step=True)
        return 
    
hparams = ed(
    batch_size = 4, 
    criterion = nn.BCEWithLogitsLoss(),
    max_epochs = 15,
#     n_features = 18, # Not hardcoded!!
    n_features = 25,
    hidden_size = 100,
    num_layers = 2,
    dropout = 0.2,
    learning_rate = 0.001,
    seg_len = 30, 
    hot_period = 7,
    test_set_plates = ["FY402YC", "FY293YC", "ZB132AR", "ZB131AR", "FY401YC", "ZB150AR", "ZB475AN", "ZB477AN"]
)

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger


def train(plates=None, categories=None):
    seed_everything(1)
    csv_logger = CSVLogger('./logs', name=f'lstm_{plates}_{categories}', version='0'),

    hparams.update({"plates": plates, "categories": categories})
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        logger=csv_logger,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=1,
    )

    model = truckLSTM(hparams)
    trainer.fit(model)
    
    
if __name__ == '__main__':
    train(foo)