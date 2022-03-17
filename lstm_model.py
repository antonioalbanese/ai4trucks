import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule

from dataset_generator import get_timeseries, FailureDataset


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
            
        n_features = dataset.shape[1] if not type(dataset) in (list, tuple) else dataset[0].shape[1]
        n_features -= 2
        self.lstm = nn.LSTM(input_size=n_features, 
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
        if type(self.dataset) in (tuple, list):
            self.train_samples = self.dataset[0]
            self.test_samples = self.dataset[1]
        else:
            lim = 32
            in_train = self.dataset.groupby('plate').apply(lambda x: x.date < x.date.max() - lim).droplevel(0)
            self.train_samples = self.dataset[in_train]
            self.test_samples = self.dataset[~in_train]
    
    def train_dataloader(self):
        train_dataset = FailureDataset(self.train_samples, 
                                       label_col="RUL" if self.task == "regression" else "attended_failure")
        return DataLoader(train_dataset,
                          batch_size = self.batch_size, 
                          shuffle = False, 
                          num_workers = 4,
                          drop_last=True)
    
    def val_dataloader(self):
        val_dataset = FailureDataset(self.test_samples,
                                     label_col="RUL" if self.task == "regression" else "attended_failure")
        return DataLoader(val_dataset,
                          batch_size = 1, 
                          shuffle = False, 
                          num_workers = 4,
                          drop_last=True)

    def forward(self, x):
        # [lstm_out]: (batch_size, seq_len, hidden_size)
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        lstm_out, _ = self.lstm(x, (h0, c0))

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
        
        if self.task == 'classification':
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
    
        if self.task == 'classification':
            accuracy = (((torch.sigmoid(y_hat) > .5) * y).sum(1) / y.shape[1]).detach().mean()
            self.log('val_acc', accuracy, on_epoch=True, on_step=True)
        return 
