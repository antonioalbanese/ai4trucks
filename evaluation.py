import torch
from tqdm import tqdm
from dataset_generator import FailureDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate(model):
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for i, plate in tqdm(enumerate(model.dataset.plate.unique())):
            torch_ds = FailureDataset(model.dataset[model.dataset.plate == plate], 
                                      label_col="RUL")
            prediction = []
            fig, ax = plt.subplots(figsize=(12,5))
            for x, y in torch_ds:
                x = x.cuda()

                y_hat = model(x.unsqueeze(0)).squeeze()
                prediction.append(y_hat.cpu().detach().numpy())
            ax.set_title(plate)

            sns.lineplot(data=model.dataset[model.dataset.plate==plate], x="date", y="RUL", ax=ax, label="ground truth")
            ax.plot(np.arange(30, len(prediction)+30), prediction, label="prediction")
            ax.axvline(model.dataset[model.dataset.plate==plate].date.max()-32, ls='--', c='r')
            ax.legend()

            plt.tight_layout()
            fig.savefig(f"Images/im{i}.png")