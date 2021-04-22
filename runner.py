import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import accuracy, f1

from resnet import ResNet


class ResNetRunner(pl.LightningModule):
    def __init__(self, in_channels, n_feature_maps, n_classes, kernel_size: list, lr=5e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResNet(self.hparams.in_channels, self.hparams.n_feature_maps, self.hparams.n_classes, self.hparams.kernel_size)
        # example data (Batch = 1, Channels, Length = 10)
        self.example_input_array = torch.rand(1, self.hparams.in_channels, 10, device=self.device)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=30, min_lr=1e-4),
            'monitor': 'Validation step loss'
        }

    def training_step(self, batch, batch_idx):
        y, x = batch
        output = self.model.forward(x)
        loss = F.nll_loss(output, y)
        self.log('Train step loss', loss, on_epoch=True, on_step=False)
        return loss

    def _evaluate(self, x):
        self.eval()
        output = self.model.forward(x)
        pred = torch.argmax(output, dim=-1)
        return pred, output

    def validation_step(self, batch, batch_idx):
        y, x = batch
        pred, output = self._evaluate(x)
        loss = F.nll_loss(output, y)

        dev_acc = accuracy(pred, y)
        dev_f1 = f1(pred, y, num_classes=self.hparams.n_classes)
        self.log('Validation step loss', loss)
        self.log('Validation Acc', dev_acc)
        self.log('Validation F1', dev_f1)
        return loss

    @torch.no_grad()
    def predict(self, x: list) -> list:
        """
        Get prediction for samples.

        :param x: a list of ndarrays with size (channel, length)
        :return: a list of labels
        """
        assert type(x) is list
        ret = []
        for i in x:
            assert i.shape[0] == self.hparams.in_channels and i.ndim == 2
            sample = torch.tensor(i).unsqueeze(0).float()
            pred, _ = self._evaluate(sample)
            ret.append(int(pred.item()))

        return ret
