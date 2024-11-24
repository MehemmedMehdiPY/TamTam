import os
import torch
from torch import nn
from torchmetrics.functional.classification import accuracy, f1_score, jaccard_index
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple
from tqdm import tqdm


class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Adam, loss_fn: torch.nn.modules.loss.BCELoss, epochs: int, filepath: str, 
                 num_classes: int = 2, device: Optional[str] = None):
        """The class to support training process
        Args:
            model:                    Model to train
            train_loader:             Data Loader for training set
            val_loader:               Data Loader for validation set
            optimizer:                Optimizer
            loss_fn:                  Loss function
            epochs:                   The number of epochs for training
            filepath:                 Filepath to save the training model
            num_classes:              The number of channels in the output
            scheduler:                Learning scheduler
            is_context_loss:          If True, BiSeNet-V1 training will be adjusted
            device:                   Device to use trainer object
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_len = len(self.train_loader)
        self.val_len = len(self.val_loader)
        
        self.epochs = epochs
        self.filepath = filepath
        self.num_classes = num_classes
  
        self.device = (device if device is not None else 'cpu')

        if not os.path.exists('./best_result'):
            os.mkdir('./best_result')

    def train_model(self, epoch: int) -> float:
        """The function to train the model

            Args:
                epoch            int

            Returns:
                train_loss       float
            
            It saves other metrices, such as accuracy, f1 score, mean IOU with TensorBoard.
            Flexible for training with both U-Net and BiSeNet-V1
        """

        self.model.train()
        train_loss = 0
        with tqdm(self.train_loader) as loop:
            # Description of current epoch
            loop.set_description('Epoch {}/{}'.format(epoch + 1, self.epochs))

            for i, (X, Y) in enumerate(loop):
                X = X.to(self.device)
                Y = Y.to(self.device)

                self.optimizer.zero_grad()

                out = self.model(X)
                loss = self.loss_fn(out, Y)

                loss_item = loss.item()
                train_loss += loss_item
                loop.set_postfix_str('Loss: ' + str(round(loss_item, 3)))
                loop.update(1)

                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / self.train_len
        return train_loss

    def evaluate_model(self) -> Tuple[float]:
        """The function to evaluate the model performance.

            Returns:
                  test_loss              float
                  avg_accuracy           torch.float32
                  avg_f1                 torch.float32
        """

        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            avg_accuracy = 0
            avg_f1 = 0

            for X, Y in self.val_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)

                out = self.model(X)

                loss = self.loss_fn(out, Y)
                test_loss += loss.item()

                accuracy_value, f1_value = self.get_metrics(out.argmax(1), Y.argmax(1))

                avg_accuracy += accuracy_value.item()
                avg_f1 += f1_value.item()

        test_loss = test_loss / self.val_len
        avg_accuracy = avg_accuracy / (self.val_len) * 100
        avg_f1 = avg_f1 / self.val_len * 100

        return test_loss, avg_accuracy, avg_f1

    def get_metrics(self, pred, label) -> Tuple[float]: 
        """The function to compute accuracy, mean IOU and F1 score"""
        accuracy_value = accuracy(pred, label, task='multiclass', num_classes=self.num_classes,
                                  average='macro')
        f1_value = f1_score(pred, label, task='multiclass', num_classes=self.num_classes,
                            average='macro')
        return accuracy_value, f1_value

    def run(self, epoch_start: Optional[int] = 0) -> None:
        """The function to control training"""
        writer = SummaryWriter()

        best_f1 = 0

        for epoch in range(epoch_start, self.epochs):
            print('-' * 50)

            train_loss = self.train_model(epoch)
            test_loss, avg_accuracy, avg_f1 = self.evaluate_model()

            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Test", test_loss, epoch + 1)
            writer.add_scalar("Score/Accuracy", avg_accuracy, epoch + 1)
            writer.add_scalar("Score/F1", avg_f1, epoch + 1)

            if avg_f1 > best_f1:
                # Saving the best model
                print('The best model is saved at {:.3f}'.format(avg_f1))
                with open('best_result/best_result.txt', 'w') as fo:
                    fo.write("f1: {}".format(avg_f1))
                self.save_model(filepath=self.filepath)
                best_f1 = avg_f1

        writer.close()

    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        checkpoints = self.model.state_dict()
        torch.save(checkpoints, filepath)