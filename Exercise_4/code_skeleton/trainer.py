import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import StepLR

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network
        yhat = self._model(x)
        # -calculate the loss
        loss = self._crit(yhat, y.float())
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()
    
    def val_test_step(self, x, y):
        # predict
        yhat = self._model(x)
        # propagate through the network and calculate the loss and predictions
        val_loss = self._crit(yhat, y.float())
        # return the loss and the predictions
        return val_loss, yhat
        
    def train_epoch(self):
        loss = 0
        # set training mode
        self._model.train()
        # iterate through the training set
        for images, labels in self._train_dl:
            if self._cuda:
                # transfer the batch to "cuda()" -> the gpu if a gpu is given
                images = images.to(t.device("cuda"))
                labels = labels.to(t.device("cuda"))

            # perform a training step
            loss += self.train_step(images, labels)
        # calculate the average loss for the epoch and return it
        print("Train loss:\t" + str((loss/len(self._train_dl))))
        return loss/len(self._train_dl)

    def val_test(self):
        # set eval mode
        self._model.eval()
        # disable gradient computation
        val_loss_test = []
        with t.no_grad():
            # iterate through the validation set
            for images, labels in self._val_test_dl:
                if self._cuda:
                    # transfer the batch to the gpu if given
                    images = images.to(t.device("cuda"))
                    labels = labels.to(t.device("cuda"))
                # perform a validation step
                val_loss_batch = self.val_test_step(images, labels)

                val_loss_test.append(val_loss_batch[0].item())

        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice.
        # You might want to calculate these metrics in designated functions
        average_loss = sum(val_loss_test)/len(val_loss_test)

        # return the loss and print the calculated metrics
        print("Test loss:\t" + str(average_loss))
        return average_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        self.epoch = 0
        self.train_loss = []
        self.val_loss = []
        scheduler = StepLR(self._optim, step_size=10, gamma=0.1)
        while True:

            # stop by epoch number
            if self.epoch == epochs or epochs == -1:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            loss = self.train_epoch()
            # append the losses to the respective lists
            self.train_loss.append(loss)
            # Get and append the loss in the validation step
            loss_val = self.val_test()
            self.val_loss.append(loss_val)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if self.epoch == 20 or self.epoch == 40:
                self.save_checkpoint(epoch=self.epoch)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            print("Epoch: " + str(self.epoch))
            if self.epoch >= self._early_stopping_patience:
                print('Early stopping')
                break
            self.epoch += 1
            scheduler.step()

        self.save_checkpoint(epoch=self.epoch)
        # return the losses for both training and validation
        return self.train_loss, self.val_loss
