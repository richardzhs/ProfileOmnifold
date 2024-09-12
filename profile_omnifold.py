import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split, ShuffleSplit
import copy
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from torch import nn, optim
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from scipy import optimize
dvc = "cuda" if torch.cuda.is_available() else "cpu"

def reweight(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

# Binary crossentropy for classifying two samples with weights
# Weights are "hidden" by zipping in y_true (the labels)

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = - weights * ((y_true) * K.log(y_pred) +
                          (1 - y_true) * K.log(1 - y_pred))

    return K.mean(t_loss)

def density_ratio_classifier(xvals, yvals, weights, xeval, model, verbose):

    X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(xvals, yvals, weights)

    # zip ("hide") the weights with the labels
    Y_train = np.stack((Y_train, w_train), axis=1)
    Y_test = np.stack((Y_test, w_test), axis=1)   


    model.compile(loss=weighted_binary_crossentropy,
                  optimizer='Adam',
                  metrics=['accuracy'])
    model.fit(X_train,
              Y_train,
              epochs=10, # more epochs, learning curve
              batch_size=10000,
              validation_data=(X_test, Y_test),
              verbose=verbose)
    return reweight(xeval,model)


def omnifold(y,x_mc,y_mc,iterations,verbose=0):

    weights = np.empty(shape=(iterations, 2, len(x_mc)))
    # shape = (iteration, step, event)

    labels0 = np.zeros(len(x_mc))
    labels_unknown = np.ones(len(y))
    labels_unknown_step2 = np.ones(len(x_mc))

    xvals_1 = np.concatenate((y_mc, y))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((x_mc, x_mc))
    yvals_2 = np.concatenate((labels0, labels_unknown_step2))
    
    inputs1 = Input((y.shape[1], ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs1)
    hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs1 = Dense(1, activation='sigmoid')(hidden_layer_3)

    model1 = Model(inputs=inputs1, outputs=outputs1)
    
    inputs2 = Input((x_mc.shape[1], ))
    hidden_layer_4 = Dense(50, activation='relu')(inputs2)
    hidden_layer_5 = Dense(50, activation='relu')(hidden_layer_4)
    hidden_layer_6 = Dense(50, activation='relu')(hidden_layer_5)
    outputs2 = Dense(1, activation='sigmoid')(hidden_layer_6)

    model2 = Model(inputs=inputs2, outputs=outputs2)
    
    
    # initial iterative weights are ones
    weights_pull = np.ones(len(y_mc))
    weights_push = np.ones(len(y_mc))

    for i in range(iterations):

        if (verbose>0):
            print("\nITERATION: {}\n".format(i + 1))
            pass

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose>0):
            print("STEP 1\n")
            pass

        weights_1 = np.concatenate((weights_push, np.ones(len(y))))

        weights_pull = weights_push * density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose)
        weights[i, :1, :] = weights_pull

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose)
        weights[i, 1:2, :] = weights_push
        pass

    return weights


def nonparametric_profile_omnifold(y, x_mc, y_mc, iterations, verbose=0):

    weights = np.empty(shape=(iterations, 3, len(x_mc)))
    # shape = (iteration, step, event)

    labels0 = np.zeros(len(x_mc))
    labels_unknown = np.ones(len(y))
    labels_unknown_step2 = np.ones(len(x_mc))

    xvals_1 = np.concatenate((y_mc, y))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((x_mc, x_mc))
    yvals_2 = np.concatenate((labels0, labels_unknown_step2))
    
    inputs1 = Input((y.shape[1], ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs1)
    hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs1 = Dense(1, activation='sigmoid')(hidden_layer_3)

    model1 = Model(inputs=inputs1, outputs=outputs1)
    
    inputs2 = Input((x_mc.shape[1], ))
    hidden_layer_4 = Dense(50, activation='relu')(inputs2)
    hidden_layer_5 = Dense(50, activation='relu')(hidden_layer_4)
    hidden_layer_6 = Dense(50, activation='relu')(hidden_layer_5)
    outputs2 = Dense(1, activation='sigmoid')(hidden_layer_6)

    model2 = Model(inputs=inputs2, outputs=outputs2)

    # initial iterative weights are ones
    weights_pull = np.ones(len(y_mc))
    weights_push = np.ones(len(y_mc))
    
    # initial weights on the response kernel
    w = np.ones(len(y_mc))

    for i in range(iterations):

        if (verbose>0):
            print("\nITERATION: {}\n".format(i + 1))
            pass

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose>0):
            print("STEP 1\n")
            pass

        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))

        weights_pull = weights_push * density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose)
        weights[i, :1, :] = weights_pull


        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose)
        weights[i, 1:2, :] = weights_push


        # STEP 3: update w
        if (verbose>0):
            print("\nSTEP 3\n")
            print(f'Update w...')
            pass
        w =  w * weights_pull / weights_push
        weights[i, 2:3, :] = w
        
    return weights


def ad_hoc_penalized_profile_omnifold(y, x_mc, y_mc, iterations, w_func, theta_bar, theta0, no_penalty=False, verbose=0):

    weights = np.empty(shape=(iterations, 4, len(x_mc)))
    # shape = (iteration, step, event)

    labels0 = np.zeros(len(x_mc))
    labels_unknown = np.ones(len(y))
    labels_unknown_step2 = np.ones(len(x_mc))

    xvals_1 = np.concatenate((y_mc, y))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((x_mc, x_mc))
    yvals_2 = np.concatenate((labels0, labels_unknown_step2))
    
    inputs1 = Input((y.shape[1], ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs1)
    hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs1 = Dense(1, activation='sigmoid')(hidden_layer_3)

    model1 = Model(inputs=inputs1, outputs=outputs1)
    
    inputs2 = Input((x_mc.shape[1], ))
    hidden_layer_4 = Dense(50, activation='relu')(inputs2)
    hidden_layer_5 = Dense(50, activation='relu')(hidden_layer_4)
    hidden_layer_6 = Dense(50, activation='relu')(hidden_layer_5)
    outputs2 = Dense(1, activation='sigmoid')(hidden_layer_6)

    model2 = Model(inputs=inputs2, outputs=outputs2)

    # initial iterative weights are ones
    weights_pull = np.ones(len(y_mc))
    weights_push = np.ones(len(y_mc))
    
    theta = theta0

    for i in range(iterations):
        # initial weights on the response kernel are determined by the initial theta
        w = w_func(theta)
        if (isinstance(w, torch.Tensor)):
            w = w.cpu().numpy().flatten()

        if (verbose>0):
            print("\nITERATION: {}\n".format(i + 1))
            pass

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose>0):
            print("STEP 1\n")
            pass

        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))


        weights_pull = weights_push * density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose)
        weights[i, 0, :] = weights_pull


        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.


        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose)
        weights[i, 1, :] = weights_push

        
        # STEP 3: update theta
        if (verbose>0):
            print("\nSTEP 3\n")
            print(f'Value of Theta before update: {theta}')
            pass
        
        def theta_loss(x):
            w_theta = w_func(x[0])
            if (isinstance(w_theta, torch.Tensor)):
                w_theta = w_theta.cpu().numpy().flatten()
            if no_penalty == False:
                return np.mean((w_theta - w * weights_pull / weights_push)**2) + (x[0]-theta_bar)**2/2
            else:
                return np.mean((w_theta - w * weights_pull / weights_push)**2)
        
        solution = optimize.minimize(fun=theta_loss, x0=theta, bounds=[(0,3)])
        theta = solution.x[0]

        if (verbose>0):
            print(f'Updated value of Theta: {theta}')
        w = w_func(theta)
        if (isinstance(w, torch.Tensor)):
            w = w.cpu().numpy().flatten()
        weights[i, 2, :] = w
        weights[i, 3, :] = theta
        
    return weights


def penalized_profile_omnifold(y, x_mc, y_mc, iterations, w_func, w_func_derivative, theta_bar, theta0, no_penalty=False, verbose=0):

    weights = np.empty(shape=(iterations, 4, len(x_mc)))
    # shape = (iteration, step, event)

    labels0 = np.zeros(len(x_mc))
    labels_unknown = np.ones(len(y))
    labels_unknown_step2 = np.ones(len(x_mc))

    xvals_1 = np.concatenate((y_mc, y))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((x_mc, x_mc))
    yvals_2 = np.concatenate((labels0, labels_unknown_step2))
    
    inputs1 = Input((y.shape[1], ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs1)
    hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs1 = Dense(1, activation='sigmoid')(hidden_layer_3)

    model1 = Model(inputs=inputs1, outputs=outputs1)
    
    inputs2 = Input((x_mc.shape[1], ))
    hidden_layer_4 = Dense(50, activation='relu')(inputs2)
    hidden_layer_5 = Dense(50, activation='relu')(hidden_layer_4)
    hidden_layer_6 = Dense(50, activation='relu')(hidden_layer_5)
    outputs2 = Dense(1, activation='sigmoid')(hidden_layer_6)

    model2 = Model(inputs=inputs2, outputs=outputs2)

    # initial iterative weights are ones
    weights_pull = np.ones(len(y_mc))
    weights_push = np.ones(len(y_mc))
    
    theta = theta0

    for i in range(iterations):
        # initial weights on the response kernel are determined by the initial theta
        w = w_func(theta)
        if (isinstance(w, torch.Tensor)):
            w = w.cpu().numpy().flatten()

        if (verbose>0):
            print("\nITERATION: {}\n".format(i + 1))
            pass

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose>0):
            print("STEP 1\n")
            pass

        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))

        ry = density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose)
        weights_pull = weights_push * ry
        weights[i, 0, :] = weights_pull
        
        
        # STEP 2: update theta
        if (verbose>0):
            print("\nSTEP 2\n")
            print(f'Value of Theta before update: {theta}')
            pass
        
        def theta_func(x):
            w_next = w_func(x)
            delta_w_next = w_func_derivative(x)
            if (isinstance(w_next, torch.Tensor)):
                w_next = w_next.cpu().numpy().flatten()
                delta_w_next = delta_w_next.cpu().numpy().flatten()
            if no_penalty == False:
                return x - theta_bar - np.mean(w*weights_push*delta_w_next/w_next*ry)
            else:
                return np.mean(w*weights_push*delta_w_next/w_next*ry)
        
        solution = optimize.root_scalar(theta_func, bracket = [0,3], method='bisect')
        theta = solution.root

        if (verbose>0):
            print(f'Updated value of Theta: {theta}')
        w_next = w_func(theta)
        if (isinstance(w_next, torch.Tensor)):
            w_next = w_next.cpu().numpy().flatten()
        weights[i, 2, :] = w_next
        weights[i, 3, :] = theta


        # STEP 3: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 3\n")
            pass

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        
        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose)
        
        weights_3 = np.concatenate((np.ones(len(x_mc)), w_next))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.
        
        weights_push = weights_push * 2 / (1 + density_ratio_classifier(xvals_2, yvals_2, weights_3, x_mc, model2, verbose))
        weights[i, 1, :] = weights_push
        
    return weights


def profile_omnifold_known_nuisance(y, x_mc, y_mc, iterations, w, verbose=0):

    weights = np.empty(shape=(iterations, 2, len(x_mc)))
    # shape = (iteration, step, event)

    labels0 = np.zeros(len(x_mc))
    labels_unknown = np.ones(len(y))
    labels_unknown_step2 = np.ones(len(x_mc))

    xvals_1 = np.concatenate((y_mc, y))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((x_mc, x_mc))
    yvals_2 = np.concatenate((labels0, labels_unknown_step2))
    
    inputs1 = Input((y.shape[1], ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs1)
    hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs1 = Dense(1, activation='sigmoid')(hidden_layer_3)

    model1 = Model(inputs=inputs1, outputs=outputs1)
    
    inputs2 = Input((x_mc.shape[1], ))
    hidden_layer_4 = Dense(50, activation='relu')(inputs2)
    hidden_layer_5 = Dense(50, activation='relu')(hidden_layer_4)
    hidden_layer_6 = Dense(50, activation='relu')(hidden_layer_5)
    outputs2 = Dense(1, activation='sigmoid')(hidden_layer_6)

    model2 = Model(inputs=inputs2, outputs=outputs2)

    # initial iterative weights are ones
    weights_pull = np.ones(len(y_mc))
    weights_push = np.ones(len(y_mc))
    

    for i in range(iterations):

        if (verbose>0):
            print("\nITERATION: {}\n".format(i + 1))
            pass

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose>0):
            print("STEP 1\n")
            pass

        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))

        weights_pull = weights_push * density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose)
        weights[i, :1, :] = weights_pull


        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose)
        weights[i, 1:2, :] = weights_push

    return weights

#######################################################################################################
# The classes below are neural networks for training w(y,x,theta) (adapted from https://github.com/jaychan-hep/UnbinnedProfiledUnfolding)
#

class w_dataset(Dataset):
    def __init__(self, T0, R0, theta0, T1, R1, theta1):
        super(w_dataset, self).__init__()
        
        self.T = np.concatenate([T0, T1])
        self.R = np.concatenate([R0, R1])
        self.theta = np.concatenate([theta0, theta1])
        self.label = np.concatenate([np.zeros((len(T0), 1)), np.ones((len(T1), 1))])
                
    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        return self.T[idx], self.R[idx], self.theta[idx], self.label[idx]

    
class test_dataset(Dataset):
    def __init__(self, T, R):
        super(test_dataset, self).__init__()
        
        self.T = T
        self.R = R
        
    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        return self.T[idx], self.R[idx]


class wRT_network(nn.Module):
    def __init__(self, nodes=[50, 50, 50], sigmoid=True, std_params=None, n_inputs=4, dropouts=[0, 0.1, 0, 0]):
        super(wRT_network, self).__init__()
        
        self.std_params = std_params
        self.sigmoid = sigmoid
        self.linear_relu_stack = nn.Sequential()
        self.n_inputs = n_inputs
        for i in range(len(nodes)+1):
            self.linear_relu_stack.add_module(f"linear_{i}", nn.Linear(self.n_inputs if i==0 else nodes[i-1], nodes[i] if i<len(nodes) else 1))
            if i<len(nodes):
                self.linear_relu_stack.add_module(f"relu_{i}", nn.ReLU())
                self.linear_relu_stack.add_module(f"batchNorm_{i}", nn.BatchNorm1d(nodes[i]))
            if dropouts[i] > 0: self.linear_relu_stack.add_module(f"dropout_{i}", nn.Dropout(p=dropouts[i]))
        if self.sigmoid: self.linear_relu_stack.add_module("sigmoid", nn.Sigmoid())
        
    def forward(self, T, R, theta):
        T = (T-self.std_params[0])/self.std_params[1]
        R = (R-self.std_params[2])/self.std_params[3]
        x = torch.cat([R, T, theta], dim=1)
        logit = self.linear_relu_stack(x)
        if self.sigmoid:
            return logit / (1 - logit + 0.00000000000001)
        else:
            return torch.exp(logit)

class wT_network(wRT_network):
    def __init__(self, n_inputs=2, *args, **kwargs):
        super(wT_network, self).__init__(n_inputs=n_inputs, *args, **kwargs)
        
    def forward(self, T, R, theta):
        T = (T-self.std_params[0])/self.std_params[1]
        x = torch.cat([T, theta], dim=1)
        logit = self.linear_relu_stack(x)
        if self.sigmoid:
            return logit / (1 - logit + 0.00000000000001)
        else:
            return torch.exp(logit)
        
class theta_module(nn.Module):
    def __init__(self, init_value=0.):
        super(theta_module, self).__init__()
        self.theta = nn.Parameter(torch.ones(1) * init_value)
        
    def forward(self):
        return self.theta
    
def roc_auc(input, target, weight=None):
    fpr, tpr, _ = roc_curve(target, input, sample_weight=weight)
    tpr, fpr = np.array(list(zip(*sorted(zip(tpr, fpr)))))
    return 1 - auc(tpr, fpr)
    
# training and testing utilities
class w_trainer:
    def __init__(self, train_dataloader, val_dataloader, model_w, loss_fn, optimizer, max_epoch=1000, patience=10):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_w = model_w
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.patience = patience
        
        print("===================== Model W =====================")
        print(model_w)

        self.best_state = self.model_w.state_dict()
        self.best_epoch = None
        self.best_val_loss = None
        self.i_try = 0
        self.epoch = 0
        self.size = len(train_dataloader.dataset)
        
    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train_step(self):
        self.model_w.train()
        for batch, (T, R, theta, label) in enumerate(self.train_dataloader):
            T, R, theta, label = T.to(dvc), R.to(dvc), theta.to(dvc), label.to(dvc)

            W = self.model_w(T, R, theta)
            logit = W/(W + 1)
            loss = self.loss_fn(logit, label.double())

            # Backpropagation
            self.backpropagation(loss)

            loss, current = loss.item(), (batch+1) * len(R)
            print("\r" + f"[Epoch {self.epoch:>3d}] [{current:>5d}/{self.size:>5d}] [Train_loss: {loss:>7f}]", end="")
        
    def eval_step(self, data_loader):
        self.model_w.eval()
        Ws, logits, labels = None, None, None
        with torch.no_grad():
            for batch, (T, R, theta, label) in enumerate(data_loader):
                T, R, theta, label = T.to(dvc), R.to(dvc), theta.to(dvc), label.to(dvc)

                W = self.model_w(T, R, theta)
                Ws = torch.cat([Ws, W]) if Ws is not None else W
                labels = torch.cat([labels, label]) if labels is not None else label
            logits = Ws/(Ws + 1)
            loss = self.loss_fn(logits, labels.double())
            auc = roc_auc(logits.cpu().numpy().reshape(-1), labels.cpu().numpy().reshape(-1))
        return loss, auc
    
    def fit(self, n_epoch=None):   
        max_epoch = (self.epoch+n_epoch+1) if n_epoch else self.max_epoch
        
        for epoch in range(self.epoch+1, max_epoch):
            self.epoch = epoch
            
            # train
            self.train_step()

            # evaluate loss for traing set
            train_loss, train_auc = self.eval_step(self.train_dataloader)

            # evaluate loss for validation set
            val_loss, val_auc = self.eval_step(self.val_dataloader)

            print("\r" + " "*(50), end="")
            print("\r" + f"[Epoch {epoch:>3d}] [Train_loss: {train_loss:>7f} Train_auc: {train_auc:>7f}] [Val_loss: {val_loss:>7f} Val_auc: {val_auc:>7f}]")

            if self.best_val_loss == None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model_w.state_dict())
                self.best_epoch = epoch
                self.i_try = 0
            elif self.i_try < self.patience:
                self.i_try += 1
            else:
                print(f"Early stopping! Restore state at epoch {self.best_epoch}.")
                print(f"[Best_val_loss: {self.best_val_loss:>7f}]")
                self.model_w.load_state_dict(self.best_state)
                break
                
def test_w(test_dataloader, model_wRT, model_wT, theta):
       
    # now evaluate performance at the epoch end
    model_wRT.eval()
    model_wT.eval()

    # evaluate loss for test set
    Ts, Rs, Ws = None, None, None
    with torch.no_grad():
        for batch, (T, R) in enumerate(test_dataloader):
            T, R = T.to(dvc), R.to(dvc)
            ones = torch.ones(len(R), 1).to(dvc)

            # Compute weights
            W = model_wRT(T, R, ones*theta)/model_wT(T, R, ones*theta)
            Ts = torch.cat([Ts, T]) if Ts is not None else T
            Rs = torch.cat([Rs, R]) if Rs is not None else R
            Ws = torch.cat([Ws, W]) if Ws is not None else W
    return Ts, Rs, Ws


def make_w_func(test_dataloader, model_wRT, model_wT):
    def w_func(theta):
        _, _, Ws = test_w(test_dataloader, model_wRT, model_wT, theta)
        return Ws
    return w_func

