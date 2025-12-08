import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split, ShuffleSplit
import copy
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from torch import nn
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
dvc = "cuda" if torch.cuda.is_available() else "cpu"

def reweight(events,model,batch_size=10000):
    """
    Reweights the event based on classifier output.

    Parameters
    ----------
    events : array-like, shape=(n_samples, n_features)
        The events to reweight.
    model : keras model
        The classifier model.
    batch_size : int, optional (default=10000)
        The batch size to use for the classifier evaluation.

    Returns
    -------
    weights : array-like, shape=(n_samples,)
        The weights for the events.
    """
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    #weights[np.isinf(weights)] = 1
    weights = np.clip(weights, 0, 10)
    return np.squeeze(np.nan_to_num(weights))

# Binary crossentropy for classifying two samples with weights
# Weights are "hidden" by zipping in y_true (the labels)

def weighted_binary_crossentropy(y_true, y_pred):
    """
    Computes the weighted binary crossentropy loss.

    Parameters
    ----------
    y_true : array-like, shape=(n_samples,)
        The actual labels.
    y_pred : array-like, shape=(n_samples,)
        The predicted labels.
    
    Returns
    -------
    loss : float
        The weighted binary crossentropy loss.
    """
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = - weights * ((y_true) * K.log(y_pred) +
                          (1 - y_true) * K.log(1 - y_pred))

    return K.mean(t_loss)

def weighted_accuracy(y_true, y_pred):
    """
    Computes the weighted classification accuracy
    """
    # Extract labels and weights
    y_actual, weights = tf.gather(y_true, [0], axis=1), tf.gather(y_true, [1], axis=1)
    y_pred_classes = K.round(y_pred)
    # Check correctness
    correct = K.cast(K.equal(y_actual, y_pred_classes), K.floatx())
    # Compute weighted accuracy
    return K.sum(correct * weights) / K.sum(weights)  


def density_ratio_classifier(xvals, yvals, weights, xeval, model, epochs=10, lr=0.001, patience=3, verbose=0, return_history=False):
    """
    Estimate the density ratio by fitting a neural network classifier.

    Parameters
    ----------
    xvals : array-like, shape=(n_samples, n_features)
        The input data.
    yvals : array-like, shape=(n_samples,)
        The labels for the input data.
    weights : array-like, shape=(n_samples,)
        The weights for the input data.
    xeval : array-like, shape=(n_eval_samples, n_features)
        The evaluation points for the density ratio.
    model : keras model
        The model to use for the classifier.
    epochs : int, optional
        The maximum number of epochs to train the classifier.
    lr : float, optional
        The learning rate for the optimizer.
    patience : int, optional
        The number of epochs with no improvement after which training will be stopped.
    verbose : int, optional
        The verbosity level of the classifier.
    return_history : bool, optional
        Whether to return the training history.

    Returns
    -------
    weights : array-like, shape=(n_eval_samples,)
        The estimated density ratio at the evaluation points.
    history : dict, optional
        The training history if return_history is True.

    """

    X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(xvals, yvals, weights)

    # zip ("hide") the weights with the labels
    Y_train = np.stack((Y_train, w_train), axis=1)
    Y_test = np.stack((Y_test, w_test), axis=1)   


    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience, # Stop if no improvement after patience epochs
        restore_best_weights=True
    )
    
    model.compile(loss=weighted_binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy', weighted_accuracy])
    
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=10000,
        validation_data=(X_test, Y_test),
        verbose=verbose,
        callbacks=[early_stopping]
    )
    
    if return_history:
        return reweight(xeval,model), history.history
    else:
        return reweight(xeval,model)


def omnifold(y, x_mc, y_mc, iterations, epochs=10, lr=0.001, patience=3, verbose=0, return_acc=True, return_loss=True):
    """
    Omnifold algorithm.

    Parameters
    ----------
    y : array-like, shape=(n_samples, n_features)
        The data to reweight.
    x_mc : array-like, shape=(n_samples, n_features)
        The Monte Carlo particle-level data.
    y_mc : array-like, shape=(n_samples, n_features)
        The Monte Carlo detector-level data.
    iterations : int
        The number of iterations to run the algorithm.
    epochs : int, optional
        The maximum number of epochs to train the classifier.
    lr : float, optional
        The learning rate for the optimizer.
    patience : int, optional
        The number of epochs with no improvement after which training will be stopped.
    verbose : int, optional
        The verbosity level of the classifier.
    return_acc : bool, optional
        Whether to return the classification accuracy.
    return_loss : bool, optional
        Whether to return the cross-entropy loss.

    Returns
    -------
    result : dict
        A dictionary containing the weights and optionally the accuracy and loss.
    """
    K.clear_session()
    
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

    # step 1 train/validation accuracy in each iteration
    train_acc_step1 = []
    val_acc_step1 = []
    # step 1 train/validation cross-entropy loss in each iteration
    train_loss_step1 = []
    val_loss_step1 = []
    
    # initial iterative weights are ones
    weights_pull = np.ones(len(y_mc))
    weights_push = np.ones(len(y_mc))

    for i in range(iterations):


        print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        print("STEP 1\n")

        weights_1 = np.concatenate((weights_push, np.ones(len(y))))

        ry, history = density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose=verbose, epochs=epochs, lr=lr, patience=patience,
                                               return_history=True)
        weights_pull = weights_push * ry
        weights[i, :1, :] = weights_pull

        if return_acc:
            train_acc_step1.append(history['weighted_accuracy'])
            val_acc_step1.append(history['val_weighted_accuracy'])
        if return_loss:
            train_loss_step1.append(history['loss'])
            val_loss_step1.append(history['val_loss'])

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        print("\nSTEP 2\n")

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose=verbose, epochs=epochs, lr=lr, patience=patience)
        weights[i, 1:2, :] = weights_push

        
    result = {"weights": weights}
    row_names = [f"Iteration{i+1}" for i in range(iterations)]
    col_names = [f"Epoch{i+1}" for i in range(epochs)]
    if return_acc:
        pad_train_acc_step1 = [row + [np.nan] * (len(col_names) - len(row)) for row in train_acc_step1]
        pad_val_acc_step1 = [row + [np.nan] * (len(col_names) - len(row)) for row in val_acc_step1]
        df_train_acc = pd.DataFrame(pad_train_acc_step1, index=row_names, columns=col_names)  
        df_val_acc = pd.DataFrame(pad_val_acc_step1, index=row_names, columns=col_names)
        result["step1_train_acc"] = df_train_acc
        result["step1_val_acc"] = df_val_acc
    if return_loss:
        pad_train_loss_step1 = [row + [np.nan] * (len(col_names) - len(row)) for row in train_loss_step1]
        pad_val_loss_step1 = [row + [np.nan] * (len(col_names) - len(row)) for row in val_loss_step1]
        df_train_loss = pd.DataFrame(pad_train_loss_step1, index=row_names, columns=col_names)
        df_val_loss = pd.DataFrame(pad_val_loss_step1, index=row_names, columns=col_names)    
        result["step1_train_loss"] = df_train_loss
        result["step1_val_loss"] = df_val_loss

    return result



def profile_omnifold(y, x_mc, y_mc, iterations, w_theta, theta_bar, theta0, theta_range, num_grid_points=20, no_penalty=False, epochs=10, 
                     lr=0.001, patience=3, verbose=0, return_Q=False, return_acc=False, return_loss=False):
    """
    Profiled Omnifold algorithm.

    Parameters
    ----------
    y : array-like, shape=(n_samples, n_features)
        The data to reweight.
    x_mc : array-like, shape=(n_samples, n_features)
        The Monte Carlo particle-level data.
    y_mc : array-like, shape=(n_samples, n_features)
        The Monte Carlo detector-level data.
    iterations : int
        The number of iterations to run the algorithm.
    w_theta : function
        w function that is parameterized by theta.
    theta_bar : float
        The central value of theta for the penalty term.
    theta0 : float
        The initial value of theta.
    theta_range : tuple
        The range of theta values to consider (min, max).
    num_grid_points : int, optional
        The number of grid points to evaluate the Q function.
    epochs : int, optional
        The number of training epochs for the neural networks.
    lr : float, optional
        The learning rate for the optimizer.
    patience : int, optional
        The number of epochs with no improvement after which training will be stopped.
    verbose : int, optional
        The verbosity level of the classifier.
    return_Q : bool, optional
        Whether to return the Q function values.
    return_acc : bool, optional
        Whether to return the classification accuracy.
    return_loss : bool, optional
        Whether to return the cross-entropy loss.


    Returns
    -------
    result : dict
        A dictionary containing the weights and optionally the Q function values, accuracy, and loss.
    """ 
    K.clear_session()
    
    weights = np.empty(shape=(iterations, 5, len(x_mc)))
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
    
    # initial theta
    theta = theta0
    # initial weights on the response kernel are determined by the initial theta
    w = w_theta(theta0)
    if (isinstance(w, torch.Tensor)):
        w = w.cpu().numpy().flatten()
    # Q values in each iteration
    Q_iter = []
    # grid of points to evaluate Q function
    theta_min, theta_max = theta_range[0], theta_range[1]
    theta_values = np.linspace(theta_min, theta_max, num_grid_points)
    print('fitting w on grid points...')
    # list of w values evaluated on the grid points for later use
    w_grid = []
    for i in range(num_grid_points):
        w_grid.append(w_theta(theta_values[i]))
        if (isinstance(w_grid[i], torch.Tensor)):
            w_grid[i] = w_grid[i].cpu().numpy().flatten()
    # step 1 train/validation accuracy in each iteration
    train_acc_step1 = []
    val_acc_step1 = []
    # step 1 train/validation cross-entropy loss in each iteration
    train_loss_step1 = []
    val_loss_step1 = []
    
    print(f'Initial Theta: {theta}')

    for i in range(iterations):

        print("\nITERATION: {}\n".format(i + 1))
        

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        print("STEP 1\n")

        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))
        
        ry, history = density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose=verbose, epochs=epochs, lr=lr, patience=patience,
                                               return_history=True)
        weights_pull = weights_push * ry
        weights[i, 0, :] = weights_pull
        weights[i, 4, :] = ry
        if return_acc:
            train_acc_step1.append(history['weighted_accuracy'])
            val_acc_step1.append(history['val_weighted_accuracy'])
        if return_loss:
            train_loss_step1.append(history['loss'])
            val_loss_step1.append(history['val_loss'])


        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        print("\nSTEP 2\n")

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.
        
        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose=verbose, epochs=epochs, lr=lr, patience=patience)
        weights[i, 1, :] = weights_push


        # STEP 3: update theta
        print("\nSTEP 3\n")
        print(f'Value of Theta before update: {theta}')

        def Q(x):
            w_next = w_grid[np.argmin(np.abs(theta_values - x))]

            if i == 0:
                Q_out = np.mean(w*ry*np.log(w_next))
            else:
                Q_out = np.mean(w*weights[i-1, 1, :]*ry*np.log(w_next))
            if no_penalty:
                return Q_out
            else:
                return Q_out - (theta-theta_bar)**2/2
            
        
        theta_min, theta_max = theta_range[0], theta_range[1]
        theta_values = np.linspace(theta_min, theta_max, num_grid_points)

        # Evaluate Q(x) for each x in the grid
        Q_values = [Q(x) for x in theta_values]
        
        if return_Q:
            Q_iter.append(Q_values)

        # Find the x that maximizes Q(x)
        optimal_index = np.argmax(Q_values)
        theta = theta_values[optimal_index]
        
        print(f'Updated value of Theta: {theta}')
        w = w_grid[optimal_index]
        weights[i, 2, :] = w
        weights[i, 3, :] = theta
        

    row_names = [f"Iteration{i+1}" for i in range(iterations)]
    col_names = [f"Epoch{i+1}" for i in range(epochs)]
    result = {"weights": weights, "theta0": theta0}
    if return_Q:
        result["theta_grid"] = theta_values
        result["Q"] = Q_iter
    if return_acc:
        pad_train_acc_step1 = [row + [np.nan] * (len(col_names) - len(row)) for row in train_acc_step1]
        pad_val_acc_step1 = [row + [np.nan] * (len(col_names) - len(row)) for row in val_acc_step1]
        df_train_acc = pd.DataFrame(pad_train_acc_step1, index=row_names, columns=col_names)  
        df_val_acc = pd.DataFrame(pad_val_acc_step1, index=row_names, columns=col_names)
        result["step1_train_acc"] = df_train_acc
        result["step1_val_acc"] = df_val_acc
    if return_loss:
        pad_train_loss_step1 = [row + [np.nan] * (len(col_names) - len(row)) for row in train_loss_step1]
        pad_val_loss_step1 = [row + [np.nan] * (len(col_names) - len(row)) for row in val_loss_step1]
        df_train_loss = pd.DataFrame(pad_train_loss_step1, index=row_names, columns=col_names)
        df_val_loss = pd.DataFrame(pad_val_loss_step1, index=row_names, columns=col_names)    
        result["step1_train_loss"] = df_train_loss
        result["step1_val_loss"] = df_val_loss

    return result




def best_weights(out_list, itr=-1, acc_col = 'step1_val_acc'):
    """
        choose the one that has closest accuracy to 0.5
    """
    best_i = 0
    best_acc = 1
    for i in range(len(out_list)):
        curr_acc = out_list[i][acc_col].ffill(axis=1).iloc[itr, -1]
        if np.abs(best_acc-0.5) > np.abs(curr_acc-0.5):
            best_i = i
            best_acc = curr_acc
    
    nu = out_list[best_i]['weights']
    return nu, best_i

#######################################################################################################
# The classes below are neural networks for training w(y,x,theta) (adapted from https://github.com/jaychan-hep/UnbinnedProfiledUnfolding)


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
    def __init__(self, nodes=[50, 50, 50], sigmoid=True, std_params=None, n_inputs=4, dropouts=[0, 0.1, 0, 0], activation=nn.ReLU()):
        super(wRT_network, self).__init__()
        
        self.std_params = std_params
        self.sigmoid = sigmoid
        self.linear_stack = nn.Sequential()
        self.n_inputs = n_inputs
        for i in range(len(nodes)+1):
            self.linear_stack.add_module(f"linear_{i}", nn.Linear(self.n_inputs if i==0 else nodes[i-1], nodes[i] if i<len(nodes) else 1))
            if i<len(nodes):
                self.linear_stack.add_module(f"activation_{i}", activation)
                self.linear_stack.add_module(f"batchNorm_{i}", nn.BatchNorm1d(nodes[i]))
            if dropouts[i] > 0: self.linear_stack.add_module(f"dropout_{i}", nn.Dropout(p=dropouts[i]))
        if self.sigmoid: self.linear_stack.add_module("sigmoid", nn.Sigmoid())
        
    def forward(self, T, R, theta):
        #T = (T-self.std_params[0])/self.std_params[1]
        #R = (R-self.std_params[2])/self.std_params[3]
        x = torch.cat([R, T, theta], dim=1)
        logit = self.linear_stack(x)
        if self.sigmoid:
            return logit / (1 - logit + 0.00000000000001)
        else:
            return torch.exp(logit)

class wT_network(wRT_network):
    def __init__(self, n_inputs=2, *args, **kwargs):
        super(wT_network, self).__init__(n_inputs=n_inputs, *args, **kwargs)
        
    def forward(self, T, R, theta):
        #T = (T-self.std_params[0])/self.std_params[1]
        x = torch.cat([T, theta], dim=1)
        logit = self.linear_stack(x)
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
    def __init__(self, train_dataloader, val_dataloader, model_w, loss_fn, optimizer, max_epoch=1000, patience=10, wandb=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_w = model_w
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.patience = patience
        self.wandb = wandb
        
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
            
            if self.wandb:
                self.wandb.log({'train_loss': train_loss, 'train_auc': train_auc,
               'val_loss': val_loss, 'val_auc': val_auc})

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
                
def test_w(test_dataloader, model_wRT, model_wT, theta, dvc=dvc):
       
    # now evaluate performance at the epoch end
    model_wRT.eval()
    if model_wT is not None:
        model_wT.eval()

    Ts, Rs, Ws = None, None, None
    with torch.no_grad():
        for batch, (T, R) in enumerate(test_dataloader):
            T, R = T.to(dvc), R.to(dvc)
            ones = torch.ones(len(R), 1).to(dvc)

            # Compute weights
            if model_wT is not None:
                W = model_wRT(T, R, ones*theta)/model_wT(T, R, ones*theta)
            else:
                W = model_wRT(T, R, ones*theta)
            Ts = torch.cat([Ts, T]) if Ts is not None else T
            Rs = torch.cat([Rs, R]) if Rs is not None else R
            Ws = torch.cat([Ws, W]) if Ws is not None else W
    return Ts, Rs, Ws


def test_w_different_thetas(test_dataloader, model_wRT, model_wT, thetas, dvc=dvc):
    n = len(thetas)

    model_wRT.eval()
    if model_wT is not None:
        model_wT.eval()

    Ts, Rs, Ws = None, None, None
    with torch.no_grad():
        for batch, (T, R) in enumerate(test_dataloader):
            T, R = T.to(dvc), R.to(dvc)
            print(T.shape)
            ones = torch.ones(len(R), 1).to(dvc)

            # Compute weights 
            if model_wT is not None:
                W = model_wRT(T.repeat(n,1), R.repeat(n,1), thetas.repeat_interleave(len(R)))/model_wT(T, R, thetas.repeat_interleave(len(R)))
            else:
                W = model_wRT(T.repeat(n,1), R.repeat(n,1), thetas.repeat_interleave(len(R)))
            Ts = torch.cat([Ts, T]) if Ts is not None else T
            Rs = torch.cat([Rs, R]) if Rs is not None else R
            Ws = torch.cat([Ws, W]) if Ws is not None else W
    if n > 1:
        Ws = Ws.view(n, -1)
    return Ts, Rs, Ws


def make_w_theta(test_dataloader, model_wRT, model_wT=None):
    def w_theta(theta):
        _, _, Ws = test_w(test_dataloader, model_wRT, model_wT, theta)
        return Ws
    return w_theta

def make_w_theta_list(test_dataloader, model_wRT_ensemble, model_wT_ensemble):
    w_list = []
    for i in range(len(model_wRT_ensemble)):
        w_list.append(make_w_theta(test_dataloader, model_wRT_ensemble[i], model_wT_ensemble[i]))
    return w_list

def make_w_theta_ensemble(test_dataloader, model_wRT_ensemble, model_wT_ensemble, func='mean'):
    w_theta_list = make_w_theta_list(test_dataloader, model_wRT_ensemble, model_wT_ensemble)
    def w_theta(theta):
        Ws = []
        for w_theta in w_theta_list:
            w = w_theta(theta)
            if (isinstance(w, torch.Tensor)):
                w = w.cpu().numpy().flatten()
            Ws.append(w)
        Ws = np.array(Ws)
        if func == 'mean':
            return np.mean(Ws, axis=0)
        elif func == 'median':
            return np.median(Ws, axis=0)
        else:
            return Ws
    return w_theta
