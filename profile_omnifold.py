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
from torch.func import vmap, grad
import gc
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from scipy import optimize
from scipy import stats
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
    weights[np.isinf(weights)] = 1
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

def density_ratio_classifier(xvals, yvals, weights, xeval, model, epochs=10, verbose=0):
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
        The number of epochs to train the classifier.
    verbose : int, optional
        The verbosity level of the classifier.

    Returns
    -------
    weights : array-like, shape=(n_eval_samples,)
        The density ratio evaluated at xeval.

    """

    X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(xvals, yvals, weights)

    # zip ("hide") the weights with the labels
    Y_train = np.stack((Y_train, w_train), axis=1)
    Y_test = np.stack((Y_test, w_test), axis=1)   


    model.compile(loss=weighted_binary_crossentropy,
                  optimizer='Adam',
                  metrics=['accuracy'])
    model.fit(X_train,
              Y_train,
              epochs=epochs, # more epochs, learning curve
              batch_size=10000,
              validation_data=(X_test, Y_test),
              verbose=verbose)
    return reweight(xeval,model)


def omnifold(y,x_mc,y_mc,iterations,verbose=0):
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
    verbose : int, optional
        The verbosity level of the classifier.

    Returns
    -------
    weights : array-like, shape=(iterations, 2, n_samples)
        The weights for each iteration and step.
    """

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


        print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        print("STEP 1\n")

        weights_1 = np.concatenate((weights_push, np.ones(len(y))))

        weights_pull = weights_push * density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose=verbose)
        weights[i, :1, :] = weights_pull

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        print("\nSTEP 2\n")

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose=verbose)
        weights[i, 1:2, :] = weights_push

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

        print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        print("STEP 1\n")

        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))

        weights_pull = weights_push * density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose=verbose)
        weights[i, :1, :] = weights_pull


        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        print("\nSTEP 2\n")

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose)
        weights[i, 1:2, :] = weights_push


        # STEP 3: update w
        print("\nSTEP 3\n")
        print(f'Update w...')
        w =  w * weights_pull / weights_push
        weights[i, 2:3, :] = w
        
    return weights



def profile_omnifold(y, x_mc, y_mc, iterations, w_theta, w_theta_grad, theta_bar, theta0, no_penalty=False, theta_range=None,                                          first_order=False, lr=0.01, verbose=0):

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
        w = w_theta(theta)
        if (isinstance(w, torch.Tensor)):
            w = w.cpu().numpy().flatten()


        print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data


        print("STEP 1\n")


        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))

        ry = density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose=verbose)
        weights_pull = weights_push * ry
        weights[i, 0, :] = weights_pull
        
        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        print("\nSTEP 2\n")


        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        
        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose=verbose)
        
        #weights_3 = np.concatenate((np.ones(len(x_mc)), w_next))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.
        
        #weights_push = weights_push * (1+lambda2) / (1 + lambda1*density_ratio_classifier(xvals_2, yvals_2, weights_3, x_mc, model2, verbose=verbose))
        weights[i, 1, :] = weights_push
        
        # STEP 3: update theta
        print("\nSTEP 3\n")
        print(f'Value of Theta before update: {theta}')
        
        
        if first_order == False:
            # direct optimization
            def theta_func(x):
                w_next = w_theta(x)
                delta_w_next = w_theta_grad(x)
                if (isinstance(w_next, torch.Tensor)):
                    w_next = w_next.cpu().numpy().flatten()
                    delta_w_next = delta_w_next.detach().cpu().numpy().flatten()
                if no_penalty:
                    return np.mean(w*weights_push*delta_w_next/w_next*ry)
                else:
                    return x - theta_bar - np.mean(w*weights_push*delta_w_next/w_next*ry)
            if theta_range is not None:
                solution = optimize.root_scalar(theta_func, bracket = theta_range, method='bisect')
            else:
                solution = optimize.root_scalar(theta_func, x0=theta, method='secant')
            theta = solution.root
        else:
            # first-order update
            w_next = w_theta(theta)
            delta_w_next = w_theta_grad(theta)
            if (isinstance(w_next, torch.Tensor)):
                    w_next = w_next.cpu().numpy().flatten()
                    delta_w_next = delta_w_next.detach().cpu().numpy().flatten()
            if no_penalty:
                theta = theta + lr * np.mean(w*weights_push*delta_w_next/w_next*ry)
            else:
                theta = theta + lr * (np.mean(w*weights_push*delta_w_next/w_next*ry) - (theta - theta_bar))
        

        print(f'Updated value of Theta: {theta}')
        w_next = w_theta(theta)
        if (isinstance(w_next, torch.Tensor)):
            w_next = w_next.cpu().numpy().flatten()
        weights[i, 2, :] = w_next
        weights[i, 3, :] = theta
        
    return weights


def profile_omnifold_no_grad(y, x_mc, y_mc, iterations, w_theta, theta_bar, theta0, theta_range, num_grid_points=20, no_penalty=False, 
                             x_range=None, y_range=None, verbose=0, return_Q=False):
    
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
    
    print(f'Initial Theta: {theta}')

    for i in range(iterations):

        print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        print("STEP 1\n")

        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))

        ry = density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose=verbose)
        weights_pull = weights_push * ry
        weights[i, 0, :] = weights_pull
        
        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        print("\nSTEP 2\n")

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.
        
        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose=verbose)
        weights[i, 1, :] = weights_push
        
        # STEP 3: update theta
        print("\nSTEP 3\n")
        print(f'Value of Theta before update: {theta}')

        def Q(x):
            #w_next = w_theta(x)
            #if (isinstance(w_next, torch.Tensor)):
            #        w_next = w_next.cpu().numpy().flatten()
            w_next = w_grid[np.argmin(np.abs(theta_values - x))]

            if i == 0:
                Q_out = np.mean(w*ry*np.log(w_next*weights_push))
            else:
                Q_out = np.mean(w*weights[i-1, 1, :]*ry*np.log(w_next*weights_push))
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
        #w = w_theta(theta)
        #if (isinstance(w, torch.Tensor)):
        #    w = w.cpu().numpy().flatten()
        weights[i, 2, :] = w
        weights[i, 3, :] = theta

    if return_Q:
        return weights, theta_values, Q_iter
    else:
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

        print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data


        print("STEP 1\n")

        weights_1 = np.concatenate((weights_push * w, np.ones(len(y))))

        weights_pull = weights_push * density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose=verbose)
        weights[i, :1, :] = weights_pull


        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        print("\nSTEP 2\n")

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose=verbose)
        weights[i, 1:2, :] = weights_push

    return weights


def ad_hoc_penalized_profile_omnifold(y, x_mc, y_mc, iterations, w_theta, theta_bar, theta0, no_penalty=False, verbose=0):

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
        w = w_theta(theta)
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


        weights_pull = weights_push * density_ratio_classifier(xvals_1, yvals_1, weights_1, y_mc, model1, verbose=verbose)
        weights[i, 0, :] = weights_pull


        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(x_mc)), weights_pull * w))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.


        weights_push = density_ratio_classifier(xvals_2, yvals_2, weights_2, x_mc, model2, verbose=verbose)
        weights[i, 1, :] = weights_push

        
        # STEP 3: update theta
        if (verbose>0):
            print("\nSTEP 3\n")
            print(f'Value of Theta before update: {theta}')
            pass
        
        def theta_loss(x):
            w = w_theta(x[0])
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
        w = w_theta(theta)
        if (isinstance(w, torch.Tensor)):
            w = w.cpu().numpy().flatten()
        weights[i, 2, :] = w
        weights[i, 3, :] = theta
        
    return weights

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


def test_w_derivative(test_dataloader, model_wRT, model_wT, theta, dvc=dvc):
    # Ensure theta requires gradients
    theta = torch.tensor([theta], dtype=torch.float32, requires_grad=True, device=dvc)

    # Put models in evaluation mode
    model_wRT.eval()
    model_wT.eval()

    # Variables to hold data
    Ts, Rs, Ws, grad_Ws_theta = None, None, None, None

    for batch, (T, R) in enumerate(test_dataloader):
        T, R = T.to(dvc), R.to(dvc)
        ones = torch.ones(len(R), 1).to(dvc)

        # Compute weights (with theta requiring gradients)
        W = model_wRT(T, R, ones * theta) / model_wT(T, R, ones * theta)
        # Compute the gradient of W w.r.t theta for the entire batch
        grad_W_theta = torch.autograd.grad(outputs=W, inputs=theta, 
                                           grad_outputs=torch.ones_like(W))[0]
                                           #create_graph=False, retain_graph=False)
        #print(grad_W_theta.shape)

        # Accumulate the data for Ts, Rs, Ws, and gradient of Ws (wrt theta)
        Ts = torch.cat([Ts, T]) if Ts is not None else T
        Rs = torch.cat([Rs, R]) if Rs is not None else R
        Ws = torch.cat([Ws, W]) if Ws is not None else W
        grad_Ws_theta = torch.cat([grad_Ws_theta, grad_W_theta]) if grad_Ws_theta is not None else grad_W_theta

    return Ts, Rs, Ws, grad_Ws_theta

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


def make_w_theta_grad(test_dataloader, model_wRT, model_wT, dvc=dvc):
    def w_func(theta, T, R):
        wRT_val = model_wRT(T.reshape(1,-1), R.reshape(1,-1), theta.reshape(1,-1))
        wT_val = model_wT(T.reshape(1,-1), R.reshape(1,-1), theta.reshape(1,-1))
        return (wRT_val / wT_val).squeeze()
    w_func_grad = grad(w_func)
    compute_w_func_batch_grad = vmap(w_func_grad, in_dims=(None, 0, 0))

    def w_theta_grad(theta):
        grads = []
        for batch, (T, R) in enumerate(test_dataloader):
            gc.collect()
            torch.cuda.empty_cache()
            T, R = T.to(dvc), R.to(dvc)
            w_func_batch_grad = compute_w_func_batch_grad(torch.tensor([theta],dtype=torch.float).to(dvc), T, R)
            grads.append(w_func_batch_grad.cpu().detach())
        return torch.cat(grads, dim=0).squeeze()

    return w_theta_grad


#def make_w_theta_derivative(test_dataloader, model_wRT, model_wT):
#    def w_theta_derivative(theta):
#        _, _, Ws = test_w_derivative(test_dataloader, model_wRT, model_wT, theta)
#        return Ws
#    return w_theta_derivative

#def make_w_theta(test_dataloader, model_wRT, model_wT, dvc=dvc):
#    def w_func(theta, T, R):
#        wRT_val = model_wRT(T.reshape(1,-1), R.reshape(1,-1), theta.reshape(1,-1))
#        wT_val = model_wT(T.reshape(1,-1), R.reshape(1,-1), theta.reshape(1,-1))
#        return (wRT_val / wT_val).squeeze()
#    w_func_grad = grad(w_func)
#    compute_w_func_batch = vmap(w_func, in_dims=(None, 0, 0))
#    compute_w_func_batch_grad = vmap(w_func_grad, in_dims=(None, 0, 0))
        
#    def w_theta(theta):
#        results = []
#        for batch, (T, R) in enumerate(test_dataloader):
#            gc.collect()
#            torch.cuda.empty_cache()
#            T, R = T.to(dvc), R.to(dvc)
#            w_func_batch = compute_w_func_batch(torch.tensor([theta],dtype=torch.float).to(dvc), T, R)
#            results.append(w_func_batch.cpu().detach())
#        return torch.cat(results, dim=0).squeeze()
#    def w_theta_grad(theta):
#        grads = []
#        for batch, (T, R) in enumerate(test_dataloader):
#            gc.collect()
#            torch.cuda.empty_cache()
#            T, R = T.to(dvc), R.to(dvc)
#            w_func_batch_grad = compute_w_func_batch_grad(torch.tensor([theta],dtype=torch.float).to(dvc), T, R)
#            grads.append(w_func_batch_grad.cpu().detach())
#        return torch.cat(grads, dim=0).squeeze()

#    return w_theta, w_theta_grad


#def make_w_theta_true_w_func(test_dataloader, model_wRT, model_wT, w_func):
#    def w_func_reorder(theta, T, R):
#        return w_func(R.reshape(1,-1),T.reshape(1,-1),theta.reshape(1,-1)).squeeze()
#    w_func_grad = grad(w_func_reorder)
#    compute_w_func_batch = vmap(w_func_reorder, in_dims=(None, 0, 0))
#    compute_w_func_batch_grad = vmap(w_func_grad, in_dims=(None, 0, 0))
    
    
#    Ts, Rs = None, None
#    for batch, (T, R) in enumerate(test_dataloader):
#        T, R = T.to(dvc), R.to(dvc)
#        Ts = torch.cat([Ts, T]) if Ts is not None else T
#        Rs = torch.cat([Rs, R]) if Rs is not None else R
        
#    def w_theta(theta):
#        gc.collect()
#        torch.cuda.empty_cache()
#        return compute_w_func_batch(torch.tensor([theta],dtype=torch.float).to(dvc), Ts, Rs).cpu().detach().squeeze()
#    def w_theta_grad(theta):
#        gc.collect()
#        torch.cuda.empty_cache()
#        return compute_w_func_batch_grad(torch.tensor([theta],dtype=torch.float).to(dvc), Ts, Rs).cpu().detach().squeeze()

    return w_theta, w_theta_grad