"""
Functions performing profile omnifold without using neural network.
"""
import numpy as np
from scipy.interpolate import splrep, BSpline
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.nonparametric.kernel_regression import KernelReg
import pandas as pd
from random import choices

def density_ratio_classifier(X, labels, X_eval=None, max_depth=5, train_ratio=0.6):
    """
    Estimate the density ratio by fitting a classifier using random forest.
    """
    train_idx = np.random.choice(X.shape[0], size=int(train_ratio*X.shape[0]), replace=False)
    X_train = X[train_idx,:]
    labels_train = labels[train_idx]
    clf = RandomForestClassifier(max_depth=max_depth)
    clf.fit(X_train, labels_train)
    if X_eval is None:
        X_eval = X[labels==1,:]
    pred_prob = clf.predict_proba(X_eval)[:,0]
    #epsilon = 0.0000001
    #pred_prob = np.clip(pred_prob, epsilon, 1-epsilon)
    ratio = pred_prob/(1-pred_prob)
    ratio[pred_prob==1] = 1
    return ratio


def omnifold(y, x_mc, y_mc, niter, save_iter=False):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm).

    Parameters:
    -----------
    y : 2darray
        The experimental detector-level data (row = events, column = features).        
    x_mc : 2darray
        The Monte Carlo particle-level events (row = events, column = features).     
    y_mc : 2darray
        The Monte Carlo detector-level events (row = events, column = features).
    niter : int
        The number of iterations.
    save_iter : bool, optional, default=False
        If True, saves the weights at each iteration for further analysis or diagnostics.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, 
        return iter_log[niter, 2, x_mc.shape[0]] which contains the computed weights during the iteration.
        1st coordinate is the iteration step
        2nd coordinate is the list of weights on detector and particle level in each iteration [wy, wx]
        3rd coordinate is the weights evaluated on each Monte Carlo event
    """
    wx = np.ones(x_mc.shape[0])
    if save_iter:
        iter_log = np.zeros((niter,2,x_mc.shape[0]))
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        wy = wx
        #X1 = np.concatenate((y, choices(y_mc, weights=wy, k=y_mc.shape[0]))).reshape(-1,1)
        X1 = np.concatenate((y,y_mc[choices(np.arange(y_mc.shape[0]), weights=wy, k=y_mc.shape[0]),:]))
        labels1 = np.concatenate((np.zeros(y.shape[0]), np.ones(y_mc.shape[0])))
        ry = density_ratio_classifier(X1, labels1, X_eval=y_mc)
        print("updated ratio of y_mc/y_exp:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        #X2 = np.concatenate((choices(x_mc, weights=ry, k=x_mc.shape[0]), x_mc)).reshape(-1,1)
        X2 = np.concatenate((x_mc[choices(np.arange(x_mc.shape[0]), weights=ry, k=x_mc.shape[0]),:], x_mc))
        labels2 = np.concatenate((np.zeros(x_mc.shape[0]), np.ones(x_mc.shape[0])))
        wx = wx * density_ratio_classifier(X2, labels2, X_eval=x_mc)
        print("updated weight on x_mc:", wx, "\n")
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = wx
    if save_iter:
        return iter_log
    else:
        return wx

    
def omnifold_reg(y, x_mc, y_mc, niter, save_iter=False):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) using the regression.

    Parameters:
    -----------
    y : 2darray
        The experimental detector-level data (row = events, column = features).        
    x_mc : 2darray
        The Monte Carlo particle-level events (row = events, column = features).     
    y_mc : 2darray
        The Monte Carlo detector-level events (row = events, column = features).
    niter : int
        The number of iterations.
    save_iter : bool, optional, default=False
        If True, saves the weights at each iteration for further analysis or diagnostics.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, 
        return iter_log[niter, 2, x_mc.shape[0]] which contains the computed weights during the iteration.
        1st coordinate is the iteration step
        2nd coordinate is the list of weights on detector and particle level in each iteration [wy, wx]
        3rd coordinate is the weights evaluated on each Monte Carlo event
    """
    wx = np.ones(x_mc.shape[0])
    if save_iter:
        iter_log = np.zeros((niter,2,x_mc.shape[0]))
    X = np.concatenate((y, y_mc)).reshape(-1,1)
    labels = np.concatenate((np.zeros(y.shape[0]), np.ones(y_mc.shape[0])))
    py_qy_ratio = density_ratio_classifier(X, labels)
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        regr1 = GradientBoostingRegressor()
        regr1.fit(y_mc.reshape(-1,1), wx)
        wy = regr1.predict(y_mc.reshape(-1,1))
        ry = py_qy_ratio/wy
        print("updated ratio of y_mc/y_exp:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        regr2 = GradientBoostingRegressor()
        regr2.fit(x_mc.reshape(-1,1), ry)
        wx = wx * regr2.predict(x_mc.reshape(-1,1))
        print("updated weight on x_mc:",wx,"\n")
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = wx
    if save_iter:
        return iter_log
    else:
        return wx



def profile_omnifold_known_nuisance(y, x_mc, y_mc, w, niter, save_iter=False):
    """
    Profile omnifold algorithm in the presence of known nuisance parameter. 
    That is, we know the true reweighting function on the response kernel.

    Parameters:
    -----------
    y : 2darray
        The experimental detector-level data (row = events, column = features).        
    x_mc : 2darray
        The Monte Carlo particle-level events (row = events, column = features).     
    y_mc : 2darray
        The Monte Carlo detector-level events (row = events, column = features).
    w : 2darray
        The true reweighting factors on the response kernel evaluated at MC events (row = events, column = features)
    niter : int
        The number of iterations.
    save_iter : bool, optional, default=False
        If True, saves the weights at each iteration for further analysis or diagnostics.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, 
        return iter_log[niter, 2, x_mc.shape[0]] which contains the computed weights during the iteration.
        1st coordinate is the iteration step
        2nd coordinate is the list of weights on detector and particle level in each iteration [wy, wx]
        3rd coordinate is the weights evaluated on each Monte Carlo event
    """
    
    # omnifold
    wx = np.ones(len(x_mc))
    if save_iter:
        iter_log = np.zeros((niter,2,x_mc.shape[0]))
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        wy = wx * w
        X1 = np.concatenate((y,y_mc[choices(np.arange(y_mc.shape[0]), weights=wy, k=y_mc.shape[0]),:]))
        labels1 = np.concatenate((np.zeros(y.shape[0]), np.ones(y_mc.shape[0])))
        ry = density_ratio_classifier(X1, labels1, X_eval=y_mc)
        print("updated ratio of y_mc/y_exp:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        X2 = np.concatenate((x_mc[choices(np.arange(x_mc.shape[0]), weights=ry*w, k=x_mc.shape[0]),:], x_mc))
        labels2 = np.concatenate((np.zeros(x_mc.shape[0]), np.ones(x_mc.shape[0])))
        wx = wx * density_ratio_classifier(X2, labels2, X_eval=x_mc)
        print("updated weight on x_mc:", wx, "\n")
        
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = wx
    if save_iter:
        return iter_log
    else:
        return wx


def profile_omnifold_reg_known_nuisance(y, x_mc, y_mc, w, niter, save_iter=False):
    """
    Profile omnifold algorithm using regression in the presence of known
    nuisance parameter. That is, we know the true reweighting function on the response kernel

    Parameters:
    -----------
    y : 2darray
        The experimental detector-level data (row = events, column = features).        
    x_mc : 2darray
        The Monte Carlo particle-level events (row = events, column = features).     
    y_mc : 2darray
        The Monte Carlo detector-level events (row = events, column = features).
    w : 2darray
        The true reweighting factors on the response kernel evaluated at MC events (row = events, column = features)
    niter : int
        The number of iterations.
    save_iter : bool, optional, default=False
        If True, saves the weights at each iteration for further analysis or diagnostics.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, 
        return iter_log[niter, 2, x_mc.shape[0]] which contains the computed weights during the iteration.
        1st coordinate is the iteration step
        2nd coordinate is the list of weights on detector and particle level in each iteration [wy, wx]
        3rd coordinate is the weights evaluated on each Monte Carlo event
    """
    
    # omnifold with regression
    wx = np.ones(len(x_mc))
    if save_iter:
        iter_log = np.zeros((niter,2,x_mc.shape[0]))
    X = np.concatenate((y, y_mc))
    labels = np.concatenate((np.zeros(y.shape[0]), np.ones(y_mc.shape[0])))
    py_qy_ratio = density_ratio_classifier(X, labels)
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        regr1 = GradientBoostingRegressor()
        regr1.fit(y_mc, wx*w)
        wy = regr1.predict(y_mc)
        ry = py_qy_ratio/wy
        print("updated ratio of y_mc/y_exp:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        regr2 = GradientBoostingRegressor()
        regr2.fit(x_mc, ry*w)
        wx = wx * regr2.predict(x_mc)
        print("updated weight on x_mc:",wx,"\n")
        
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = wx
    if save_iter:
        return iter_log
    else:
        return wx


def nonparametric_profile_omnifold(y, x_mc, y_mc, niter, save_iter=False):
    """
    Nonparametric profile omnifold algorithm in the presence of nuisance parameter.

    Parameters:
    -----------
    y : 2darray
        The experimental detector-level data (row = events, column = features).        
    x_mc : 2darray
        The Monte Carlo particle-level events (row = events, column = features).     
    y_mc : 2darray
        The Monte Carlo detector-level events (row = events, column = features).
    niter : int
        The number of iterations.
    save_iter : bool, optional, default=False
        If True, saves the weights at each iteration for further analysis or diagnostics.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, 
        return iter_log[niter, 3, x_mc.shape[0]] which contains the computed weights during the iteration.
        1st coordinate is the iteration step
        2nd coordinate is the list of weights on detector and particle level and updated reweighting function w(y,x) in each iteration [wy, wx, w]
        3rd coordinate is the weights evaluated on each Monte Carlo event
    """
    
    nu = np.ones(x_mc.shape[0])
    w = np.ones(x_mc.shape[0])
    if save_iter:
        iter_log = np.zeros((niter,3,x_mc.shape[0]))

    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        wy = nu*w
        X1 = np.concatenate((y, y_mc[choices(np.arange(y_mc.shape[0]), weights=wy, k=y_mc.shape[0]),:]))
        labels1 = np.concatenate((np.zeros(y.shape[0]), np.ones(y_mc.shape[0])))
        ry = density_ratio_classifier(X1, labels1, X_eval=y_mc)
        print("updated ratio of y_mc/y_exp:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        X2 = np.concatenate((x_mc[choices(np.arange(x_mc.shape[0]), weights=ry*w, k=x_mc.shape[0]),:], x_mc))
        labels2 = np.concatenate((np.zeros(x_mc.shape[0]), np.ones(x_mc.shape[0])))
        nunext = nu * density_ratio_classifier(X2, labels2, X_eval=x_mc)
        print("updated weight on x_mc:", nunext, "\n")

        w = w * nu/nunext * ry
        print("updated weight on p_mc(y|x):", w, "\n")
        nu = nunext
        
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = nu
            iter_log[t,2,:] = w
        
    if save_iter:
        return iter_log
    else:
        return nu



def ad_hoc_penalized_profile_omnifold(y, x_mc, y_mc, theta_bar, theta0, w_func, niter, no_penalty=False, save_iter=False):
    """
    Profile omnifold algorithm in the presence of nuisance parameter with penalization step
    derived from nonparametric profile omnifold.

    Parameters:
    -----------
    y : 2darray
        The experimental detector-level data (row = events, column = features).        
    x_mc : 2darray
        The Monte Carlo particle-level events (row = events, column = features).     
    y_mc : 2darray
        The Monte Carlo detector-level events (row = events, column = features).
    theta_bar : float
        nuisance parameter for the Monte Carlo dataset      
    theta0 : float
        Initial value for the nuisance parameter
    w_func : callable
        A function that computes the reweighting factors on the MC dataset given an input theta
    niter : int
        The number of iterations.
    save_iter : bool, optional, default=False
        If True, saves the weights at each iteration for further analysis or diagnostics.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, 
        return iter_log[niter, 3, x_mc.shape[0]] which contains the computed weights during the iteration.
        1st coordinate is the iteration step
        2nd coordinate is the list of weights on detector and particle level and updated theta in each iteration [wy, wx, theta]
        3rd coordinate is the weights evaluated on each Monte Carlo event
    """
    
    nu = np.ones(x_mc.shape[0])
    theta = theta0
    if save_iter:
        iter_log = np.zeros((niter,4,x_mc.shape[0]))

    for t in range(niter):
        w = w_func(theta)

        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        wy = nu*w
        X1 = np.concatenate((y, y_mc[choices(np.arange(y_mc.shape[0]), weights=wy, k=y_mc.shape[0]),:]))
        labels1 = np.concatenate((np.zeros(y.shape[0]), np.ones(y_mc.shape[0])))
        ry = density_ratio_classifier(X1, labels1, X_eval=y_mc)
        print("updated ratio of y_mc/y_exp:", ry, "\n")

        print("Fitting pull-back weights on x_mc...")
        X2 = np.concatenate((x_mc[choices(np.arange(x_mc.shape[0]), weights=ry*w, k=x_mc.shape[0]),:], x_mc))
        labels2 = np.concatenate((np.zeros(x_mc.shape[0]), np.ones(x_mc.shape[0])))
        nunext = nu * density_ratio_classifier(X2, labels2, X_eval=x_mc)
        print("updated weight on x_mc:", nunext, "\n")
        print(f'Value of Theta before update: {theta}')
        
        def theta_loss(x):
            w_theta = w_func(x[0])
            if no_penalty == False:
                return np.mean((w_theta - w * nu / nunext * ry)**2) + (x[0]-theta_bar)**2/2
            else:
                return np.mean((w_theta - w * nu / nunext * ry)**2)
        
        solution = optimize.minimize(fun=theta_loss, x0=theta, bounds=[(0,3)])
        theta = solution.x[0]

        print(f'Updated theta: {theta}')
        nu = nunext
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = nu
            iter_log[t,2,:] = w_func(theta)
            iter_log[t,3,:] = theta
        
    if save_iter:
        return iter_log
    else:
        return nu

    
def penalized_profile_omnifold(y, x_mc, y_mc, theta_bar, theta0, w_func, w_func_derivative, niter, no_penalty=False, save_iter=False):
    """
    Profile omnifold algorithm in the presence of nuisance parameter.

    Parameters:
    -----------
    y : 2darray
        The experimental detector-level data (row = events, column = features).        
    x_mc : 2darray
        The Monte Carlo particle-level events (row = events, column = features).     
    y_mc : 2darray
        The Monte Carlo detector-level events (row = events, column = features).
    theta_bar : float
        nuisance parameter for the Monte Carlo dataset      
    theta0 : float
        Initial value for the nuisance parameter
    w_func : callable
        A function that computes the reweighting factors on the MC dataset given an input theta
    w_func_derivative : callable
        The derivative function of w_func with respect to theta
    niter : int
        The number of iterations.
    save_iter : bool, optional, default=False
        If True, saves the weights at each iteration for further analysis or diagnostics.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, 
        return iter_log[niter, 3, x_mc.shape[0]] which contains the computed weights during the iteration.
        1st coordinate is the iteration step
        2nd coordinate is the list of weights on detector and particle level and updated theta in each iteration [wy, wx, theta]
        3rd coordinate is the weights evaluated on each Monte Carlo event
    """
    nu = np.ones(x_mc.shape[0])
    theta = theta0
    if save_iter:
        iter_log = np.zeros((niter,4,x_mc.shape[0]))

    for t in range(niter):
        w = w_func(theta)

        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        wy = nu*w
        X1 = np.concatenate((y, y_mc[choices(np.arange(y_mc.shape[0]), weights=wy, k=y_mc.shape[0]),:]))
        labels1 = np.concatenate((np.zeros(y.shape[0]), np.ones(y_mc.shape[0])))
        ry = density_ratio_classifier(X1, labels1, X_eval=y_mc)
        print("updated ratio of y_mc/y_exp:", ry, "\n")
        
        print(f'Value of Theta before update: {theta}')
        def theta_func(x):
            w_next = w_func(x)
            delta_w_next = w_func_derivative(x)
            if no_penalty == False:
                return x - theta_bar - np.mean(w*nu*delta_w_next/w_next*ry)
            else:
                return np.mean(w*nu*delta_w_next/w_next*ry)
            
        solution = optimize.root_scalar(theta_func, bracket = [0,3], method='bisect')
        theta = solution.root
        
        #def theta_loss(x):
        #    w_next = w_func(x[0])
        #    delta_w_next = w_func_derivative(x[0])
        #    return (x[0] - theta_bar - np.mean(w*nu*delta_w_next/w_next*ry))**2
        #solution = optimize.minimize(theta_loss, theta)
        #theta = solution.x[0]
        #if t == 5:
        #    return theta_func

        print(f'Updated theta: {theta}')
        w_next = w_func(theta)

        print("Fitting pull-back weights on x_mc...")
        X2 = np.concatenate((x_mc[choices(np.arange(x_mc.shape[0]), weights=ry*w, k=x_mc.shape[0]),:], x_mc))
        X3 = np.concatenate((x_mc[choices(np.arange(x_mc.shape[0]), weights=w_next, k=x_mc.shape[0]),:], x_mc))
        labels2 = np.concatenate((np.zeros(x_mc.shape[0]), np.ones(x_mc.shape[0])))
        nu = nu * density_ratio_classifier(X2, labels2, X_eval=x_mc) * 2 / (1+density_ratio_classifier(X3, labels2, X_eval=x_mc))
        print("updated weight on x_mc:", nu, "\n")
        
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = nu
            iter_log[t,2,:] = w
            iter_log[t,3,:] = theta
        
    if save_iter:
        return iter_log
    else:
        return nu