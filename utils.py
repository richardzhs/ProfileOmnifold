"""
functions for unfolding methods
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.linear_model import Ridge
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy import optimize
import scipy.integrate as integrate
from scipy.interpolate import splrep, BSpline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.nonparametric.kernel_regression import KernelReg
import pandas as pd
from random import choices
import tensorflow as tf
import torch
dvc = "cuda" if torch.cuda.is_available() else "cpu"


def rejection_sampling(n, f, lb, ub, scale=1):
    """
    Perform rejection sampling to sample from a given density with a bounded support.
    """
    def min_f(x):
        return -f(x)*scale
    M = -optimize.minimize(min_f, lb+(ub-lb)/2, bounds=[(lb,ub)]).fun/scale
    x = np.zeros(n)
    i = 0
    while i < n:
        u = np.random.uniform(0,1,n)
        v = np.random.uniform(lb,ub,n)
        accept_sample = u < f(v)/M
        nsample = np.sum(accept_sample)
        if i+nsample <= n:
            x[i:(i+nsample)] = v[accept_sample]
        else:
            x[i:n] = v[accept_sample][:(n-i)]
        i += nsample
    return x

def cholesky_trans(y, smear_means, K):
    # find the change in basis
    Sigma_data = np.diag(smear_means)
    L_data = np.linalg.cholesky(Sigma_data)
    L_data_inv = np.linalg.inv(L_data)
    # transform the matrix
    K_trans = L_data_inv @ K
    # transform the data
    y_trans = L_data_inv @ y
    return K_trans, y_trans

def generate_ls_point_estimators(y, smear_means, K):
    K_trans, y_trans = cholesky_trans(y, smear_means, K)
    return np.linalg.lstsq(K_trans, y_trans, rcond=-1)[0]
    #return sp.linalg.lstsq(K_trans, y_trans)[0]

def generate_ridge_point_estimators(y, smear_means, K, alpha=1.0):
    K_trans, y_trans = cholesky_trans(y, smear_means, K)
    ridgefit = Ridge(alpha=alpha, fit_intercept=False, solver='svd')
    ridgefit.fit(K_trans, y_trans)
    return ridgefit.coef_

def generate_svd_point_estimators(y, smear_means, K, alpha=1.0):
    K_trans, y_trans = cholesky_trans(y, smear_means, K)
    U, S, Vh = np.linalg.svd(K_trans, full_matrices=True)
    r = np.linalg.matrix_rank(K_trans)
    res = np.zeros(K.shape[1])
    for i in range(r):
        res += S[i]/(S[i]**2+alpha)*(np.dot(np.dot(np.transpose(U[:,i]),y_trans), Vh[i,:]))
    return res
    
def generate_EM_point_estimators_for_loop(y, K, niter):
    lambdat = np.full(K.shape[1], np.mean(y))
    for t in range(niter):
        lambdatplusone = np.zeros(len(lambdat))
        denom = np.zeros(len(y))
        for i in range(len(y)):
            denom[i] = np.sum(K.iloc[i,:]*lambdat)
        for j in range(len(lambdatplusone)):
            lambdatplusone[j] = lambdat[j]/np.sum(K.iloc[:,j])*np.sum(K.iloc[:,j]*y/denom)
        lambdat = lambdatplusone
    return lambdat

def generate_EM_point_estimators(y, K, niter):
    lambdat = np.full(K.shape[1], np.mean(y))
    for t in range(niter):
        norm_factor = np.sum(K, axis=0)
        denom = K @ lambdat
        lambdatplusone = lambdat / norm_factor * (np.transpose(K) @ (y / denom))
        lambdat = lambdatplusone
    return lambdat


def density_ratio_classifier(X, labels, X_eval=None, max_depth=5, train_ratio=0.6):
    """
    Estimate the density ratio using a classifier.
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
    Omnifold algorithm (i.e. unbinned EM algorithm) without using the regression

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, also iter_log[niter, 3, x_mc.shape[0]] which contains the 
    computed quantities during the iteration.
    1st coordinate is the iteration step
    2nd coordinate is the list of quantities [wy, ry, wx]
    3rd coordinate is the quantities evaluated on each Monte Carlo point
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

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc
    if save_iter == True, also iter_log[niter, 3, x_mc.shape[0]] which contains the 
    computed quantities during the iteration.
    1st coordinate is the iteration step
    2nd coordinate is the list of quantities [wy, wx]
    3rd coordinate is the quantities evaluated on each Monte Carlo point
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


def profile_omnifold_validation(y, x_mc, y_mc, x_val, y_val, niter):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) with access to a validation sample that shares the 
    same nuisance parameter as the experimental data.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    """
    # First, estimate the reweighting function for the response kernel
    X1 = np.transpose(np.vstack((np.concatenate((x_val, x_mc)), np.concatenate((y_val, y_mc)))))
    labels1 = np.concatenate((np.zeros(len(y_val)), np.ones(len(y_mc))))
    product_ratio = density_ratio_classifier(X1, labels1)
    X2 = np.concatenate((x_mc, x_val)).reshape(-1,1)
    labels2 = np.concatenate((np.zeros(len(x_mc)), np.ones(len(x_val))))
    marginal_ratio = density_ratio_classifier(X2, labels2)
    w = product_ratio * marginal_ratio
    print("Fitted weights on the response kernel:", w, "\n")
    
    # omnifold
    wx = np.ones(len(x_mc))
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        wy = wx * w
        X1 = np.concatenate((y, choices(y_mc, weights=wy, k=len(y_mc)))).reshape(-1,1)
        labels1 = np.concatenate((np.zeros(len(y)), np.ones(len(y_mc))))
        ry = density_ratio_classifier(X1, labels1, X_eval=y_mc.reshape(-1,1))
        print("updated ratio of y_mc/y_exp:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        X2 = np.concatenate((choices(x_mc, weights=ry, k=len(x_mc)), x_mc)).reshape(-1,1)
        labels2 = np.concatenate((np.zeros(len(x_mc)), np.ones(len(x_mc))))
        wx = wx * w * density_ratio_classifier(X2, labels2, X_eval=x_mc.reshape(-1,1))
        print("updated weight on x_mc:", wx, "\n")

    return wx


def profile_omnifold_reg_validation(y, x_mc, y_mc, x_val, y_val, niter):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) using regression with access to a validation sample that 
    shares the same nuisance parameter as the experimental data.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    """
    # First, estimate the reweighting function for the response kernel
    X1 = np.transpose(np.vstack((np.concatenate((x_val, x_mc)), np.concatenate((y_val, y_mc)))))
    labels1 = np.concatenate((np.zeros(len(y_val)), np.ones(len(y_mc))))
    product_ratio = density_ratio_classifier(X1, labels1)
    X2 = np.concatenate((x_mc, x_val)).reshape(-1,1)
    labels2 = np.concatenate((np.zeros(len(x_mc)), np.ones(len(x_val))))
    marginal_ratio = density_ratio_classifier(X2, labels2)
    w = product_ratio * marginal_ratio
    print("Fitted weights on the response kernel:", w, "\n")
    
    # omnifold with regression
    wx = np.ones(len(x_mc))
    X = np.concatenate((y, y_mc)).reshape(-1,1)
    labels = np.concatenate((np.zeros(len(y)), np.ones(len(y_mc))))
    py_qy_ratio = density_ratio_classifier(X, labels)
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        regr1 = GradientBoostingRegressor()
        regr1.fit(y_mc.reshape(-1,1), wx*w)
        wy = regr1.predict(y_mc.reshape(-1,1))
        ry = py_qy_ratio/wy
        print("updated ratio of y_mc/y_exp:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        regr2 = GradientBoostingRegressor()
        regr2.fit(x_mc.reshape(-1,1), ry*w)
        wx = wx * regr2.predict(x_mc.reshape(-1,1))
        print("updated weight on x_mc:",wx,"\n")
    return wx


def profile_omnifold_known_nuisance(y, x_mc, y_mc, w, niter, save_iter=False):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) in the presence of known
    nuisance parameter. That is, we know the true reweighting function on the response kernel.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
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
    Omnifold algorithm (i.e. unbinned EM algorithm) using regression in the presence of known
    nuisance parameter. That is, we know the true reweighting function on the response kernel

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
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
    Omnifold algorithm (i.e. unbinned EM algorithm) in the presence of nuisance parameter.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
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



def ad_hoc_penalized_profile_omnifold(y, x_mc, y_mc, theta_bar, theta0, w_func, niter, save_iter=False):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) in the presence of nuisance parameter with penalization step
    derived from nonparametric profile omnifold.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    """
    
    nu = np.ones(x_mc.shape[0])
    theta = theta0
    if save_iter:
        iter_log = np.zeros((niter,4,x_mc.shape[0]))

    for t in range(niter):
        w = w_func(theta)
        if (isinstance(w, torch.Tensor)):
            w = w.cpu().numpy().flatten()

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
            if (isinstance(w_theta, torch.Tensor)):
                w_theta = w_theta.cpu().numpy().flatten()
            return np.mean((w_theta - w * nu / nunext * ry)**2) + (x[0]-theta_bar)**2/2
        
        solution = optimize.minimize(theta_loss, theta)
        theta = solution.x[0]

        print(f'Updated theta: {theta}')
        nu = nunext
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = nu
            w_next = w_func(theta)
            if (isinstance(w_next, torch.Tensor)):
                w_next = w_next.cpu().numpy().flatten()
            iter_log[t,2,:] = w_next
            iter_log[t,3,:] = theta
        
    if save_iter:
        return iter_log
    else:
        return nu

    
def penalized_profile_omnifold(y, x_mc, y_mc, theta_bar, theta0, w_func, w_func_derivative, niter, save_iter=False):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) in the presence of nuisance parameter with penalization step
    derived from nonparametric profile omnifold.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    """
    nu = np.ones(x_mc.shape[0])
    theta = theta0
    if save_iter:
        iter_log = np.zeros((niter,4,x_mc.shape[0]))

    for t in range(niter):
        w = w_func(theta)
        if (isinstance(w, torch.Tensor)):
            w = w.cpu().numpy().flatten()

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
            if (isinstance(w_next, torch.Tensor)):
                w_next = w_next.cpu().numpy().flatten()
                delta_w_next = delta_w_next.cpu().numpy().flatten()
            return x - theta_bar - np.mean(w*nu*delta_w_next/w_next*ry)
        
        solution = optimize.root_scalar(theta_func, bracket = [0,3], method='bisect')
        theta = solution.root

        print(f'Updated theta: {theta}')
        w_next = w_func(theta)
        if (isinstance(w_next, torch.Tensor)):
            w_next = w_next.cpu().numpy().flatten()
        
        print("Fitting pull-back weights on x_mc...")
        X2 = np.concatenate((x_mc[choices(np.arange(x_mc.shape[0]), weights=ry*w, k=x_mc.shape[0]),:], x_mc))
        X3 = np.concatenate((x_mc[choices(np.arange(x_mc.shape[0]), weights=w_next, k=x_mc.shape[0]),:], x_mc))
        labels2 = np.concatenate((np.zeros(x_mc.shape[0]), np.ones(x_mc.shape[0])))
        nu = nu * density_ratio_classifier(X2, labels2, X_eval=x_mc) * 2 / (1+density_ratio_classifier(X3, labels2, X_eval=x_mc))
        print("updated weight on x_mc:", nu, "\n")
        
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = nu
            iter_log[t,2,:] = w_next
            iter_log[t,3,:] = theta
        
    if save_iter:
        return iter_log
    else:
        return nu


# Visualization

class comparison_plots_with_ratio:
    
    def __init__(self, xmin, xmax, nbins, xlabel=r"$T$", ratio_label="Data/Pred.", header="Gaussian Example", density=True, save_name=None, legend_corner="upper left"):
        
        self.xmin = xmin
        self.xmax = xmax
        self.nbins = nbins
        self.density = density
        self.save_name = save_name
        self.legend_corner = legend_corner
        self.nTt = None
        self.bTt = None
        self.nTs = []
        self.bTs = []
        self.argss = []
        
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        
        self.ax0 = plt.subplot(gs[0])
        self.ax0.yaxis.set_ticks_position('both')
        self.ax0.xaxis.set_ticks_position('both')
        self.ax0.tick_params(direction="in", which="both")
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=20)
        self.ax0.minorticks_on()
        
        plt.ylabel("Normalized to Unity" if self.density else "Events", fontsize=20)
        plt.xlim([xmin,xmax])
        plt.title(header,loc="right",fontsize=20, fontstyle="italic")
        
        self.ax1 = plt.subplot(gs[1])
        self.ax1.yaxis.set_ticks_position('both')
        self.ax1.xaxis.set_ticks_position('both')
        self.ax1.tick_params(direction="in",which="both")
        self.ax1.minorticks_on()
        
        plt.xlim([xmin,xmax])
        plt.locator_params(axis='x', nbins=6)
        plt.xlabel(xlabel,fontsize=20)
        plt.ylabel(ratio_label,fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=15)
        plt.axhline(y=1,linewidth=2, color='gray')   
        plt.ylim(0.5,1.5)
        
    def add_data(self, data, label, target=False, weights=None, histtype=None, color=None, ls=None, lw=None, alpha=None):
        args = {"label": label}
        if type(weights) != type(None): args["weights"] = weights
        if type(histtype) != type(None): args["histtype"] = histtype
        if type(color) != type(None): args["color"] = color
        if type(ls) != type(None): args["ls"] = ls
        if type(lw) != type(None): args["lw"] = lw
        if type(alpha) != type(None): args["alpha"] = alpha
        nT, bT, _ = self.ax0.hist(data, bins=np.linspace(self.xmin, self.xmax, self.nbins), density=self.density, **args)
        if target:
            self.nTt = nT
            self.bTt = bT
        else:
            self.nTs.append(nT)
            self.bTs.append(bT)
            self.argss.append(args)
        
    def plot_ratio(self):
        if type(self.nTt) == type(None):
            return
        for nT, bT, _args in zip(self.nTs, self.bTs, self.argss):
            args = {}
            if "color" in _args: args["color"] = _args["color"]
            if "ls" in _args: args["ls"] = _args["ls"]
            self.ax1.plot(0.5*(self.bTt[1:]+self.bTt[:-1]),self.nTt/(0.000001+nT), **args)
            
    def save(self):
        if type(self.save_name) == type(None):
            return
        os.makedirs("plot", exist_ok=True)
        if os.path.isfile(f"plot/{self.save_name}.pdf"):
            i = 0
            while os.path.isfile(f"plot/{self.save_name}_{i}.pdf"):
                i += 1
            self.save_name = f"{self.save_name}_{i}"
        plt.savefig(f"plot/{self.save_name}.pdf", bbox_inches='tight')
    
    def show(self):
        self.plot_ratio()
        self.ax0.locator_params(axis='y', nbins=6)
        self.ax0.legend(frameon=False,fontsize=20, loc=self.legend_corner)
        self.save()
        plt.show()
        plt.clf()




##############################################################################
## old scripts for record

def iterative_reweight(y, x_mc, y_mc, niter):
    y_train = choices(y, k=int(0.6*len(y)))
    y_mc_train = choices(y_mc, k=int(0.6*len(y_mc)))
    X1 = np.concatenate((y_train, y_mc_train)).reshape(-1,1)
    labels1 = np.concatenate((np.zeros(len(y_train)), np.ones(len(y_mc_train))))
    clf1 = RandomForestClassifier(max_depth=5)
    clf1.fit(X1, labels1)
    pred_prob1 = clf1.predict_proba(y_mc.reshape(-1,1))[:,0]
    wy = pred_prob1/(1-pred_prob1)
    wx = np.ones(len(x_mc))
    print("weight on y_mc:", wy, "\n")
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        order = np.argsort(y_mc)
        fit = splrep(y_mc[order],x_mc[order]*wx[order]/wy[order])
        x = BSpline(*fit)(y)
        x_train = choices(x, k=int(0.6*len(x)))
        x_mc_train = choices(x_mc, k=int(0.6*len(x_mc)))
        X2 = np.concatenate((x_train, x_mc_train)).reshape(-1,1)
        labels2 = np.concatenate((np.zeros(len(x_train)), np.ones(len(x_mc_train))))
        clf2 = RandomForestClassifier(max_depth=5)
        clf2.fit(X2, labels2)
        pred_prob2 = clf2.predict_proba(x_mc.reshape(-1,1))[:,0]
        wx = pred_prob2/(1-pred_prob2)
        print("updated weight on x_mc:",wx,"\n")
    return x


def omnifold_old(y, x_mc, y_mc, niter, save_iter=False):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) without using the regression

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc.
    if save_iter == True, also iter_log[niter, 3, len(x_mc)] which contains the 
    computed quantities during the iteration.
    1st coordinate is the iteration step
    2nd coordinate is the list of quantities [wy, ry, wx]
    3rd coordinate is the quantities evaluated on each Monte Carlo point
    """
    wx = np.ones(len(x_mc))
    if save_iter:
        iter_log = np.zeros((niter,3,len(x_mc)))
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        wy = wx
        y_train = choices(y, weights=np.ones(len(y)), k=int(0.6*len(y)))
        y_mc_train = choices(y_mc, weights=wy, k=int(0.6*len(y_mc)))
        X1 = np.concatenate((y_train, y_mc_train)).reshape(-1,1)
        labels1 = np.concatenate((np.zeros(len(y_train)), np.ones(len(y_mc_train))))
        clf1 = RandomForestClassifier(max_depth=5)
        clf1.fit(X1, labels1)
        pred_prob1 = clf1.predict_proba(y_mc.reshape(-1,1))[:,0]
        print(pred_prob1)
        ry = pred_prob1/(1-pred_prob1)
        print("updated weight on y_mc:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        x_mc_train1 = choices(x_mc, weights=ry, k=int(0.6*len(x_mc)))
        x_mc_train2 = choices(x_mc, weights=np.ones(len(x_mc)), k=int(0.6*len(x_mc)))
        X2 = np.concatenate((x_mc_train1, x_mc_train2)).reshape(-1,1)
        labels2 = np.concatenate((np.zeros(len(x_mc_train1)), np.ones(len(x_mc_train2))))
        clf2 = RandomForestClassifier(max_depth=5)
        clf2.fit(X2, labels2)
        pred_prob2 = clf2.predict_proba(x_mc.reshape(-1,1))[:,0]
        wx = wx * pred_prob2/(1-pred_prob2)
        print("updated weight on x_mc:",wx,"\n")
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = ry
            iter_log[t,2,:] = wx
    if save_iter:
        return wx, iter_log
    else:
        return wx



def omnifold_reg_old(y, x_mc, y_mc, niter, save_iter=False):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) using the regression.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc
    if save_iter == True, also iter_log[niter, 3, len(x_mc)] which contains the 
    computed quantities during the iteration.
    1st coordinate is the iteration step
    2nd coordinate is the list of quantities [wy, ry, wx]
    3rd coordinate is the quantities evaluated on each Monte Carlo point
    """
    wx = np.ones(len(x_mc))
    if save_iter:
        iter_log = np.zeros((niter,3,len(x_mc)))
    y_train = choices(y, k=int(0.6*len(y)))
    y_mc_train = choices(y_mc, k=int(0.6*len(y_mc)))
    X1 = np.concatenate((y_train, y_mc_train)).reshape(-1,1)
    labels1 = np.concatenate((np.zeros(len(y_train)), np.ones(len(y_mc_train))))
    clf1 = RandomForestClassifier(max_depth=5)
    clf1.fit(X1, labels1)
    pred_prob1 = clf1.predict_proba(y_mc.reshape(-1,1))[:,0]
    for t in range(niter):
        print("\nITERATION: {}\n".format(t + 1))
        print("Fitting push-forward weights on y_mc...")
        regr1 = GradientBoostingRegressor()
        regr1.fit(y_mc.reshape(-1,1), wx)
        wy = regr1.predict(y_mc.reshape(-1,1))
        ry = (pred_prob1/(1-pred_prob1))/wy
        print("updated weight on y_mc:", ry, "\n")
        
        print("Fitting pull-back weights on x_mc...")
        regr2 = GradientBoostingRegressor()
        regr2.fit(x_mc.reshape(-1,1), ry)
        wx = wx * regr2.predict(x_mc.reshape(-1,1))
        print("updated weight on x_mc:",wx,"\n")
        if save_iter:
            iter_log[t,0,:] = wy
            iter_log[t,1,:] = ry
            iter_log[t,2,:] = wx
    if save_iter:
        return wx, iter_log
    else:
        return wx


def omnifold_reg_known_densities(x_mc, y_mc, py, px_mc, py_mc, niter):
    """
    Omnifold algorithm (i.e. unbinned EM algorithm) but the densities for
    Monte Carlo (MC) data and experimental detector-level data are known.

    Returns
    -------
    unfolded experimental partile-level density p(x) evaluated at point x_mc

    """
    wx = np.ones(len(x_mc)) # initialize the weights to be 1
    for t in range(niter):
     
        print("\nITERATION: {}\n".format(t + 1))
        # First, fit the reweighted function Wy_mc on the MC detector-level data 
        # by regressing Wx_mc on Y_mc
        print("Fitting push-forward weights on y_mc...")
        #order1 = np.argsort(y_mc)
        #fit1 = splrep(y_mc[order1],wx[order1])
        #wy = BSpline(*fit1)(y_mc)
        regr1 = GradientBoostingRegressor()
        regr1.fit(y_mc.reshape(-1,1), wx)
        wy = regr1.predict(y_mc.reshape(-1,1))
        #wy = wx
        print("updated weight on y_mc:", py(y_mc)/(wy*py_mc(y_mc)), "\n")
        # Second, fit the reweighted function on the MC particle-level data by
        # regressing py/(Wy_mc*py_mc) on X_mc
        print("Fitting pull-back weights on x_mc...")
        #order2 = np.argsort(x_mc)
        #fit2 = splrep(x_mc[order2],py(y_mc[order2])/(wy[order2]*py_mc(y_mc[order2])),s=0.2)
        #wx = wx * BSpline(*fit2)(x_mc)
        regr2 = GradientBoostingRegressor()
        regr2.fit(x_mc.reshape(-1,1), py(y_mc)/(wy*py_mc(y_mc)))
        wx = wx * regr2.predict(x_mc.reshape(-1,1))
        print("updated weight on x_mc:", wx, "\n")

    return wx