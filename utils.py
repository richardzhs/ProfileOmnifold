"""
utility functions for unfolding methods
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.linear_model import Ridge
from scipy.stats import norm
from scipy import optimize
import scipy.integrate as integrate
from scipy.interpolate import splrep, BSpline
import pandas as pd


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
    """
    Perform Cholesky transformation.
    """
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
    
def generate_IBU_point_estimators_for_loop(y, K, niter):
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

def generate_IBU_point_estimators(y, K, niter):
    lambdat = np.full(K.shape[1], np.mean(y))
    for t in range(niter):
        norm_factor = np.sum(K, axis=0)
        denom = K @ lambdat
        lambdatplusone = lambdat / norm_factor * (np.transpose(K) @ (y / denom))
        lambdat = lambdatplusone
    return lambdat



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
