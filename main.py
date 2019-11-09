#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 10:30:27 2018

@author: marin
"""
# echo | powershell -command "python .\main.py | tee ('logs\{0}-output_log.txt' -f (Get-Date -format 'yyyy.MM.dd-HH.mm'))"
# C:\local\Anaconda3-4.1.1-Windows-x86_64\Scripts\activate.bat


import utils as ut
import datetime
import time
import numpy as np
# from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import pickle
from scipy.special import legendre
from numpy.random import poisson
import glob
# from numpy.random import normal
from scipy.stats import hmean

# np.random.seed(142)   # dobar
# np.random.seed(442)  # za kurac

plt.rcParams['legend.numpoints'] = 1
plt.rcParams['hatch.linewidth'] = 0.2
# plt.interactive(False)
# plt.ioff()
# plt.rcPa-rams["interactive"] = False

# plt.ion()

LOG_FILE = "log.txt"
file = ut.FILE
out_path = ut.OUT_PATH
CACHE_PATH = "data/cache/"
bin_num = ut.bin_Num


# class funParams:
#     def __init__(self):
#         param = fun_params()
#         self.coeff = param["coeff"]
#         self.orders = param["orders"]
#         self.upper_limit = param["upper_limit"]
#         self.lower_limit = param["lower_limit"]


def write_log(total_num):
    date = datetime.datetime.now()
    with open(LOG_FILE, "a") as f:
        f.write("Run date:  "+str(date)+"\t"+"Total number:  "+str(total_num)+"\n")


def rand(upper, lower):
    return (upper - lower) * np.random.random() + lower


# function that will be used for modifying angular distribution
def fun_params(n=6):
    # n = poisson(n)
    orders = np.arange(n+1)

    upper_limit = 4000*np.exp(-0.3*orders)
    lower_limit = 2000*np.exp(-0.5*orders)

    # upper_limit = -0.6*orders + 10
    # lower_limit = -0.4*orders + 5

    coeff = np.asarray([rand(i, j) for i, j in zip(upper_limit, lower_limit)])
    scale = coeff[0]/0.5
    coeff = coeff/scale

    return {"orders": orders, "coeff": coeff, "upper_limit": upper_limit/scale, "lower_limit": lower_limit/scale}


params = fun_params()
print("Parametri zadane distribucije: orders =", params["orders"], "koeficijenti =", params["coeff"])


def fun(x):
    coeff = params["coeff"]
    orders = params["orders"]
    f = np.sum(coeff.reshape(len(coeff), 1) * np.asarray([legendre(i)(x) for i in orders]), axis=0)

    # a = 0.5; b = 0.1; c = 5; d = 2
    # f = a + b*np.tanh(c*x - d)
    return f


def integral(x):
    # return quad(f, a, b)[0]
    return np.trapz(y=fun(x), x=x)


def weight(x):
    return 2*fun(x)/integral(x)


# # function that will be used for modifying angular distribution
# def fun(x, uniform):
#     if uniform:
#         return 1/2*np.ones_like(x)
#     else:
#         # return a + b*tanh(c*x - d), where a, b, c and d are parameters
#         a = 0.5; b = 0.1; c = 5; d = 2
#         return a + b*np.tanh(c*x-d)
#         # return 1 + x**3 + x**4
#         # return 1/(np.exp((0.5-x)/0.1) + 1)
#
#
# def integral(f, uniform, a=-1, b=1):
#     return quad(f, a, b, args=uniform)[0]
#
#
# def weight(x, uniform):
#     return 2*fun(x, uniform)/integral(fun, uniform)


def runAllHists():
    # data_list, tot_num = ut.loadAllFiles()
    # hists = ut.histFromDictList(data_list)

    data, tot_num = ut.loadDict(file)
    hists = ut.histFromDict(data)

    # saveToCache(data_list, "data")
    saveToCache(hists, "hists")
    saveToCache(tot_num, "tot_num")

    # for h in hists:
    #    w = np.ones_like(hists[h])/len(hists[h])
    #    ut.plotHist((h, hists[h]), bin_num=100, toFile=True, weights=w, show=False)

    # plt.figure()
    # for h in hists:
    #    ut.plotToOneHist((h, hists[h]), bin_num=100)
    # plt.close()

    # write_log(tot_num)
    print("Total num:  ", tot_num)
    return hists, tot_num


def detectorNeighbours(name, hists, n=5, legend_loc=None):
    pairs = []

    tot_num = loadFromCache("tot_num")

    asc = np.asarray([ord(c) for c in name])
    for i in range(int(np.ceil(-n/2)), int(np.ceil(n/2))):
        t = np.zeros_like(asc)
        t[-1] = i
        tmp = asc + t
        # s = [chr(c) for c in tmp]
        s = ''.join(chr(c) for c in tmp)
        # print(s)
        pairs.append((s, i))

    path = os.path.join(out_path, "neighbours/")

    legend = []
    plt.figure()
    if legend_loc is not None:
        plt.rcParams['legend.loc'] = legend_loc
    for pair, i in pairs:
        try:
            # w = np.ones_like(hists[pair])/len(hists[pair])
            w = bin_num/tot_num * np.ones_like(hists[pair])
            # w = None
            plt.hist(hists[pair], bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=False, weights=w)
            if i == 0:
                legend.append(r"$\epsilon_{i,i}$")
            else:
                legend.append(r"$\epsilon_{i,i" + "{0:+}".format(i) + r"}$")
        except KeyError as err:
            print("Key error: Detector pair "+str(err)+" not found.")
    if os.path.isfile(path+name+".png"):
        os.remove(path+name+".png")
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\epsilon$")
    plt.legend(legend)
    plt.savefig(path+name+".png", format="png", bbox_inches="tight")
    plt.show()
    # plt.close()

    # write_log(tot_num)
    print("Total num:  ", tot_num)


def modifiedDist(bin__num=100):
    bin_num = bin__num

    array, tot_num = ut.loadCosFull(file)

    # n1, bins = ut.plotHist(("Full hist", array), bin_num=100, normed=False)
    # print(n1); print("\n")

    # n2, bins = ut.plotHist(("Full hist", array), bin_num=100, normed=True)
    # print(n2); print("\n")

    # n, bins = ut.plotHist(("Full hist", array), bin_num=100, normed=False, weights=(np.ones_like(array)/tot_num))
    # print(n); print("\n")

    # ut.plotHist(("00_Full hist", array), bin_num=100, normed=False, weights=(np.ones_like(array)/tot_num), toFile=True)

    name = "00_Full_hist"
    plt.figure()
    plt.hist(array, bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=False, weights=bin_num/tot_num*np.ones_like(array))
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"$\epsilon_{i,j}(\chi)$")
    if os.path.isfile(out_path+name+".png"):
        os.remove(out_path+name+".png")
    plt.savefig(out_path+name+".png", format="png", bbox_inches="tight")
    plt.show()

    # name = "00_uniform"
    # u = np.random.uniform(-1, 1, np.shape(array))
    # w = weight(u)
    # plt.figure()
    # plt.hist(u, bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=False, weights=bin_num/tot_num*np.ones_like(array))
    # plt.hist(u, bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=False, weights=bin_num/tot_num*w)
    # plt.xlabel(r"$\chi$")
    # plt.ylabel(r"Detekcijska učinkovitost")
    # if os.path.isfile(out_path+name+".png"):
    #     os.remove(out_path+name+".png")
    # plt.savefig(out_path+name+".png")
    # plt.show()
    # # plt.close()
    #
    # name = "00_full_hist_2"
    # w = weight(array)
    # plt.figure()
    # plt.hist(array, bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=False, weights=bin_num/tot_num*np.ones_like(array))
    # plt.hist(array, bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=False, weights=bin_num/tot_num*w)
    # plt.xlabel(r"$\chi$")
    # plt.ylabel(r"Detekcijska učinkovitost")
    # if os.path.isfile(out_path+name+".png"):
    #     os.remove(out_path+name+".png")
    # plt.savefig(out_path+name+".png")
    # plt.show()
    # # plt.close()

    # write_log(tot_num)
    print("Total num:  ", tot_num)


def name_hash(name):
    N = 2 * ut.N
    index = N * (ord(name[4])-ord("A")) + (ord(name[10])-ord("A")) + N**2 * (int(name[9]) % 2)
    return index


def countsToFile():
    array1 = np.zeros(2*(2*ut.N)**2)
    array2 = np.zeros(2*(2*ut.N)**2)
    array = [""] * 2*(2*ut.N)**2
    data, tot_num = ut.loadDict(file)
    hists = ut.histFromDict(data)

    for h in hists:
        array[name_hash(h)] = h
        array1[name_hash(h)] = len(hists[h])
        array2[name_hash(h)] = sum(2*fun(np.asarray(hists[h])))

    with open("hash.txt", "w") as f:
        for i in range(len(array)):
            f.write(str(i)+":  "+str(array[i])+"\n")

    with open("counts1.txt", "w") as f:
        for i in range(len(array1)):
            f.write(str(i)+":  "+str(array1[i])+"\n")

    with open("counts2.txt", "w") as f:
        for i in range(len(array2)):
            f.write(str(i)+":  "+str(array2[i])+"\n")


def plot_counts(array1, array2, discard_percent=0.0, toFile=False, show=True, keep_zeros=False):
    discard_treshold1 = discard_percent*max(array1)
    discard_treshold2 = discard_percent*max(array2)

    array1_cut = np.copy(array1)
    threshold_indices1 = array1 < discard_treshold1
    array1_cut[threshold_indices1] = 0

    name = "00_counts"
    plt.figure()
    plt.plot(array1, "b.")
    plt.plot(array2, "r.")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$N_{\alpha}$")
    if toFile is True:
        if os.path.isfile(out_path+name+".png"):
            os.remove(out_path+name+".png")
        plt.savefig(out_path+name+".png", format="png", bbox_inches="tight")
    if show is True:
        plt.show()
    plt.close()

    name = "00_counts_cut"

    if keep_zeros is True:
        array2_cut = np.copy(array2)
        threshold_indices2 = array2 < discard_treshold2
        array2_cut[threshold_indices2] = 0

        plt.figure()
        plt.plot(array1_cut, "b.")
        plt.plot(array2_cut, "r.")
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$N_{\alpha}$")
        if toFile is True:
            if os.path.isfile(out_path+name+".png"):
                os.remove(out_path+name+".png")
            plt.savefig(out_path+name+".png", format="png", bbox_inches="tight")
        if show is True:
            plt.show()
        plt.close()

    else:
        plt.figure()
        plt.plot(array1[array1 > discard_treshold1], "b.")
        plt.plot(array2[array2 > discard_treshold2], "r.")
        if toFile is True:
            if os.path.isfile(out_path+name+".png"):
                os.remove(out_path+name+".png")
            plt.savefig(out_path+name+".png", format="png", bbox_inches="tight")
        if show is True:
            plt.show()
        plt.close()


def chi_legendre_percents(order, discard_percents, N_uncut, N0_uncut, uniform):

    if uniform:
        title = "UNIFORM"
    elif not uniform:
        title = "NONLINEAR"

    chi_vals = []
    d_chi_vals = []
    for discard_percent in discard_percents:
        discard_percent = np.round(discard_percent, 2)

        gamma, P_orders = Gamma_uncut(order)
        N, _, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent)
        gamma = Cut_epsilon_or_gamma(gamma, indices)
        M = np.linalg.inv(np.dot(np.dot(gamma.T, V_inv), gamma))
        aa = np.dot(np.dot(np.dot(M, gamma.T), V_inv), N)

#        a = aa/(2*aa[0])
        daa = np.sqrt(M.diagonal())
#        da = np.sqrt( (daa/(2*aa[0]))**2 + ((aa*daa[0])/(2*aa[0]**2))**2 )
#        da[0] = 0

#        xs = np.linspace(-1,1, num=100)
#         if uniform:
#             A = 1 / 2 * np.ones_like(xs)
#         else:
#             A = fun(xs) / integral(xs)
#        A_r = np.sum(a.reshape(len(a),1) *np.asarray([legendre(i)(xs) for i in P_orders]), axis=0)
#        dA_r = np.sqrt(np.sum((da.reshape(len(da),1) *np.asarray([legendre(i)(xs) for i in P_orders]))**2, axis=0))

        N_r = np.dot(gamma, aa)

        d_chi = np.sqrt(1/(len(N)-len(P_orders)) * np.sum((2*daa * np.sum(np.dot((N_r-N)/N, gamma)))**2))
        d_chi_vals.append(d_chi)

        chi_val = chi(N_r, N, len(P_orders))
        chi_vals.append(chi_val)

    chi_vals = np.asarray(chi_vals)
    d_chi_vals = np.asarray(d_chi_vals)

    alpha = 0.05
    loss = chi_vals*d_chi_vals + chi_vals + d_chi_vals + alpha*order

    fig = plt.figure()
    fig.canvas.set_window_title('Chi values '+title)
    plt.title("Chi values")
    plt.plot(discard_percents, loss, "r.")
    plt.errorbar(discard_percents, chi_vals, yerr=d_chi_vals, fmt='b.', elinewidth=0.5, capsize=5)
    plt.legend(["Loss", "Chi"])
    # plt.savefig(os.path.join(out_path, file), format="svg")
    plt.show()
    # plt.close()


def CountsVec_uncut(uniform, alpha=50.0):
    hists = loadFromCache("hists")
    N = np.zeros(((2*ut.N)**2, 1))

    if uniform:
        for h in hists:
            index = name_hash(h)
            if index < 1024:
                N[index] = len(hists[h])

    else:
        for h in hists:
            index = name_hash(h)
            if index < 1024:
                N[index] = sum(2*fun(np.asarray(hists[h])))

    N = N/(alpha**2)
    N0 = np.copy(N)
    N = poisson(N0)
    N = N*np.sum(N0)/np.sum(N)

    return N, N0, uniform


def Epsilon_uncut(bin_num):
    hists = loadFromCache("hists")
    n_tot = loadFromCache("tot_num")
    epsilon = np.zeros((1024, bin_num-1))

    for h in hists:
        index = name_hash(h)
        if index < 1024:
            n, bins = ut.plotHist((h, hists[h]), bin_num=bin_num, plot=False, toFile=False, weights=None, show=False, normed=False)
            epsilon[name_hash(h), :] = (2/n_tot * n)

    X = np.asarray([bins[i]-bins[i-1] for i in range(1, len(bins))])

    return epsilon, X


def Cut_N(N, N0, discard_percent):

    discard_treshold = discard_percent*max(N)
    indices = N > discard_treshold

    N = N[indices]
    N0 = N0[indices]

    V_inv = np.eye(len(N))*1/N
    # V_inv = (alpha**2)*V_inv

    return N, N0, V_inv, indices


def Cut_epsilon_or_gamma(epsilon_or_gamma, indices):

    epsilon_or_gamma = epsilon_or_gamma[indices.flatten(), :]

    return epsilon_or_gamma


def saveToCache(data, fileName, folder=CACHE_PATH):
    with open(os.path.join(folder, fileName+".pkl"), "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def loadFromCache(fileName, folder=CACHE_PATH):
    with open(os.path.join(folder, fileName+".pkl"), "rb") as f:
        return pickle.load(f)


def saveToCache_numpy(data, fileName, folder=CACHE_PATH):
    save_path = os.path.join(folder, fileName+".npy")
    np.save(save_path, data)


def loadFromCache_numpy(fileName, folder=CACHE_PATH):
    load_path = os.path.join(folder, fileName+".npy")
    return np.load(load_path).item()


def linearFit(N, N0, epsilon, V_inv, X, uniform, plot=True, show=True, to_file=False):
    start_time = time.time()

    if uniform:
        title = "UNIFORM"
    else:
        title = "NONLINEAR"

    out_path = os.path.join("out", "reconstructed_dist")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # num_of_files = len([i for i in next(os.walk(out_path))[2] if i.endswith(".svg") or i.endswith(".png")])
    # num_of_files = len(os.listdir(out_path))
    # num_of_files = len([name for name in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, name))])
    num_of_files = len(glob.glob(out_path + r"\*_{}_dist_linear.svg".format(title)))

    run_num = "run{}".format(num_of_files)

    # print(np.shape(N), np.shape(epsilon), np.shape(V_inv))

    M = np.linalg.inv(np.dot(np.dot(epsilon.T, V_inv), epsilon))

    AA_r = np.dot(np.dot(np.dot(M, epsilon.T), V_inv), N)
    # print(np.shape(AA_r))

    deltaAA_r = np.sqrt(M.diagonal())
    # print(deltaAA_r.shape, deltaAA_r.shape)

    s = np.sum(AA_r*X)
    A_r = AA_r/s

    deltaA_r = (A_r/s)**2 * np.sum((X*deltaAA_r)**2) + (deltaAA_r/s)**2 - 2*AA_r*X * deltaAA_r**2/s**3

    xs = np.linspace(-1, 1, num=len(A_r))

    if uniform:
        A = 1/2*np.ones_like(xs)
    else:
        A = fun(xs)/integral(xs)

    # N_r = np.dot(epsilon, AA_r)

    if plot:
        fig = plt.figure()
        fig.canvas.set_window_title('Counts with Linear Fit '+title)
        # plt.subplot(2, 1, 1)
        plt.title("Brojevi detekcija")
        # plt.plot(N_r, "r.", label="N_r")
        plt.plot(np.sort(N0)[::-1]/max(N0), "b.", label="N")
        print("max(N0) =", max(N0))
        # plt.plot(N0, "g.", label="N_0")
        # plt.legend((r"$N_{\alpha}^{\mathrm{(r)}}$", r"$N_\alpha$", r"$N_\alpha^{(o)}$"))
        # plt.legend([r"$N_{\alpha}$"])
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$N_{\alpha}/\mathrm{max}(N_{\alpha})$")
        # plt.yscale("log")

        if to_file:
            # plt.savefig(os.path.join(out_path, "{r}_{t}_counts_linear.svg".format(t=title, r=run_num)), format="svg", bbox_inches="tight")
            plt.savefig(os.path.join(out_path, "{r}_{t}_counts_linear.png".format(t=title, r=run_num)), format="png", bbox_inches="tight")

        if show:
            plt.show()
        if not show:
            plt.close()

        # plt.subplot(2, 1, 2)
        fig2 = plt.figure()
        fig2.canvas.set_window_title('Distribution with Linear Fit ' + title)
        plt.title("Dstribucija")
        plt.plot(xs, A_r, "r.")
        plt.plot(xs, A_r+deltaA_r, "r--", linewidth=1)
        plt.plot(xs, A_r-deltaA_r, "r--", linewidth=1)
        plt.fill_between(xs, A_r-deltaA_r, A_r+deltaA_r, facecolor='red', hatch="///", interpolate=True, alpha=0.1)
        plt.plot(xs, A, "b-")
        plt.ylim((-0.5, 3.0))
        plt.xlabel(r"$\chi$")
        plt.ylabel(r"$A(\chi)$")

        if to_file:
            # plt.savefig(os.path.join(out_path, "{r}_{t}_dist_linear.svg".format(t=title, r=run_num)), format="svg", bbox_inches="tight")
            plt.savefig(os.path.join(out_path, "{r}_{t}_dist_linear.png".format(t=title, r=run_num)), format="png", bbox_inches="tight")

        if show:
            plt.show()
        if not show:
            plt.close()

    end_time = time.time()
    print("'linearFit()' run time: ", end_time-start_time, "seconds")


def Gamma_uncut(n):

    # P_orders = np.arange(0, n+1)
    P_orders = range(0, n+1)

    hists = loadFromCache("hists")
    n_tot = loadFromCache("tot_num")

    gamma = np.zeros((1024, len(P_orders)))

    for h in hists:
        index = name_hash(h)
        if index < 1024:
            hist = np.asarray(hists[h])
            gamma[index, :] = 2/n_tot * np.asarray([np.sum(legendre(i)(hist), axis=0) for i in P_orders])

    return gamma, P_orders


def legendreFit(N, N0, gamma, V_inv, P_orders, uniform, plot=True, show=True, to_file=False, plot_title=None):
    start_t = time.time()

    if uniform:
        title = "UNIFORM"
    else:
        title = "NONLINEAR"

    out_path = os.path.join("out", "reconstructed_dist")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # num_of_files = len(next(os.walk(out_path))[2])
    # num_of_files = len(os.listdir(out_path))
    # num_of_files = len([name for name in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, name))])
    num_of_files = len(glob.glob(out_path + r"\*_{}_dist_legendre.svg".format(title)))

    run_num = "run{}".format(num_of_files)

    M = np.linalg.inv(np.dot(np.dot(gamma.T, V_inv), gamma))

    aa = np.dot(np.dot(np.dot(M, gamma.T), V_inv), N)
    a = aa/(2*aa[0])

    daa = np.sqrt(M.diagonal())

    da = np.sqrt((daa/(2*aa[0]))**2 + ((aa*daa[0])/(2*aa[0]**2))**2)
    da[0] = 0

    # xs = np.arange(-1,1, 2/100)
    xs = np.linspace(-1, 1, num=100)

    if uniform:
        A = 1 / 2 * np.ones_like(xs)
    else:
        A = fun(xs) / integral(xs)

    A_r = np.sum(a.reshape(len(a), 1) * np.asarray([legendre(i)(xs) for i in P_orders]), axis=0)

    dA_r = np.sqrt(np.sum((da.reshape(len(da), 1) * np.asarray([legendre(i)(xs) for i in P_orders]))**2, axis=0))

    N_r = np.dot(gamma, aa)

    # print("NORM:  ", np.linalg.norm(A-A_r), end="\n\n")

    if plot:
        fig = plt.figure()
        fig.canvas.set_window_title('Distribution with Legendre Fit '+title)

        plt.subplot(2, 1, 2)
        if plot_title is not None:
            plt.title(plot_title)
        plt.plot(xs, A_r, "r.")
        plt.plot(xs, A_r+dA_r, "r--", linewidth=1)
        plt.plot(xs, A_r-dA_r, "r--", linewidth=1)
        plt.fill_between(xs, A_r-dA_r, A_r+dA_r, facecolor='red', hatch="///", interpolate=True, alpha=0.1)
        plt.plot(xs, A, "b-")
        plt.xlabel(r"$\chi$")
        plt.ylabel(r"$A(\chi)$")
        # plt.show()

        plt.subplot(2, 1, 1)
        # plt.title("Koeficijenti")

        if uniform:
            plt.plot([0], [0.5], "b.", label="zadani")
        else:
            plt.plot(params["orders"], params["coeff"], "b.", label="default")
            # plt.fill_between(params["orders"], params["lower_limit"], params["upper_limit"], facecolor='blue',
            #                                 hatch="///", linewidth=0.1, interpolate=True, alpha=0.1, label="range")

        plt.errorbar(np.arange(len(a)), a, yerr=da, fmt='r.', elinewidth=0.5, capsize=5, label="reconstructed")
        plt.xlabel(r"$l$")
        plt.ylabel(r"$a_l$")
        plt.legend(("referentni", "rekonstruirani"))

        # if uniform:
        #     plt.legend(("referentni", "rekonstruirani"))
        # else:
        #     plt.legend(("referentni", "područje", "rekonstruirani"))

        if to_file:
            # plt.savefig(os.path.join(out_path, "{r}_{t}_dist_legendre.svg".format(t=title, r=run_num)), format="svg", bbox_inches="tight")
            plt.savefig(os.path.join(out_path, "{r}_{t}_dist_legendre.png".format(t=title, r=run_num)), format="png")
        if show:
            plt.show()
        if not show:
            plt.close()

        # fig2 = plt.figure()
        # fig2.canvas.set_window_title('Counts and Coefficients: '+title)
        # # plt.subplot(2, 2, 1)
        # plt.title("Brojevi detekcija")
        # # plt.plot(N_r, "r.", label="N_r")
        # plt.plot(np.sort(N0)[::-1]/max(N0), "b.", label="N")
        # print("max(N0) =", max(N0))
        # # plt.plot(N0, "g.", label="N_0")
        # plt.xlabel(r"$\alpha$")
        # plt.ylabel(r"$N_\alpha\: /\: \mathrm{max}[N_{\alpha}]$")
        # # plt.legend((r"$N_{\alpha}^{\mathrm{(r)}}$", r"$N_\alpha$", r"$N_\alpha^{(o)}$"))
        #
        # if to_file:
        #     # plt.savefig(os.path.join(out_path, "{r}_{t}_dist_legendre.svg".format(t=title, r=run_num)), format="svg", bbox_inches="tight")
        #     plt.savefig(os.path.join(out_path, "{r}_{t}_counts_legendre.png".format(t=title, r=run_num)), format="png")
        # if show:
        #     plt.show()
        # if not show:
        #     plt.close()

    end_t = time.time()
    print("'legendreFit()' run time: ", end_t-start_t, "seconds")

    return {"A": A, "A_r": A_r, "dA_r": dA_r, "N_r": N_r}


def chi(N_r, N, par_num, exponent=2):
    # return np.sum((N_r-N)**2/N) / (len(N)-par_num)    # chi-squared
    return np.sum(np.abs((N_r - N) / np.sqrt(N))**exponent) / (len(N) - par_num)
    # return np.sum((N_r-N)**2/N)


def derivative(x_plus, x_minus, delta):
    return (x_plus-x_minus) / (2*delta)


def d_chi(aa, gamma, N, par_num, i, delta):
    delta_aa = np.zeros_like(aa)
    delta_aa[i] = delta
    aa_plus = aa + delta_aa
    aa_minus = aa - delta_aa
    N_r_plus = np.dot(gamma, aa_plus)
    N_r_minus = np.dot(gamma, aa_minus)
    chi_plus = chi(N_r_plus, N, par_num)
    chi_minus = chi(N_r_minus, N, par_num)
    return (4*derivative(chi_plus, chi_minus, delta/2) - derivative(chi_plus, chi_minus, delta))/3


def delta_chi(daa, aa, gamma, N, par_num, delta):
    delta_chi_sq = 0
    for i in range(len(aa)):
        delta_chi_sq += (d_chi(aa, gamma, N, par_num, i, delta)*daa[i])**2

    return np.sqrt(delta_chi_sq)


def chi_legendre(orders, discard_percent, N_uncut, N0_uncut, uniform, plot=True, show=True, to_file=False, lam=0.04, exponent=2):
    # good lam beetwen 0.01 and 0.06

    if uniform:
        title = "UNIFORM"
    else:
        title = "NONLINEAR"

    out_path = os.path.join("out", "reconstructed_dist")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # num_of_files = len(next(os.walk(out_path))[2])
    # num_of_files = len(os.listdir(out_path))
    # num_of_files = len([name for name in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, name))])
    num_of_files = len(glob.glob(out_path + r"\*_{}_chi_vals_legendre.svg".format(title)))

    run_num = "run{}".format(num_of_files)

    graphs = []
    chi_vals = []
    d_chi_vals = []
    for o in orders:
        # print("order = ", o)

        gamma, P_orders = Gamma_uncut(o)
        N, _, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent)
        gamma = Cut_epsilon_or_gamma(gamma, indices)
        M = np.linalg.inv(np.dot(np.dot(gamma.T, V_inv), gamma))
        aa = np.dot(np.dot(np.dot(M, gamma.T), V_inv), N)
        daa = np.sqrt(M.diagonal())

        # a = aa/(2*aa[0])
        # da = np.sqrt((daa/(2*aa[0]))**2 + ((aa*daa[0])/(2*aa[0]**2))**2)
        # da[0] = 0

        # xs = np.linspace(-1, 1, num=100)

        # if uniform:
        #     A = 1 / 2 * np.ones_like(xs)
        # else:
        #     A = fun(xs) / integral(xs)

        # A_r = np.sum(a.reshape(len(a), 1) * np.asarray([legendre(i)(xs) for i in P_orders]), axis=0)
        #
        # dA_r = np.sqrt(np.sum((da.reshape(len(da), 1) * np.asarray([legendre(i)(xs) for i in P_orders]))**2, axis=0))

        N_r = np.dot(gamma, aa)

        d_Chi = 1/(len(N)-len(P_orders)) * np.sqrt(np.sum((2*daa * np.dot((N_r-N)/N, gamma))**2))
        # d_Chi = np.sqrt(np.sum((2 * daa * np.dot((N_r - N) / N, gamma)) ** 2))

        # d_Chi = np.sqrt(np.sum(np.dot(gamma, daa)**2))

        # d_Chi = np.trapz(y=A_r+dA_r, x=xs) - np.trapz(y=A_r-dA_r, x=xs)

        # dChi = delta_chi(daa, aa, gamma, N, par_num=len(P_orders), delta=0.001)

        # print("d_Chi =", d_Chi, "dChi =", dChi, "d_chi difference =", d_Chi-dChi)

        d_chi_vals.append(d_Chi)

        # chi_val = np.abs(np.linalg.norm(N_r) - np.linalg.norm(N))
        # chi_val = np.linalg.norm(N_r - N)
        chi_val = chi(N_r, N, len(P_orders), exponent)

        chi_vals.append(chi_val)
    graphs.append((chi_vals, d_chi_vals))

    chi_vals = np.asarray(chi_vals)
    d_chi_vals = np.asarray(d_chi_vals)

    loss = chi_vals
    # loss = chi_vals * d_chi_vals
    # loss = np.log(chi_vals) + lam*np.log(d_chi_vals)
    # loss = np.sqrt(chi_vals * d_chi_vals)

    # loss = chi_vals*d_chi_vals + chi_vals + d_chi_vals + alpha*orders
    # loss = hmean([chi_vals, d_chi_vals], axis=0)
    # loss = (np.sqrt(chi_vals*d_chi_vals) + hmean([chi_vals, d_chi_vals], axis=0))/2

    if plot:
        fig = plt.figure()
        fig.canvas.set_window_title('Chi values with Legendre Fit '+title)

        plt.subplot(2, 1, 1)
        # plt.title(r"$\chi^2$ vrijednosti")
        plt.errorbar(orders, chi_vals, yerr=d_chi_vals, fmt='b.', elinewidth=0.5, capsize=5)
        plt.xticks(orders)
        plt.xlabel(r"$\mathcal{L}$")
        plt.ylabel(r"$\chi^2_\mathcal{L}$")
        # plt.legend([r"$\chi^2$"])

        plt.subplot(2, 1, 2)
        # plt.title(r"$\chi^2 \cdot \Delta \chi^2$ vrijednosti")
        plt.yscale("log")
        plt.plot(orders, loss, "r.")
        plt.xticks(orders)
        plt.xlabel(r"$\mathcal{L}$")
        plt.ylabel(r"$\chi^2_\mathcal{L} \cdot \Delta \chi^2_\mathcal{L}$")
        # plt.legend([r"$\chi^2 \cdot \Delta \chi^2$"])

        if to_file:
            # plt.savefig(os.path.join(out_path, "{r}_{t}_chi_vals_legendre.svg".format(t=title, r=run_num)), format="svg", bbox_inches="tight")
            plt.savefig(os.path.join(out_path, "{r}_{t}_chi_vals_legendre.png".format(t=title, r=run_num)), format="png")

        if show:
            plt.show()
        if not show:
            plt.close()

    # cache = {"graphs": graphs, "orders": orders, "discard_percent": discard_percent}
    # saveToCache(cache, "run{}".format(num_of_files), folder=out_path)
    #
    # with open(os.path.join(out_path, "run{}.txt".format(num_of_files)), "w") as f:
    #     for chi_vals in graphs:
    #         f.write(str(chi_vals) + "\n")

    # print("loss = ", loss)

    # optimal_ord = orders[np.where(loss == np.min(np.abs(loss)))[0][0]]
    optimal_ord = orders[np.where(loss == np.min(loss))[0][0]]
    # print("Optimal order found is: ", optimal_ord)
    return optimal_ord


def chi_linear(bin_nums, discard_percent, N_uncut, N0_uncut, uniform, plot=True, show=True, to_file=False):

    if uniform:
        title = "UNIFORM"
    else:
        title = "NONLINEAR"

    out_path = os.path.join("out", "reconstructed_dist")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # num_of_files = len([i for i in next(os.walk(out_path))[2] if i.endswith(".svg") or i.endswith(".png")])
    # num_of_files = len(os.listdir(out_path))
    # num_of_files = len([name for name in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, name))])
    num_of_files = len(glob.glob(out_path + r"\*_{}_chi_vals_linear.svg".format(title)))

    run_num = "run{}".format(num_of_files)

    chi_vals = []
    dA_vals = []
    for bin_num in bin_nums:
        print("bin_num = ", bin_num)

        epsilon, X = Epsilon_uncut(bin_num)
        N, _, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent)
        epsilon = Cut_epsilon_or_gamma(epsilon, indices)

        M = np.linalg.inv(np.dot(np.dot(epsilon.T, V_inv), epsilon))
        AA_r = np.dot(np.dot(np.dot(M, epsilon.T), V_inv), N)

        # deltaAA_r = np.sqrt(M.diagonal())
        # s = np.sum(AA_r*X)
        # A_r = AA_r/s
        # deltaA_r = (A_r / s) ** 2 * np.sum((X * deltaAA_r) ** 2) + (deltaAA_r / s) ** 2 - 2 * AA_r * X * deltaAA_r ** 2 / s ** 3

        # xs = np.linspace(-1,1, num=len(A_r))
        # if uniform:
        #     A = 1 / 2 * np.ones_like(xs)
        # else:
        #     A = fun(xs) / integral(xs)

        N_r = np.dot(epsilon, AA_r)

#        d_chi = np.sqrt(1/(len(N)-len(bin_nums-1)) * np.sum((2*daa * np.sum(np.dot((N_r-N)/N, gamma)))**2))
#        d_chi_vals.append(d_chi)

        chi_val = chi(N_r, N, bin_num-1)
        chi_vals.append(chi_val)
        # dA_vals.append(np.sum(deltaA_r))

    chi_vals = np.asarray(chi_vals)
    # dA_vals = np.asarray(dA_vals)
    loss = chi_vals

    if plot:
        fig = plt.figure()
        plt.title("Chi kvadrat vrijednosti")
        fig.canvas.set_window_title('Chi values with Linear Fit '+title)
        plt.plot(bin_nums, chi_vals, ".")
        plt.xlabel(r"$\mathcal{L}")
        plt.ylabel(r"$\chi$")

        # plt.ylim(bottom=0, top=10)
        # plt.legend((discard_percent))
        if to_file:
            plt.savefig(os.path.join(out_path, "{r}_{t}_chi_vals_linear.svg".format(t=title, r=run_num)), format="svg", bbox_inches="tight")
            plt.savefig(os.path.join(out_path, "{r}_{t}_chi_vals_linear.png".format(t=title, r=run_num)), format="png", bbox_inches="tight")
        if show:
            plt.show()
        if not show:
            plt.close()
            # plt.close()

    # cache = {"graphs": graphs, "bin_nums": bin_nums, "discard_percent": discard_percent}
    # saveToCache(cache, "run{}".format(num_of_files), folder=out_path)
    #
    # with open(os.path.join(out_path, "run{}.txt".format(num_of_files)), "w") as f:
    #     for chi_vals in graphs:
    #         f.write(str(chi_vals) + "\n")

    optimal_bin_num = bin_nums[np.where(loss == np.min(np.abs(loss)))[0][0]]
    # print("Optimal number of bins found is: ", optimal_bin_num)
    return optimal_bin_num


def main():
    # runAllHists()
    # modifiedDist()

    # hists = loadFromCache("hists")
    # for i in range(32):
    #     c = chr(ord("A")+i)
    #     detectorNeighbours("Si_1"+c+"-Si_2"+c, hists)

    # hists = loadFromCache("hists")
    # plt.figure()
    # for h in hists:
    #     plt.hist(hists[h], bins=np.arange(-1, 1, 2/bin_num), histtype='step', density=False, weights=None)
    # if os.path.isfile(out_path+"00_all_pairs.png"):
    #     os.remove(out_path+"00_all_pairs.png")
    # plt.savefig(out_path+"00_all_pairs.png", format="png", bbox_inches="tight")
    # plt.show()

    # hists = loadFromCache("hists")
    # tot_num = loadFromCache("tot_num")
    # w = bin_num / tot_num * np.ones_like(hists["Si_1M-Si_2M"])
    # plt.figure()
    # plt.hist(hists["Si_1M-Si_2M"], bins=np.arange(-1, 1, 2 / bin_num), histtype='step', density=False, weights=w)
    # plt.xlabel(r"$\chi$")
    # plt.ylabel(r"$\epsilon$")
    # plt.legend([r"$\epsilon_{i,i}$"])
    # plt.savefig("out.png", format="png", bbox_inches="tight")
    # plt.show()
    #
    # detectorNeighbours("Si_1M-Si_2M", hists)
    # detectorNeighbours("Si_1Y-Si_2Y", hists, legend_loc="upper left")

    # hists = loadFromCache("hists")
    # array1 = np.zeros(((2 * ut.N) ** 2, 1))
    # array2 = np.zeros(((2 * ut.N) ** 2, 1))
    # for h in hists:
    #     index = name_hash(h)
    #     if index < 1024:
    #         array1[index] = len(hists[h])
    #         array2[index] = sum(2 * fun(np.asarray(hists[h])))
    # plt.figure()
    # plt.plot(array1, "b.")
    # plt.plot(array2, "r.")
    # plt.xlabel(r"$\alpha$")
    # plt.ylabel(r"$N_{\alpha}$")
    # plt.legend([r"$N_{\alpha}$", r"$N_{\alpha}^{\mathrm{mod}}$"])
    # if os.path.isfile("Figure_1.png"):
    #     os.remove("Figure_1.png")
    # plt.savefig("Figure_1.png", format="png", bbox_inches="tight")
    # plt.show()
    #
    # N_uncut, N0_uncut, _ = CountsVec_uncut(uniform=True, alpha=50.0)
    # N1, _, _, _ = Cut_N(N_uncut, N0_uncut, discard_percent=0.05)
    # N_uncut, N0_uncut, _ = CountsVec_uncut(uniform=False, alpha=50.0)
    # N2, _, _, _ = Cut_N(N_uncut, N0_uncut, discard_percent=0.05)
    # plt.figure()
    # plt.plot(N1, "b.")
    # plt.plot(N2, "r.")
    # plt.xlabel(r"$\alpha$")
    # plt.ylabel(r"$N_{\alpha}$")
    # plt.legend([r"$N_{\alpha}$", r"$N_{\alpha}^{\mathrm{mod}}$"])
    # if os.path.isfile("Figure_2.png"):
    #     os.remove("Figure_2.png")
    # plt.savefig("Figure_2.png", format="png", bbox_inches="tight")
    # plt.show()

    # tot_num = loadFromCache("tot_num")
    # hists = loadFromCache("hists")
    #
    # pairs = ["Si_1M-Si_2K", "Si_1M-Si_2L", "Si_1M-Si_2M", "Si_1M-Si_2N"]
    # plt.figure()
    # for pair in pairs:
    #     w = bin_num / tot_num * np.ones_like(hists[pair])
    #     plt.hist(hists[pair], bins=np.arange(-1, 1, 2 / bin_num), histtype='step', density=False, weights=w)
    # legend = [r"$\epsilon_{i,i-2}$", r"$\epsilon_{i,i-1}$", r"$\epsilon_{i,i}$", r"$\epsilon_{i,i+1}$"]
    # plt.xlabel(r"$\chi$")
    # plt.ylabel(r"$\epsilon_{i,j}(\chi)$")
    # plt.legend(legend)
    # plt.show()
    #
    # pairs = ["Si_1Y-Si_2X", "Si_1Y-Si_2Y", "Si_1Y-Si_2Z"]
    # plt.figure()
    # plt.rcParams['legend.loc'] = "upper left"
    # for pair in pairs:
    #     w = bin_num / tot_num * np.ones_like(hists[pair])
    #     plt.hist(hists[pair], bins=np.arange(-1, 1, 2 / bin_num), histtype='step', density=False, weights=w)
    # legend = [r"$\epsilon_{i,i-1}$", r"$\epsilon_{i,i}$", r"$\epsilon_{i,i+1}$"]
    # plt.xlabel(r"$\chi$")
    # plt.ylabel(r"$\epsilon_{i,j}(\chi)$")
    # plt.legend(legend)
    # plt.show()

    #
    # ------------------------------------------------------------------------
    #

    for i in range(1):
        print("\n", "Run: ", i, "\n")

        # N_uncut, N0_uncut, uniform = CountsVec_uncut(uniform=True, alpha=55)
        #
        # o = chi_legendre(np.arange(0, 20, 1), 0.04, N_uncut, N0_uncut, uniform, plot=True, show=True, to_file=False)
        #
        # # chi_legendre_percents(o, np.arange(0.0, 0.2, 0.02), N_uncut, N0_uncut, uniform)
        #
        # N, N0, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent=0.04)
        # gamma, P_orders = Gamma_uncut(o)
        # gamma = Cut_epsilon_or_gamma(gamma, indices)
        # legendreFit(N, N0, gamma, V_inv, P_orders, uniform, plot=True, show=True, to_file=False)
        #
        # # -------------------------------------------------------
        #
        # bin_n = chi_linear(range(10, 70, 5), 0.001, N_uncut, N0_uncut, uniform, plot=False, show=True, to_file=False)
        #
        # N, N0, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent=0.001)
        #
        # epsilon, X, = Epsilon_uncut(bin_n)
        # epsilon = Cut_epsilon_or_gamma(epsilon, indices)
        # linearFit(N, N0, epsilon, V_inv, X, uniform, plot=True, show=True, to_file=False)

        #
        # --------------------------------------------------
        #

        N_uncut, N0_uncut, uniform = CountsVec_uncut(uniform=False, alpha=55.0)

        # lam = rand(0.05, 0.05)
        # lam = 0.02
        lam = 0.01
        print("lambda = ", lam)
        o = chi_legendre(np.arange(0, 10, 1), 0.04, N_uncut, N0_uncut, uniform, lam=lam, plot=False, show=True, to_file=False)
        N, N0, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent=0.04)
        gamma, P_orders = Gamma_uncut(o)
        gamma = Cut_epsilon_or_gamma(gamma, indices)
        legendreFit(N, N0, gamma, V_inv, P_orders, uniform, plot=True, show=True, to_file=False)

        # -------------------------------------------------------

        # bin_n = chi_linear(range(10, 70, 5), 0.001, N_uncut, N0_uncut, uniform, plot=False, show=True, to_file=False)
        # N, N0, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent=0.001)
        # epsilon, X, = Epsilon_uncut(bin_n)
        # epsilon = Cut_epsilon_or_gamma(epsilon, indices)
        # linearFit(N, N0, epsilon, V_inv, X, uniform, plot=True, show=True, to_file=False)

        #
        # --------------------------------------------------
        #


def lambda_sampling():
    for i in range(10):
        lams = []
        # diff = []
        norm_diff = []
        # diff_norm = []
        # chis = []

        N_uncut, N0_uncut, uniform = CountsVec_uncut(uniform=False, alpha=55.0)

        print(f"Run {i}")
        for lam in np.arange(0, 0.1, 0.005):
            o = chi_legendre(np.arange(0, 10, 1), 0.04, N_uncut, N0_uncut, uniform, lam=lam, plot=False, show=True, to_file=False)
            N, N0, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent=0.04)
            gamma, P_orders = Gamma_uncut(o)
            gamma = Cut_epsilon_or_gamma(gamma, indices)
            data = legendreFit(N, N0, gamma, V_inv, P_orders, uniform, plot=False, show=True, to_file=False)

            # chi_sq = np.sum((data["A"]-data["A_r"])**2/data["dA_r"]) / (len(data["A"])-o)
            # estims = [np.abs(np.sum(data["A"]-data["A_r"])), np.linalg.norm(data["A"]-data["A_r"]),
            #           np.abs(np.linalg.norm(data["A"])-np.linalg.norm(data["A_r"])), chi_sq]

            print("lambda = ", lam,  "norm(A-A_r) = ", np.linalg.norm(data["A"]-data["A_r"]))

        #     print("lambda = ", lam)
        #     print(*["A-A_r", "norm(A-A_r)", "norm(A)-norm(A_r)", "chi(A, A_r)"], sep="\t\t")
        #     print(*estims, sep="\t\t")
        #
            lams.append(lam)
        #     diff.append(estims[0])
            norm_diff.append(np.linalg.norm(data["A"]-data["A_r"]))
        #     diff_norm.append(estims[2])
        #     chis.append(estims[3])
        #
        # print("Best lambda A-A_r:               lam = ", lams[diff.index(min(diff))], "A-A_r = ", min(diff))
        print("Best lambda norm(A-A_r):         lam = ", lams[norm_diff.index(min(norm_diff))], "norm(A-A_r) = ", min(norm_diff))
        # print("Best lambda norm(A)-norm(A_r):   lam = ", lams[diff_norm.index(min(diff_norm))], "norm(A)-norm(A_r) = ", min(diff_norm))
        # print("Best lambda chi(A, A_r):         lam = ", lams[chis.index(min(chis))], "chi(A, A_r) = ", min(chis))


def lambda_test():
    for lam in np.arange(0.05, 0.1, 0.002):
        N_uncut, N0_uncut, uniform = CountsVec_uncut(uniform=False, alpha=55.0)

        o = chi_legendre(np.arange(0, 10, 1), 0.04, N_uncut, N0_uncut, uniform, lam=lam, plot=False, show=True,
                         to_file=False)
        N, N0, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent=0.04)
        gamma, P_orders = Gamma_uncut(o)
        gamma = Cut_epsilon_or_gamma(gamma, indices)
        legendreFit(N, N0, gamma, V_inv, P_orders, uniform, plot=True, show=True, to_file=False)
        print("lambda = ", lam, "order = ", o)


def chi_exponent_sampling():
    for i in range(10):
        pows = []
        norm_diff = []

        for e in np.arange(0.1, 10, 0.5):
            N_uncut, N0_uncut, uniform = CountsVec_uncut(uniform=False, alpha=55.0)

            # lam = rand(0.05, 0.05)
            # lam = 0.02
            lam = 0.01
            o = chi_legendre(np.arange(0, 10, 1), 0.04, N_uncut, N0_uncut, uniform, lam=lam, exponent=e, plot=False,
                             show=True, to_file=False)
            N, N0, V_inv, indices = Cut_N(N_uncut, N0_uncut, discard_percent=0.04)
            gamma, P_orders = Gamma_uncut(o)
            gamma = Cut_epsilon_or_gamma(gamma, indices)
            title = "".join(["chi exponent = ", str(e)])
            data = legendreFit(N, N0, gamma, V_inv, P_orders, uniform, plot=False, show=True, to_file=False, plot_title=title)

            print("exponent = ", e,  "norm(A-A_r) = ", np.linalg.norm(data["A"]-data["A_r"]))

            pows.append(e)
            norm_diff.append(np.linalg.norm(data["A"]-data["A_r"]))

        print("Best lambda norm(A-A_r):         lam = ", pows[norm_diff.index(min(norm_diff))], "norm(A-A_r) = ", min(norm_diff))


if __name__ == "__main__":
    # main()
    # lambda_sampling()
    # lambda_test()
    chi_exponent_sampling()
