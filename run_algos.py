from pyspark import SparkContext
import sys
import os

import numpy as np
import multiprocessing as mp
from functools import partial
import pandas as pd
import time

import itertools
#import numba

import numpy as np
def BF_sgnchange(y,doFind = 0):
    if doFind == 0:
        return (np.multiply(y[1:],y[0:len(y)-1]) < 0)
    indexs = np.where((np.multiply(y[1:],y[0:len(y)-1]) < 0))
    return indexs

def BF_makeBuffer(y, bufferSize):

    N = len(y)

    numBuffers = int(np.floor(N / bufferSize))

    y_buffer = y[0:numBuffers*bufferSize]

    y_buffer = y_buffer.reshape((numBuffers,bufferSize))

    return y_buffer

def BF_embed(y,tau = 1,m = 2,makeSignal = 0,randomSeed = [],beVocal = 0):

    N = len(y)

    N_embed = N - (m - 1)*tau

    if N_embed <= 0:
        raise Exception('Time Series (N = %u) too short to embed with these embedding parameters')
    y_embed = np.zeros((N_embed,m))

    for i in range(1,m+1):

        y_embed[:,i-1] = y[(i-1)*tau:N_embed+(i-1)*tau]
    return(y_embed)

def BF_iszscored(x):
    numericThreshold = 2.2204E-16
    iszscored = ((np.absolute(np.mean(x)) < numericThreshold) & (np.absolute(np.std(x)-1) < numericThreshold))
    return(iszscored)



#@numba.jit(nopython=True,parallel=True)
def EN_PermEn(y,m = 2,tau = 1):

    x = BF_embed(y,tau,m)


    Nx = x.shape[0]

    permList = perms(m)
    numPerms = len(permList)

    countPerms = np.zeros(numPerms)


    for j in range(Nx):
        ix = np.argsort(x[j,:])

        for k in range(numPerms):
            if not (permList[k,:] - ix).all() :
                countPerms[k] = countPerms[k] + 1
                break

    p = countPerms / Nx
    p_0 = p[p > 0]
    permEn = -sum(np.multiply(p_0,np.log2(p_0)))



    mFact = math.factorial(m)
    normPermEn = permEn / np.log2(mFact)

    out = {'permEn':permEn,'normPermEn':normPermEn}

    return out

def perms(n):
    permut = itertools.permutations(np.arange(n))
    permut_array = np.empty((0,n))
    for p in permut:
        permut_array = np.append(permut_array,np.atleast_2d(p),axis=0)

    return(permut_array)

def DN_Moments(y,theMom = 1):
    if np.std(y) != 0:
        return stats.moment(y,theMom) / np.std(y)
    else:
        return 0

#@numba.jit(nopython=True,parallel=True)
def DN_Withinp(x,p = 1,meanOrMedian = 'mean'):
    N = len(x)

    if meanOrMedian == 'mean':
        mu = np.mean(x)
        sig = np.std(x)
    elif meanOrMedian == 'median':
        mu = np.median(x)
        sig = 1.35*stats.iqr(x)
    else:
        raise Exception('Unknown meanOrMedian should be mean or median')
    return np.sum((x >= mu-p*sig) & (x <= mu + p*sig)) / N

#@numba.jit(nopython=True)
#Quantile function seems to be slower with numba
def DN_Quantile(y,q = .5):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.quantile(y,q))

def DN_RemovePoints(y,removeHow = 'absfar',p = .99):

    if removeHow == 'absclose':
        i =  np.argsort(-np.absolute(y),kind = 'mergesort')
    elif removeHow == 'absfar':
        i = np.argsort(np.absolute(y),kind = 'mergesort')
    elif removeHow == 'min':
        i =  np.argsort(-y,kind = 'mergesort')
    elif removeHow == 'max':
        i = np.argsort(y,kind = 'mergesort')

    N = len(y)

    out = {}

    rKeep = np.sort(i[0:int(np.round(N*(1-p)))])
    y_trim = y[rKeep]

    #print(rKeep)


    acf_y = SUB_acf(y,8)
    acf_y_trim = SUB_acf(y_trim,8)

    out['fzcacrat'] = CO_FirstZero(y_trim,'ac')/CO_FirstZero(y,'ac')

    out['ac1rat'] = acf_y_trim[1]/acf_y[1]

    out['ac1diff'] = np.absolute(acf_y_trim[1]-acf_y[1])

    out['ac2rat'] = acf_y_trim[2]/acf_y[2]

    out['ac2diff'] = np.absolute(acf_y_trim[2]-acf_y[2])

    out['ac3rat'] = acf_y_trim[3]/acf_y[3]

    out['ac3diff'] = np.absolute(acf_y_trim[3]-acf_y[3])

    out['sumabsacfdiff'] = sum(np.absolute(acf_y_trim-acf_y))

    out['mean'] = np.mean(y_trim)

    out['median'] = np.median(y_trim)

    out['std'] = np.std(y_trim)

    if stats.skew(y) != 0:
        out['skewnessrat'] = stats.skew(y_trim)/stats.skew(y)

    out['kurtosisrat'] = stats.kurtosis(y_trim)/stats.kurtosis(y)

    return out

def SUB_acf(x,n):
    acf = np.zeros(n)
    for i in range(n):
        acf[i] = CO_AutoCorr(x,i-1,'Fourier')
    return acf

def DN_OutlierInclude(y,thresholdHow='abs',inc=.01):
    if not BF_iszscored(y):
        muhat, sigmahat = stats.norm.fit(y)
        y = (y - muhat) / sigmahat
        #warnings.warn('DN_OutlierInclude y should be z scored. So just converted y to z-scores')
    N = len(y)
    if thresholdHow == 'abs':
        thr = np.arange(0,np.max(np.absolute(y)),inc)
        tot = N
    if thresholdHow == 'p':
        thr = np.arange(0,np.max(y),inc)
        tot = sum( y >= 0)
    if thresholdHow == 'n':
        thr = np.arange(0,np.max(-y),inc)
        tot = sum( y <= 0)
    msDt = np.zeros((len(thr),6))
    for i in range(len(thr)):
        th = thr[i]

        if thresholdHow == 'abs':
            r = np.where(np.absolute(y) >= th)
        if thresholdHow == 'p':
            r = np.where(y >= th)
        if thresholdHow == 'n':
            r = np.where(y <= -th)

        Dt_exc = np.diff(r)

        msDt[i,0] = np.mean(Dt_exc)
        msDt[i,1] = np.std(Dt_exc) / np.sqrt(len(r))
        msDt[i,2] = len(Dt_exc) / tot * 100
        msDt[i,3] = np.median(r) / (N/2) - 1
        msDt[i,4] = np.mean(r) / (N/2) -1
        msDt[i,5] = np.std(r) / np.sqrt(len(r))

        return msDt

#@numba.jit(nopython=True,parallel=True)
def DN_Burstiness(y):
    r = np.std(y) / y.mean()
    B = ( r - 1 ) / ( r + 1 )
    return(B)

#@numba.jit(nopython=True,parallel=True)
#oddly this function slows down with numba
def DN_pleft(y,th = .1):

    p  = np.quantile(np.absolute(y - np.mean(y)),1-th)


    return p / np.std(y)

def CO_FirstZero(y,corrFun = 'ac'):
    acf = CO_AutoCorr(y,[],'Fourier')
    N = len(y)
    for i in range(1,N-1):
        if acf[i] < 0:
            return i
    return N

def DN_Fit_mle(y,fitWhat = 'gaussian'):
    if fitWhat == 'gaussian':
        phat = stats.norm.fit(y)
        out = {'mean':phat[0],'std':phat[1]}
        return out
    else:
        print('Use gaussian geometric not implemented yet')

def CO_FirstMin(y, minWhat = 'ac'):
    if minWhat == 'mi':
        x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
        acf = IN_AutoMutualInfo(y,x,'gaussian')
    else:
        acf = CO_AutoCorr(y,[],'Fourier')
    N = len(y)

    for i in range(1,N-1):
        if i == 2 and (acf[2] > acf[1]):
            return 1
        elif (i > 2) and (acf[i-2] > acf[i-1]) and (acf[i-1] < acf[i]):
            return i-1
    return N

def DN_IQR(y):
    return stats.iqr(y)

def DN_CompareKSFit(x,whatDist = 'norm'):
    xStep = np.std(x) / 100
    if whatDist == 'norm':
        a, b = stats.norm.fit(x)
        peak = stats.norm.pdf(a,a,b)
        thresh = peak / 100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(x)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2,a,b)


    #since some outliers real far away can take long time
    #should probably do pre-proccessing before functions
    if whatDist == "uni":

        a,b = stats.uniform.fit(x)
        peak = stats.uniform.pdf(np.mean(x),a,b-a)
        thresh = peak / 100
        xf1 = np.mean(x)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.norm.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(x)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.norm.pdf(xf2,a,b)

    #might over write y since changing x
    if whatDist == 'beta':
        scaledx = (x - np.min(x) + .01*np.std(x)) / (np.max(x)-np.min(x)+.02*np.std(x))
        xStep = np.std(scaledx) /100
        a = stats.beta.fit(scaledx)
        b = a[2]
        a = a[1]
        thresh = 1E-5
        xf1 = np.mean(scaledx)
        ange = 10
        while ange > thresh:
            xf1 = xf1 - xStep
            ange = stats.beta.pdf(xf1,a,b)
        ange = 10
        xf2 = np.mean(scaledx)
        while ange > thresh:
            xf2 = xf2 + xStep
            ange = stats.beta.pdf(xf2,a,b)
        x = scaledx


    kde = stats.gaussian_kde(x)
    test_space = np.linspace(np.min(x),np.max(x),1000)
    kde_est = kde(test_space)
    if whatDist == 'norm':
        ffit = stats.norm.pdf(test_space,a,b)
    if whatDist == 'uni':
        ffit = stats.uniform.pdf(test_space,a,b-a)
    if whatDist == 'beta':
        ffit = stats.beta.pdf(test_space,a,b)

    out = {}

    out['adiff'] = sum(np.absolute(kde_est - ffit)*(test_space[1]-test_space[0]))

    out['peaksepy'] = np.max(ffit) - np.max(kde_est)

    r = (ffit != 0)

    out['relent'] = sum(np.multiply(kde_est[r],np.log(np.divide(kde_est[r],ffit[r])))*(test_space[1]-test_space[0]))

    return out

from scipy import stats
def DN_Mode(y):
    #y must be numpy array
    if not isinstance(y,np.ndarray):
        y = np.asarray(y)
    return float(stats.mode(y).mode)

import numpy as np
#import numba

#@numba.jit(nopython=True,parallel=True)
def EN_SampEn(x,m=2,r=.2,scale=True):
    if scale:
        r = np.std(x) * r
    templates = make_templates(x,m)
    A = 0
    B = 0
    for i in range(templates.shape[0]):
        template = templates[i,:]
        A = A + np.sum(np.amax(np.abs(templates-template), axis=1) < r) -1
        B = B + np.sum(np.amax(np.absolute(templates[:,0:m]-template[0:m]),axis=1) < r) - 1
    return {'Sample Entropy':- np.log(A/B),"Quadratic Entropy": - np.log(A/B) + np.log(2*r)}
#@numba.jit(nopython=True,parallel=True)
def make_templates(x,m):
    N = int(len(x) - (m))
    templates = np.zeros((N,m+1))
    for i in range(N):
        templates[i,:] = x[i:i+m+1]
    return templates
# def EN_SampEn(y,M = 2,r = 0,pre = ''):
#     if r == 0:
#         r = .1*np.std(y)
#     else:
#         r = r*np.std(y)
#     M = M + 1
#     N = len(y)
#     lastrun = np.zeros(N)
#     run = np.zeros(N)
#     A = np.zeros(M)
#     B = np.zeros(M)
#     p = np.zeros(M)
#     e = np.zeros(M)
#
#     for i in range(1,N):
#         y1 = y[i-1]
#
#         for jj in range(1,N-i + 1):
#
#             j = i + jj - 1
#
#             if np.absolute(y[j] - y1) < r:
#
#                 run[jj] = lastrun[jj] + 1
#                 M1 = min(M,run[jj])
#                 for m in range(int(M1)):
#                     A[m] = A[m] + 1
#                     if j < N:
#                         B[m] = B[m] + 1
#             else:
#                 run[jj] = 0
#         for j in range(N-1):
#             lastrun[j] = run[j]
#
#     NN = N * (N - 1) / 2
#     p[0] = A[0] / NN
#     e[0] = - np.log(p[0])
#     for m in range(1,int(M)):
#         p[m] = A[m] / B[m-1]
#         e[m] = -np.log(p[m])
#     i = 0
#     out = {'sampen':np.zeros(len(e)),'quadSampEn':np.zeros(len(e))}
#     for ent in e:
#         quaden1 = ent + np.log(2*r)
#         out['sampen'][i] = ent
#         out['quadSampEn'][i] = quaden1
#         i = i + 1
#
#     return out

from scipy import signal
def SY_Trend(y):

    N  = len(y)
    stdRatio = np.std(signal.detrend(y)) / np.std(y)

    gradient, intercept = LinearFit(np.arange(N),y)

    yC = np.cumsum(y)
    meanYC = np.mean(yC)
    stdYC = np.std(yC)

    #print(gradient)
    #print(intercept)

    gradientYC, interceptYC = LinearFit(np.arange(N),yC)

    meanYC12 = np.mean(yC[0:int(np.floor(N/2))])
    meanYC22 = np.mean(yC[int(np.floor(N/2)):])

    out = {'stdRatio':stdRatio,'gradient':gradient,'intercept':intercept,
            'meanYC':meanYC,'stdYC':stdYC,'gradientYC':gradientYC,
            'interceptYC':interceptYC,'meanYC12':meanYC12,'meanYC22':meanYC22}
    return out

def LinearFit(xData,yData):
    m, b = np.polyfit(xData,yData,1)
    return m,b

import numpy as np


#@numba.jit(nopython=True,parallel=True)
def DN_Mean(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(y.mean())

def CO_glscf(y,alpha = 1.0,beta = 1.0,tau = ''):
    if tau == '':
        tau = CO_FirstZero(y,'ac')
    N = len(y)
    beta = float(beta)
    alpha = float(alpha)
    y1 = np.absolute(y[0:N-tau])
    y2 = np.absolute(y[tau:N])
    top = np.mean(np.multiply(np.power(y1,alpha),np.power(y2,beta))) - np.mean(np.power(y1,alpha)) * np.mean(np.power(y2,beta))
    bot =  np.sqrt(np.mean(np.power(y1,2*alpha)) - np.mean(np.power(y1,alpha))**2) * np.sqrt(np.mean(np.power(y2,2*beta)) - np.mean(np.power(y2,beta))**2)
    if bot == 0:
        return np.inf
    glscf = top / bot
    return glscf

def DN_Cumulants(y,cumWhatMay = 'skew1'):
    if cumWhatMay == 'skew1':
        return stats.skew(y)
    elif cumWhatMay == 'skew2':
        return stats.skew(y,0)
    elif cumWhatMay == 'kurt1':
        return stats.kurtosis(y)
    elif cumWhatMay == 'kurt2':
        return stats.kurtosis(y,0)
    else:
         raise Exception('Requested Unknown cumulant must be: skew1, skew2, kurt1, or kurt2')

def DN_Range(y):
    return np.max(y) - np.min(y)


def DN_FitKernalSmooth(x,varargin = {}):
    #varargin should be dict with possible keys numcross
    #area and arclength

    out = {}

    m = np.mean(x)

    kde = stats.gaussian_kde(x)
    #i think matlabs kde uses 100 points
    #but end numbers end up being midly off
    #seems to be rounding entropy max, min line up
    test_space = np.linspace(np.min(x),np.max(x),100)

    f = kde(test_space)

    df = np.diff(f)

    ddf  = np.diff(df)

    sdsp = ddf[BF_sgnchange(df,1)]

    out['npeaks'] = sum(sdsp < -.0002)

    out['max'] = np.max(f)

    out['entropy'] = - sum(np.multiply(f[f>0],np.log(f[f>0])))*(test_space[2]-test_space[1])

    out1 = sum(f[test_space > m]) * (test_space[2]-test_space[1])
    out2 = sum(f[test_space < m]) * (test_space[2]-test_space[1])
    out['asym'] = out1 / out2

    out1 = sum(np.absolute(np.diff(f[test_space < m]))) * (test_space[2]-test_space[1])
    out1 = sum(np.absolute(np.diff(f[test_space > m]))) * (test_space[2]-test_space[1])
    out['plsym'] = out1 / out2

    if 'numcross' in varargin:
        thresholds = varargin['numcross']
        out['numCrosses']  = {}
        for i in range(len(thresholds)):
            numCrosses = sum(BF_sgnchange(f - thresholds[i]))
            out['numCrosses'][thresholds[i]] = numCrosses
    if 'area' in varargin:
        thresholds = varargin['area']
        out['area']  = {}
        for i in range(len(thresholds)):
            areaHere = sum(f[f < thresholds[i]]) * (test_space[2]-test_space[1])
            out['area'][thresholds[i]] = areaHere
    if 'arclength' in varargin:
        thresholds = varargin['arclength']
        out['arclength']  = {}
        for i in range(len(thresholds)):
            fd = np.absolute(np.diff(f[(test_space > m - thresholds[i]) & (test_space < m + thresholds[i])]))
            arclengthHere = sum(fd) * (test_space[2]-test_space[1])
            out['arclength'][thresholds[i]] = arclengthHere
    return out

import numpy as np
#@numba.jit(nopython=True)
def DN_Median(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.median(y))

#@numba.jit(nopython=True,parallel=True)
def DN_Spread(y,spreadMeasure = 'std'):
    if spreadMeasure == 'std':
        return np.std(y)
    elif spreadMeasure == 'iqr':
        return stats.iqr(y)
    elif spreadMeasure == 'mad':
        return mad(y)
    elif spreadMeasure == 'mead':
        return mead(y)#stats.median_absolute_deviation(y)
    else:
        raise Exception('spreadMeasure must be one of std, iqr, mad or mead')
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def mead(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)

#@numba.jit(nopython=True,parallel=True)
def DN_MinMax(y,which = 'max'):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    if which == 'min':
        return(y.min())
    else:
        return(y.max())

#@numba.jit(nopython=True,parallel=True)
def DN_CustomSkewness(y,whatSkew = 'pearson'):
    if whatSkew == 'pearson':
        if np.std(y) != 0:
            return (3*np.mean(y) - np.median(y)) / np.std(y)
        else:
            return 0
    elif whatSkew == 'bowley':
        qs = np.quantile(y,[.25,.5,.75])
        if np.std(y) != 0:
            return (qs[2] + qs[0] - 2*qs[1]) / (qs[2] - qs[0])
        else:
            return 0

    else:
         raise Exception('whatSkew must be either pearson or bowley.')

def EN_mse(y,scale=range(2,11),m=2,r=.15,adjust_r=True):

    minTSLength = 20
    numscales = len(scale)
    y_cg = []

    for i in range(numscales):
        bufferSize = scale[i]
        y_buffer = BF_makeBuffer(y,bufferSize)
        y_cg.append(np.mean(y_buffer,1))

    outEns = []

    for si in range(numscales):
        if len(y_cg[si]) >= minTSLength:

            sampEnStruct = EN_SampEn(y_cg[si],m,r)
            outEns.append(sampEnStruct)
        else:
            outEns.append(np.nan)
    sampEns = []
    for out in outEns:
        sampEns.append(out['Sample Entropy'])

    maxSampen = np.max(sampEns)
    maxIndx = np.argmax(sampEns)

    minSampen = np.min(sampEns)
    minIndx = np.argmin(sampEns)

    meanSampen = np.mean(sampEns)

    stdSampen = np.std(sampEns)

    meanchSampen = np.mean(np.diff(sampEns))

    out = {'sampEns':sampEns,'max Samp En':maxSampen,'max point':scale[maxIndx],'min Samp En':minSampen,\
    'min point':scale[minIndx],'mean Samp En':meanSampen,'std Samp En':stdSampen, 'Mean Change':meanchSampen}

    return out

def IN_AutoMutualInfo(y,timeDelay = 1,estMethod = 'gaussian',extraParam = []):
    if isinstance(timeDelay,str):
        timeDelay = CO_FirstZero(y,'ac')
    N = len(y)

    if isinstance(timeDelay,list):
        numTimeDelays = len(timeDelay)
    else:
        numTimeDelays = 1
        timeDelay = [timeDelay]
    amis = []
    out = {}
    for k in range(numTimeDelays):
        y1 = y[0:N-timeDelay[k]]
        y2 = y[timeDelay[k]:N]
        if estMethod == 'gaussian':
            r = np.corrcoef(y1,y2)[1,0]
            amis.append(-.5 * np.log(1 - r**2))
            out['Auto Mutual ' + str(timeDelay[k])] = -.5 * np.log(1 - r**2)

    return out

def EN_CID(y):


    CE1 = f_CE1(y)
    CE2 = f_CE2(y)

    minCE1 = f_CE1(np.sort(y))
    minCE2 = f_CE2(np.sort(y))

    CE1_norm = CE1 / minCE1
    CE2_norm = CE2 / minCE2

    out = {'CE1':CE1,'CE2':CE2,'minCE1':minCE1,'minCE2':minCE2,
            'CE1_norm':CE1_norm,'CE2_norm':CE2_norm}
    return out

def f_CE1(y):
    return np.sqrt(np.mean( 1 + np.power(np.diff(y),2) ) )

def f_CE2(y):
    return np.mean(np.sqrt( 1 + np.power(np.diff(y),2) ) )

def DN_Unique(x):
    return len(np.unique(x)) / len(x)

from scipy import optimize
def DT_IsSeasonal(y):

    N = len(y)

    th_fit = 0.3
    th_ampl = 0.5

    try:
        params, params_covariance = optimize.curve_fit(test_func, np.arange(N), y, p0=[10, 13,600,0])
    except:
        return False

    a,b,c,d = params



    y_pred = a * np.sin(b * np.arange(N) + d) + c

    SST = sum(np.power(y - np.mean(y),2))
    SSr = sum(np.power(y - y_pred,2))

    R = 1 - SSr / SST


    if R > th_fit: #and (np.absolute(a) > th_ampl*.1*np.std(y)):
        return True
    else:
        return False

def test_func(x, a, b,c,d):
    return a * np.sin(b * x + d) + c

#@numba.jit(nopython=True,parallel=True)
def EN_ApEn(y,mnom = 1,rth = .2):

    r = rth * np.std(y)
    N = len(y)
    phi = np.zeros(2)

    for k in range(2):
        m = mnom + k
        m = int(m)
        C = np.zeros(N-m+1)

        x = np.zeros((N - m + 1, m))

        for i in range(N - m + 1):
            x[i,:] = y[i:i+m]

        ax = np.ones((N - m + 1, m))
        for i in range(N-m+1):

            for j in range(m):
                ax[:,j] = x[i,j]

            d = np.absolute(x-ax)
            if m > 1:
                d = np.maximum(d[:,0],d[:,1])
            dr = ( d <= r )
            C[i] = np.sum(dr) / (N-m+1)
        phi[k] = np.mean(np.log(C))
    return phi[0] - phi[1]

#import matplotlib.pyplot as plt
def SC_HurstExp(x):

    N = len(x)

    splits = int(np.log2(N))

    rescaledRanges = []

    n = []

    for i in range(splits):

        chunks = 2**(i)

        n.append(int(N / chunks))


        y = x[:N - N % chunks]

        y = y.reshape((chunks,int(N/chunks)))

        m = y.mean(axis = 1,keepdims = True)

        y = y - m

        z = np.cumsum(y,1)

        R = np.max(z,1) - np.min(z,1)

        S = np.std(y,1)

        S[S == 0] = 1


        rescaledRanges.append(np.mean(R/S))

    logRS = np.log(rescaledRanges)
    logn = np.log(n)

    # plt.plot(logn,logRS)
    # plt.show()

    p = np.polyfit(logn,logRS,1)

    return p[0]

def DN_ObsCount(y):
    return np.count_nonzero(~np.isnan(y))

#@numba.jit(nopython=True,parallel=True)
def EN_ShannonEn(y):
    p = np.zeros(len(np.unique(y)))
    n = 0
    for i in np.unique(y):
        p[n] = len(y[y == i]) / len(y)
        n = n + 1

    return -np.sum(p*np.log2(p))

# author: Dominik Krzeminski (dokato)
# https://github.com/dokato/dfa
import numpy as np

import scipy.signal as ss

# detrended fluctuation analysis
def CO_Embed2_Basic(y,tau = 5,scale = 1):
    '''
    CO_Embed2_Basic Point density statistics in a 2-d embedding space
    %
    % Computes a set of point density measures in a plot of y_i against y_{i-tau}.
    %
    % INPUTS:
    % y, the input time series
    %
    % tau, the time lag (can be set to 'tau' to set the time lag the first zero
    %                       crossing of the autocorrelation function)
    %scale since .1 and .5 don't make sense for HR RESP ...
    % Outputs include the number of points near the diagonal, and similarly, the
    % number of points that are close to certain geometric shapes in the y_{i-tau},
    % y_{tau} plot, including parabolas, rings, and circles.
    '''
    if tau == 'tau':

        tau = CO_FirstZero(y,'ac')

    xt = y[:-tau]
    xtp = y[tau:]
    N = len(y) - tau

    outDict = {}

    outDict['updiag01'] = np.sum((np.absolute(xtp - xt) < 1*scale)) / N
    outDict['updiag05'] = np.sum((np.absolute(xtp - xt) < 5*scale)) / N

    outDict['downdiag01'] = np.sum((np.absolute(xtp + xt) < 1*scale)) / N
    outDict['downdiag05'] = np.sum((np.absolute(xtp + xt) < 5*scale)) / N

    outDict['ratdiag01'] = outDict['updiag01'] / outDict['downdiag01']
    outDict['ratdiag05'] = outDict['updiag05'] / outDict['downdiag05']

    outDict['parabup01'] = np.sum( ( np.absolute( xtp - np.square(xt) ) < 1 * scale ) ) / N
    outDict['parabup05'] = np.sum( ( np.absolute( xtp - np.square(xt) ) < 5 * scale ) ) / N

    outDict['parabdown01'] = np.sum( ( np.absolute( xtp + np.square(xt) ) < 1 * scale ) ) / N
    outDict['parabdown05'] = np.sum( ( np.absolute( xtp + np.square(xt) ) < 5 * scale ) ) / N

    outDict['parabup01_1'] = np.sum( ( np.absolute( xtp -( np.square(xt) + 1 ) ) < 1 * scale ) ) / N
    outDict['parabup05_1'] = np.sum( ( np.absolute( xtp - (np.square(xt) + 1)) < 5 * scale ) ) / N

    outDict['parabdown01_1'] = np.sum( ( np.absolute( xtp + np.square(xt) - 1 ) < 1 * scale ) ) / N
    outDict['parabdown05_1'] = np.sum( ( np.absolute( xtp + np.square(xt) - 1) < 5 * scale ) ) / N

    outDict['parabup01_n1'] = np.sum( ( np.absolute( xtp -( np.square(xt) - 1 ) ) < 1 * scale ) ) / N
    outDict['parabup05_n1'] = np.sum( ( np.absolute( xtp - (np.square(xt) - 1)) < 5 * scale ) ) / N

    outDict['parabdown01_n1'] = np.sum( ( np.absolute( xtp + np.square(xt) + 1 ) < 1 * scale ) ) / N
    outDict['parabdown05_n1'] = np.sum( ( np.absolute( xtp + np.square(xt) + 1) < 5 * scale ) ) / N

    outDict['ring1_01'] = np.sum( np.absolute( np.square(xtp) + np.square(xt) - 1 ) < 1 * scale ) / N
    outDict['ring1_02'] = np.sum( np.absolute( np.square(xtp) + np.square(xt) - 1 ) < 2 * scale ) / N
    outDict['ring1_05'] = np.sum( np.absolute( np.square(xtp) + np.square(xt) - 1 ) < 5 * scale ) / N

    outDict['incircle_01'] = np.sum(  np.square(xtp) + np.square(xt)  < 1 * scale ) / N
    outDict['incircle_02'] = np.sum( np.square(xtp) + np.square(xt)  < 2 * scale ) / N
    outDict['incircle_05'] = np.sum(  np.square(xtp) + np.square(xt)  < 5 * scale ) / N


    outDict['incircle_1'] = np.sum(  np.square(xtp) + np.square(xt)  < 10 * scale ) / N
    outDict['incircle_2'] = np.sum( np.square(xtp) + np.square(xt)  < 20 * scale ) / N
    outDict['incircle_3'] = np.sum(  np.square(xtp) + np.square(xt)  < 30 * scale ) / N

    outDict['medianincircle'] = np.median([outDict['incircle_01'], outDict['incircle_02'], outDict['incircle_05'], \
                            outDict['incircle_1'], outDict['incircle_2'], outDict['incircle_3']])

    outDict['stdincircle'] = np.std([outDict['incircle_01'], outDict['incircle_02'], outDict['incircle_05'], \
                            outDict['incircle_1'], outDict['incircle_2'], outDict['incircle_3']],ddof = 1)


    return outDict
def calc_rms(x, scale):
    """
    windowed Root Mean Square (RMS) with linear detrending.

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale* : int
        length of the window in which RMS will be calculaed
    Returns:
    --------
      *rms* : numpy.array
        RMS data in each window with length len(x)//scale
    """
    # making an array with data divided in windows
    shape = (x.shape[0]//scale, scale)
    X = np.lib.stride_tricks.as_strided(x,shape=shape)
    # vector of x-axis points to regression
    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeff = np.polyfit(scale_ax, xcut, 1)
        xfit = np.polyval(coeff, scale_ax)
        # detrending and computing RMS of each window
        rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
    return rms

def dfa(x, scale_lim=[5,9], scale_dens=0.25, show=False):
    """
    Detrended Fluctuation Analysis - measures power law scaling coefficient
    of the given signal *x*.

    More details about the algorithm you can find e.g. here:
    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free
    view on neuronal oscillations, (2012).

    Args:
    -----
      *x* : numpy.array
        one dimensional data vector
      *scale_lim* = [5,9] : list of length 2
        boundaries of the scale, where scale means windows among which RMS
        is calculated. Numbers from list are exponents of 2 to the power
        of X, eg. [5,9] is in fact [2**5, 2**9].
        You can think of it that if your signal is sampled with F_s = 128 Hz,
        then the lowest considered scale would be 2**5/128 = 32/128 = 0.25,
        so 250 ms.
      *scale_dens* = 0.25 : float
        density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ]
      *show* = False
        if True it shows matplotlib log-log plot.
    Returns:
    --------
      *scales* : numpy.array
        vector of scales (x axis)
      *fluct* : numpy.array
        fluctuation function values (y axis)
      *alpha* : float
        estimation of DFA exponent
    """
    # cumulative sum of data with substracted offset
    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))
    # computing RMS for each window
    for e, sc in enumerate(scales):
        if len(calc_rms(y, sc)**2) == 0:
            continue
        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc)**2))

    # fitting a line to rms data
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    # if show:
    #     fluctfit = 2**np.polyval(coeff,np.log2(scales))
    #     plt.loglog(scales, fluct, 'bo')
    #     plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f'%coeff[0])
    #     plt.title('DFA')
    #     plt.xlabel(r'$\log_{10}$(time window)')
    #     plt.ylabel(r'$\log_{10}$<F(t)>')
    #     plt.legend()
    #     plt.show()
    #return scales, fluct, coeff[0]
    return coeff[0]

def CO_tc3(y,tau = 'ac'):
    if tau == 'ac':
        tau = CO_FirstZero(y,'ac')
    else:
        tau = CO_FirstMin(y,'mi')

    N = len(y)
    yn = y[0:N-2*tau]
    yn1 = y[tau:N-tau]
    yn2 = y[tau*2:N]
    try:
        raw = np.mean(np.multiply(np.multiply(yn,yn1),yn2)) / (np.absolute(np.mean(np.multiply(yn,yn1))) ** (3/2))
    except:
        raw = np.nan
    return raw

def DN_nlogL_norm(y):
    muhat, sigmahat = stats.norm.fit(y)
    z = (y - muhat) / sigmahat
    L = -.5*np.power(z,2) - np.log(np.sqrt(2*math.pi)*sigmahat)
    return -sum(L) / len(y)


def CO_AutoCorr(y,lag = 1,method = 'TimeDomianStat',t=1):
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    if method == 'TimeDomianStat':
        if lag == []:
            acf = [1]
            for i in range(1,len(y)-1):
                acf.append(np.corrcoef(y[:-lag],y[lag:])[0,1])
            return acf
        return(np.corrcoef(y[:-lag],y[lag:])[0,1])
    else:
        N = len(y)
        nFFT = int(2**(np.ceil(np.log2(N)) + 1))
        F = np.fft.fft(y - y.mean(),nFFT)
        F = np.multiply(F,np.conj(F))
        acf = np.fft.ifft(F)
        if acf[0] == 0:
            if lag == []:
                return acf
            return acf[lag]


        acf = acf / acf[0]
        acf = acf.real
        if lag == []:
            return acf
        return acf[lag]

import math
def CO_f1ecac(y):
    N = len(y)
    thresh = 1 / math.exp(1)
    for i in range(1,N):
        auto = CO_AutoCorr(y,i)
        if ( auto - thresh ) < 0:
            return i
    return N

def DN_ProportionValues(x,propWhat = 'positive'):
    N = len(x)
    if propWhat == 'zeros':
        return sum(x == 0) / N
    elif propWhat == 'positive':
        return sum(x > 0) / N
    elif propWhat == 'negative':
        return sum(x < 0) / N
    else:
        raise Exception('Only negative, positve, zeros accepted for propWhat.')

#@numba.jit(nopython=True,parallel=True)
def DN_STD(y):
    #y must be numpy array
    # if not isinstance(y,np.ndarray):
    #     y = np.asarray(y)
    return(np.std(y))

def CO_trev(y,tau = 'ac'):
        if tau == 'ac':
            tau = CO_FirstZero(y,'ac')
        else:
            tau = CO_FirstMin(y,'mi')
        N = len(y)
        yn = y[0:N-tau]
        yn1 = y[tau:N]

        raw = np.mean(np.power(yn1-yn,3)) / np.mean(np.power(yn1 - yn,2))**(3/2)

        return raw

#import warnings
#@numba.jit(nopython=True,parallel=True)
def DN_cv(x,k = 1):
    # if k % 1 != 0 or k < 0:
    #     warnings.warn("k should probably be positive int")
    return (np.std(x)**k) / (np.mean(x)**k)

#@numba.jit(nopython=True,parallel=True)
def DN_TrimmedMean(y,n = 0):
    N = len(y)
    trim = int(np.round(N * n / 2))
    y = np.sort(y)
    #return stats.trim_mean(y,n) doesn't agree with matlab
    return np.mean(y[trim:N-trim])

def SC_DFA(y):

    N = len(y)

    tau = int(np.floor(N/2))

    y = y - np.mean(y)

    x = np.cumsum(y)

    taus = np.arange(5,tau+1)

    ntau = len(taus)

    F = np.zeros(ntau)

    for i in range(ntau):

        t = int(taus[i])



        x_buff = x[:N - N % t]

        x_buff = x_buff.reshape((int(N / t),t))


        y_buff = np.zeros((int(N / t),t))

        for j in range(int(N / t)):

            tt = range(0,int(t))

            p = np.polyfit(tt,x_buff[j,:],1)

            y_buff[j,:] =  np.power(x_buff[j,:] - np.polyval(p,tt),2)



        y_buff.reshape((N - N % t,1))

        F[i] = np.sqrt(np.mean(y_buff))

    logtaur = np.log(taus)

    logF = np.log(F)

    p = np.polyfit(logtaur,logF,1)

    return p[0]

#@numba.jit(nopython=True,parallel=True)
def DN_HighLowMu(y):
    mu = np.mean(y)
    mhi = np.mean(y[y>mu])
    mlo = np.mean(y[y<mu])
    return (mhi - mu) / (mu - mlo)

def read_in_data(id):
    id = str(id).zfill(4)
    data = np.genfromtxt('/Users/justinniestroy-admin/Documents/Work/Randall Data/houlter data/HR/UVA' + id +'_hr.csv', delimiter=',')
    time = data[:,0]
    hr = data[:,1]
    return(time,hr)




def run_histogram_algos(y,algos = 'all',results = {},impute = False):

    if impute:
        y = impute(y)
    else:
        y = y[~np.isnan(y)]

    if 'DN_Mean' in algos:
        results['DN_Mean'] = DN_Mean(y)

    if 'DN_Range' in algos:
        results['DN_Range'] = DN_Range(y)

    if 'DN_IQR' in algos:
        results['DN_IQR'] = DN_IQR(y)

    if 'DN_Median' in algos:
        results['DN_Median'] = DN_Median(y)

    if 'DN_MinMax' in algos:
        results['DN_Max'] = DN_MinMax(y)
        results['DN_Min'] = DN_MinMax(y,'min')

    if 'DN_Mode' in algos:
        results['DN_Mode'] = DN_Mode(y)

    if 'DN_Cumulants' in algos:
        results['DN_skew1'] = DN_Cumulants(y,'skew1')
        results['DN_skew2'] = DN_Cumulants(y,'skew2')
        results['DN_kurt1'] = DN_Cumulants(y,'kurt1')
        results['DN_kurt2'] = DN_Cumulants(y,'kurt2')

    if 'DN_Burstiness' in algos:
        results['DN_Burstiness'] = DN_Burstiness(y)

    if 'DN_Unique' in algos:
        results['DN_Unique'] = DN_Unique(y)

    if 'DN_Withinp' in algos:
        results['DN_Withinp 1'] = DN_Withinp(y)
        results['DN_Withinp 2'] = DN_Withinp(y,2)

    if 'EN_ShannonEn':
        results['EN_ShannonEntropy'] = EN_ShannonEn(y)

    if 'DN_STD' in algos:
        results['DN_std'] = DN_STD(y)
        if results['DN_std'] == 0:
            return results

    if 'DN_Moments' in algos:
        results['DN_Moment 2'] = DN_Moments(y,2)
        results['DN_Moment 3'] = DN_Moments(y,3)
        results['DN_Moment 4'] = DN_Moments(y,4)
        results['DN_Moment 5'] = DN_Moments(y,5)
        results['DN_Moment 6'] = DN_Moments(y,6)

    if 'DN_pleft' in algos:
        results['DN_pleft'] = DN_pleft(y)

    if 'DN_CustomSkewness' in algos:
        results['DN_CustomSkewness'] = DN_CustomSkewness(y)

    if 'DN_HighLowMu' in algos:
        results['DN_HighLowMu'] = DN_HighLowMu(y)


    if 'DN_nlogL_norm' in algos:
        results['DN_nlogL_norm'] = DN_nlogL_norm(y)

    if 'DN_Quantile' in algos:
        results['DN_Quantile_50'] = DN_Quantile(y)
        results['DN_Quantile_75'] = DN_Quantile(y,.75)
        results['DN_Quantile_90'] = DN_Quantile(y,.90)
        results['DN_Quantile_95'] = DN_Quantile(y,.95)
        results['DN_Quantile_99'] = DN_Quantile(y,.99)

    if 'DN_RemovePoints' in algos:
        out = DN_RemovePoints(y,p = .5)
        results = parse_outputs(out,results,'DN_RemovePoints')

    if 'DN_Spread':
        results['DN_Spread_mad'] = DN_Spread(y,'mad')
        results['DN__Spread_mead'] = DN_Spread(y,'mead')

    if 'DN_TrimmedMean' in algos:
        results['DN_TrimmedMean 50'] = DN_TrimmedMean(y,.5)
        results['DN_TrimmedMean 75'] = DN_TrimmedMean(y,.75)
        results['DN_TrimmedMean 25'] = DN_TrimmedMean(y,.25)

    if 'DN_cv' in algos:
        results['DN_cv 1'] = DN_cv(y)
        results['DN_cv 2'] = DN_cv(y,2)
        results['DN_cv 3'] = DN_cv(y,3)

    return results

def DN_RemovePoints(y,removeHow = 'absfar',p = .1):

    if removeHow == 'absclose' or 'absclose' in removeHow:

        i =  np.argsort(-np.absolute(y),kind = 'mergesort')

    elif removeHow == 'absfar' or 'absfar' in removeHow:

        i = np.argsort(np.absolute(y),kind = 'mergesort')

    elif removeHow == 'min' or 'min' in removeHow:

        i =  np.argsort(-y,kind = 'mergesort')

    elif removeHow == 'max' or 'max' in removeHow:

        i = np.argsort(y,kind = 'mergesort')

    else:

        return

    N = len(y)

    out = {}

    rKeep = np.sort(i[0:int(np.round(N*(1-p)))])
    y_trim = y[rKeep]

    #print(rKeep)


    acf_y = SUB_acf(y,8)
    acf_y_trim = SUB_acf(y_trim,8)

    out['fzcacrat'] = CO_FirstZero(y_trim,'ac')/CO_FirstZero(y,'ac')

    out['ac1rat'] = acf_y_trim[0]/acf_y[0]

    out['ac1diff'] = np.absolute(acf_y_trim[0]-acf_y[0])

    out['ac2rat'] = acf_y_trim[1]/acf_y[1]

    out['ac2diff'] = np.absolute(acf_y_trim[1]-acf_y[1])

    out['ac3rat'] = acf_y_trim[2]/acf_y[2]

    out['ac3diff'] = np.absolute(acf_y_trim[2]-acf_y[2])

    out['sumabsacfdiff'] = sum(np.absolute(acf_y_trim-acf_y))

    out['mean'] = np.mean(y_trim)

    out['median'] = np.median(y_trim)

    out['std'] = np.std(y_trim,ddof = 1)

    if stats.skew(y) != 0:

        out['skewnessrat'] = stats.skew(y_trim)/stats.skew(y)
    try:

        out['kurtosisrat'] = stats.kurtosis(y_trim,fisher=False)/stats.kurtosis(y,fisher=False)

    except:

        pass

    return out
def BF_Binarize(y,binarizeHow = 'diff'):

    if binarizeHow == 'diff':

        yBin = stepBinary(np.diff(y))

    if binarizeHow == 'mean':

        yBin = stepBinary(y - np.mean(y))

    if binarizeHow == 'median':

        yBin = stepBinary(y - np.median(y))

    if binarizeHow == 'iqr':

        iqr = np.quantile(y,[.25,.75])

        iniqr = np.logical_and(y > iqr[0], y<iqr[1])

        yBin = np.zeros(len(y))

        yBin[iniqr] = 1

    return yBin

def stepBinary(X):

    Y = np.zeros(len(X))

    Y[ X > 0 ] = 1

    return Y
def DK_crinkle(x):

    x = x - np.mean(x)

    x2 = np.mean(np.square(x))**2

    if x2 == 0:
        return 0

    d2 = 2*x[1:-1] - x[0:-2] - x[2:]

    return np.mean(np.power(d2,4)) / x2

def DK_theilerQ(x):
    x2 = np.mean(np.square(x))**(3/2)

    if x2 == 0:
        return 0

    d2 = x[0:-1] + x[1:]
    Q = np.mean(np.power(d2,3)) / x2

    return Q

def SB_BinaryStats(y,binaryMethod = 'diff'):

    yBin = BF_Binarize(y,binaryMethod)

    N = len(yBin)

    outDict = {}

    outDict['pupstat2'] = np.sum((yBin[math.floor(N /2):] == 1))  / np.sum((yBin[:math.floor(N /2)] == 1))

    stretch1 = []
    stretch0 = []
    count = 1



    for i in range(1,N):

        if yBin[i] == yBin[i - 1]:

            count = count + 1

        else:

            if yBin[i - 1] == 1:

                stretch1.append(count)

            else:

                stretch0.append(count)
            count = 1
    if yBin[N-1] == 1:

        stretch1.append(count)

    else:

        stretch0.append(count)


    outDict['pstretch1'] = len(stretch1) / N

    if stretch0 == []:

        outDict['longstretch0'] = 0
        outDict['meanstretch0'] = 0
        outDict['stdstretch0'] = np.nan

    else:

        outDict['longstretch0'] = np.max(stretch0)
        outDict['meanstretch0'] = np.mean(stretch0)
        outDict['stdstretch0'] = np.std(stretch0,ddof = 1)

    if stretch1 == []:

        outDict['longstretch1'] = 0
        outDict['meanstretch1'] = 0
        outDict['stdstretch1'] = np.nan

    else:

        outDict['longstretch1'] = np.max(stretch1)
        outDict['meanstretch1'] = np.mean(stretch1)
        outDict['stdstretch1'] = np.std(stretch1,ddof = 1)

    try:

        outDict['meanstretchdiff'] = outDict['meanstretch1'] - outDict['meanstretch0']
        outDict['stdstretchdiff'] = outDict['stdstretch1'] - outDict['stdstretch0']

    except:

        outDict['meanstretchdiff'] = np.nan
        outDict['stdstretchdiff'] = np.nan


    return outDict
def SY_StatAv(y,whatType = 'seg',n = 5):

    N = len(y)

    if whatType == 'seg':

        M = np.zeros(n)
        p = math.floor(N/n)

        for j in range(1,n+1):

            M[j - 1] = np.mean(y[p*(j-1):p*j])

    elif whatType == 'len':

        if N > 2*n:

            pn = math.floor(N / n)
            M = np.zeros(pn)

            for j in range(1,pn + 1):

                M[j-1] = np.mean(y[(j-1)*n:j*n])

        else:

            return

    s = np.std(y,ddof = 1)
    sdav = np.std(M,ddof = 1)

    out = sdav / s

    return out
def EX_MovingThreshold(y,a = 1,b = .1):

    if b < 0 or b > 1:

        print("b must be between 0 and 1")
        return None

    N = len(y)
    y = np.absolute(y)
    q = np.zeros(N)
    kicks = np.zeros(N)

    q[0] = 1

    for i in range(1,N):

        if y[i] > q[i-1]:

            q[i] = (1 + a) * y[i]
            kicks[i] = q[i] - q[i-1]

        else:

            q[i] = ( 1 - b ) *  q[i-1]


    outDict = {}

    outDict['meanq'] = np.mean(q)
    outDict['medianq'] = np.median(q)
    outDict['iqrq'] = stats.iqr(q)
    outDict['maxq'] = np.max(q)
    outDict['minq'] = np.min(q)
    outDict['stdq'] = np.std(q)
    outDict['meanqover'] = np.mean(q - y)



    outDict['pkick'] = np.sum(kicks) / N - 1
    fkicks = np.argwhere(kicks > 0).flatten()
    Ikicks = np.diff(fkicks)
    outDict['stdkicks'] = np.std(Ikicks)
    outDict['meankickf'] = np.mean(Ikicks)
    outDict['mediankicksf'] = np.median(Ikicks)

    return outDict

def SY_DriftingMean(y,howl = 'num',l = ''):

    N = len(y)

    if howl == 'num':

        if l != '':

            l = math.floor(N/l)

    if l == '':

        if howl == 'num':

            l = 5

        elif howl == 'fix':

            l = 200

    if l == 0 or N < l:

        return

    numFits = math.floor(N / l)
    z = np.zeros((l,numFits))

    for i in range(0,numFits):

        z[:,i] = y[i*l :(i + 1)*l]



    zm = np.mean(z,axis = 0)
    zv = np.var(z,axis = 0,ddof = 1)

    meanvar = np.mean(zv)
    maxmean = np.max(zm)
    minmean = np.min(zm)
    meanmean = np.mean(zm)

    outDict = {}

    outDict['max'] = maxmean/meanvar
    outDict['min'] = minmean/meanvar
    outDict['mean'] = meanmean/meanvar
    outDict['meanmaxmin'] = (outDict['max']+outDict['min'])/2
    outDict['meanabsmaxmin'] = (np.absolute(outDict['max'])+np.absolute(outDict['min']))/2


    return outDict

def f_entropy(x):

    h = -np.sum(np.multiply(x[ x > 0],np.log(x[x > 0])))

    return h

def SB_MotifTwo(y,binarizeHow = 'diff'):

    yBin = BF_Binarize(y,binarizeHow)

    N = len(yBin)

    r1 = (yBin == 1)

    r0 = (yBin == 0)

    outDict = {}

    outDict['u'] = np.mean(r1)
    outDict['d'] = np.mean(r0)

    pp  = np.asarray([ np.mean(r1), np.mean(r0)])

    outDict['h'] = f_entropy(pp)

    r1 = r1[:-1]
    r0 = r0[:-1]

    rs1 = [r0,r1]

    rs2 = [[0,0],[0,0]]
    pp = np.zeros((2,2))

    letters = ['d','u']

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            rs2[i][j] = np.logical_and(rs1[i],yBin[1:] == j)

            outDict[l1 + l2] = np.mean(rs2[i][j])

            pp[i,j] = np.mean(rs2[i][j])

            rs2[i][j] = rs2[i][j][:-1]

    outDict['hh'] = f_entropy(pp)

    rs3 = [[[0,0],[0,0]],[[0,0],[0,0]]]
    pp = np.zeros((2,2,2))

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            for k in range(2):

                l3 = letters[k]

                rs3[i][j][k] =np.logical_and(rs2[i][j],yBin[2:] == k)

                outDict[l1 + l2 + l3] = np.mean(rs3[i][j][k])

                pp[i,j,k] = np.mean(rs3[i][j][k])

                rs3[i][j][k] = rs3[i][j][k][:-1]

    outDict['hhh'] = f_entropy(pp)

    rs4 = [[[[0,0],[0,0]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[0,0],[0,0]]]]
    pp = np.zeros((2,2,2,2))

    for i in range(2):

        l1 = letters[i]

        for j in range(2):

            l2 = letters[j]

            for k in range(2):

                l3 = letters[k]

                for l in range(2):

                    l4 = letters[l]

                    rs4[i][j][k][l] =np.logical_and(rs3[i][j][k],yBin[3:] == l)

                    outDict[l1 + l2 + l3 + l4] = np.mean(rs4[i][j][k][l])

                    pp[i,j,k,l] = np.mean(rs4[i][j][k][l])


    outDict['hhhh'] = f_entropy(pp)


    return outDict
def time_series_dependent_algos(y,algos,results,t):
        if np.count_nonzero(np.isnan(y)) > 0:
            #print(y)
            #print(t)
            raise Exception('Missing Value')
        #print('Corr')
        if 'CO_AutoCorr' in algos:
            corr = CO_AutoCorr(y,[],'Forier',t)

            i = 0

            for c in corr:
                if i > 25:
                    break
                elif i == 0:
                    i = i + 1
                    continue

                results['CO_AutoCorr lag ' + str(i)] = c
                i = i + 1

        #print('f1')
        if 'CO_f1ecac' in algos:
            results['CO_f1ecac'] = CO_f1ecac(y)

        #print('first min')
        if 'CO_FirstMin' in algos:
            results['CO_FirstMin'] = CO_FirstMin(y)

        if 'CO_FirstZero' in algos:
            results['CO_FirstZero'] = CO_FirstZero(y)

        #print('glscf')
        if 'CO_glscf' in algos:
            for alpha in range(1,5):
                for beta in range(1,5):
                    results['CO_glscf ' + str(alpha) + ' ' + str(beta)] = CO_glscf(y,alpha,beta)

        if 'CO_tc3' in algos:
            results['CO_tc3'] = CO_tc3(y)

        if 'CO_trev' in algos:
            results['CO_trev'] = CO_trev(y)

        # if 'dfa' in algos:
        #     results['dfa'] = dfa(y)

        if 'DN_CompareKSFit' in algos:
            out = DN_CompareKSFit(y)
            results = parse_outputs(out,results,'DN_CompareKSFit')


        if 'DT_IsSeasonal' in algos:
            results['IsSeasonal?'] = DT_IsSeasonal(y)

        if 'EN_ApEn' in algos:
            results['EN_ApEn'] = EN_ApEn(y)


        if 'EN_CID' in algos:
            out = EN_CID(y)
            results = parse_outputs(out,results,'EN_CID')

        if 'EN_PermEn' in algos:
            results['EN_PermEn 2, 1'] = EN_PermEn(y)
            results['EN_PermEn 3, 6'] = EN_PermEn(y,3,6)

        if 'EN_SampEn' in algos:
            out = EN_SampEn(y)
            results['EN_Sample Entropy'] = out["Sample Entropy"]
            results["EN_Quadratic Entropy"] = out["Quadratic Entropy"]


        if 'IN_AutoMutualInfo' in algos:
            out = IN_AutoMutualInfo(y)
            results = parse_outputs(out,results,'IN_AutoMutualInfo')

        if 'SY_Trend'in algos:
            if not BF_iszscored(y):
                out = SY_Trend((y-np.mean(y)) / np.std(y))
            else:
                out = SY_Trend(y)
            results = parse_outputs(out,results,'Trend')

        # if 'SC_HurstExp' in algos:
        #     results['Hurst Exp'] = SC_HurstExp(y)
        # if 'SC_DFA' in algos:
        #     results['DFA alpha'] = SC_DFA(y)
        how = ['absfar','absclose']
        ps = [.6]

        for h in how:

            for p in ps:

                out = DN_RemovePoints(y,h,p)

                if isinstance(out,dict):

                    results = parse_outputs(out,results,'DN_RemovePoints' + str(h) + '_' + str(p))

        results['DK_crinkle'] = DK_crinkle(y)
        results['DK_theilerQ'] = DK_theilerQ(y)

        binaryMethods = ['diff']

        for bMethod in binaryMethods:

            out = SB_MotifTwo(y,bMethod)

            if isinstance(out,dict):

                results = parse_outputs(out,results,'SB_MotifTwo_'  + bMethod)

        binarizeHows = ['diff','mean','median','iqr']

        for binary in binarizeHows:

            out = SB_BinaryStats(y,binary)

            if isinstance(out,dict):

                results = parse_outputs(out,results,'SB_BinaryMethod_' + str(binary))

        types = ['seg','len']
        numSegs = [2,3,4,5,10]

        for ty in types:

            for seg in numSegs:

                out = SY_StatAv(y,ty,seg)

                if isinstance(out,dict):

                    results = parse_outputs(out,results,'SY_StatAv' + str(ty) + '_ ' + str(seg) + '_')
        samplAs = [.25,.5,1]
        bs = [.05,.1,.25]

        for a in samplAs:

            for b in bs:

                out = EX_MovingThreshold(y,a,b)

                if isinstance(out,dict):

                    results = parse_outputs(out,results,'EX_MovingThreshold_a' + str(a)+ '_b' + str(b))
        out = SY_DriftingMean(y)

        if isinstance(out,dict):

            results = parse_outputs(out,results,'SY_DriftingMean_')
        return results

def run_algos(y,algos = 'all',last_non_nan = np.nan,t=1):

    results = {}

    if algos == 'all':
        algos = ['EN_PermEm', 'DN_Moments', 'DN_Withinp', 'DN_Quantile', 'DN_RemovePoints', 'DN_OutlierInclude', 'DN_Burstiness', 'DN_pleft', 'CO_FirstZero', 'DN_Fit_mle', 'CO_FirstMin', 'DN_IQR', 'DN_CompareKSFit', 'DN_Mode', 'EN_SampEn', 'SY_Trend', 'DN_Mean', 'CO_glscf', 'DN_Cumulants', 'DN_Range', 'DN_FitKernalSmooth', 'DN_Median', 'DN_Spread', 'DN_MinMax', 'DN_CustomSkewness', 'EN_mse', 'IN_AutoMutualInfo', 'EN_CID', 'DN_Unique', 'DT_IsSeasonal', 'EN_ApEn', 'SC_HurstExp', 'DN_ObsCount',  'EN_ShannonEn', 'dfa', 'CO_tc3', 'DN_nlogL_norm', 'CO_AutoCorr', 'CO_f1ecac', 'DN_ProportionValues', 'DN_STD', 'CO_trev', 'DN_cv', 'DN_TrimmedMean', 'SC_DFA', 'DN_HighLowMu']
    #print(algos)
    #Make sure time series isn't empty
    if 'DN_ObsCount' in algos:
        results['Observations'] = DN_ObsCount(y)
        if results['Observations'] <= 10:
            return results
    if len(algos)>1:
    #Compute all histogram stats on non-imputed data
        results = run_histogram_algos(y,algos,results)
    else:
        return results
    #if y is only 1 value don't calc time depedent stuff
    if results['DN_std'] == 0.0:

        return results

    #impute data for algos that can't run with nans
    y_imputed = impute(y,last_non_nan)

    results = time_series_dependent_algos(impute(y,last_non_nan),algos,results,t)

    return results

def parse_outputs(outputs,results,func):
    for key in outputs:
        if isinstance(outputs[key],list) or isinstance(outputs[key],np.ndarray):
            i = 1
            for out in outputs[key]:
                results[func + ' ' + key + ' ' + str(i)] = out
                i = i + 1
        else:
            results[func + ' ' + key] = outputs[key]
    return results




def get_interval(interval_length,end_time,times):
    # interval_length is in seconds
    # end_time is in seconds
    return np.where((times <= end_time) & (times > end_time - interval_length))

def all(t,hr,time,interval_length = 60*10):
    indx = get_interval(interval_length,int(t),time)
    results = run_algos(hr[indx])
    results['time'] = t
    return results

def all_times(t,series,time,interval_length = 600):
    indx = get_interval(interval_length,int(t),time)
    indx = indx[0]
    if len(indx) <= 1:
        return {'time':t}
    if np.isnan(series[np.min(indx)]):
        nonnan = np.argwhere(~np.isnan(series))[np.argwhere(~np.isnan(series)) < np.min(indx)]
        if len(nonnan) != 0:
            last_non_nan_indx = np.max(nonnan)
            lastvalue = series[last_non_nan_indx]
        else:
            lastvalue = np.nan
        results = run_algos(series[indx],'all',lastvalue,t)
        #results = run_algos(series[indx],['DN_ObsCount'],lastvalue,t)
    else:
        #results = run_algos(series[indx],['DN_ObsCount'],1,t)
        results = run_algos(series[indx],'all',1,t)
    results['time'] = t
    return results

def impute(y_test,last):
    if y_test[0] == np.nan and last == np.nan:
        min = np.min(np.argwhere(~np.isnan(y_test)))
        return impute(y_test[min:],last)
    elif y_test[0] == np.nan:
        y_test[0] = last
    y_test = nanfill(y_test)
    return y_test

def nanfill(x):
    for i in np.argwhere(np.isnan(x)):
        x[i] = x[i-1]
    return x
# from Operations import *
# if os.path.exists('/Users/justinniestroy-admin/Documents/Work/Tests/Spark/dependencies.zip'):
#     with open('/Users/justinniestroy-admin/Documents/Work/Tests/Spark/write.txt','w') as f:
#         f.write('test2')
#     sys.path.append('/Users/justinniestroy-admin/Documents/Work/Tests/Spark/dependencies.zip')

import sys

import numpy as np
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql import SQLContext

# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]': return DateType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)


# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    i = 0
    for column, typo in zip(columns, types):
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)

def get_interval(interval_length,end_time,times):
    # interval_length is in seconds
    # end_time is in seconds
    return np.where((times <= end_time) & (times > end_time - interval_length))

file_location = sys.argv[2]
output_location = sys.argv[1]

sc =SparkContext()
sqlContext = SQLContext(sc)
test = sc.textFile(file_location).map(lambda line: line.split(","))

a = np.array(test.collect())
a = a[1:,:]
a[10562,0] = a[10561,0]

times = a[:,1].astype(float)
hr = a[:,0].astype(float)
step_size=60*5
interval_length = 60*10
end_times = np.arange(np.min(times) + interval_length,np.max(times),step_size)

results = []
for t in end_times:
    results.append(all_times(t,hr,times))

df = pd.DataFrame(results)
spark_df = pandas_to_spark(df).coalesce(1)
spark_df.write.option("header", "true").csv(output_location)
