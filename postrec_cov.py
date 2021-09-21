#!/bin/python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.linalg import cholesky, solve_triangular, det, eigh, solve
from scipy.special import legendre
from scipy.integrate import quad, dblquad
from scipy import integrate, interpolate
from argparse import ArgumentParser

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-z',type=float,default=0,help='output redshift [default:0]')
    argparser.add_argument('-r',type=float,default=10.,help='reconstruction scale in unit of Mpc/h [default:10]')
    argparser.add_argument('-s',type=str,default='z',help='real space or redshift space[z/r]')
    argparser.add_argument('-ssc',type=str,default='n',help='SSC is included? [n/y]')
    argparser.add_argument('-c',type=str,default='n',help='Calculate covariance (if no, read pre-calculated one) [n/y]')
    argparser.add_argument('-kmax',type=float,default=0.2,help='output kmax in unit of [h/Mpc] [default:0.2]')
    argparser.add_argument('-nkmax',type=float,default=20,help='binning number of k [default:20]')
    argparser.add_argument('-lbox',type=float,default=500,help='simulation boxsize in unit of Mpc/h [default:500]')
    argparser.add_argument('-npix',type=int,default=512,help='grid number per side [default:512]')
    argparser.add_argument('-inf',type=str,default='planck15_lin_matterpower_z0.dat',help='input file of linear P(k)')
    argparser.add_argument('-outf',type=str,default='sn.pdf',help='output file of S/N')
    return argparser.parse_args()

args = get_option()

### set output redshift
zin=args.z

### set reconstruction scale in unit of Mpc/h (no reconstruction when Rs=0) 
Rs=args.r

### real/redshift space [r/z]
space=args.s

### set cosmological parameters (this is only used in growth factor/rate caclulations)
#params={'flat':True,'H0':67.27,'Om0':0.3156,'Ob0':0.04917,'sigma8':0.831,'ns':0.9645}
#cosmo_for_colossus=cosmology.setCosmology('myCosmo',params)
cosmo_for_colossus=cosmology.setCosmology('planck15')

### consider super sample covariance or not [n/y]
ssc = args.ssc

### covariance calculation or just reading pre-caluclated covariance [y/n]
calc_covtree = args.c

### set box-size and grid number per side
lbox = args.lbox
vol = lbox**3
npix = args.npix
nden = (npix/lbox)**3

use_mono = 'y'
if space == 'z':
    use_quad = 'y'
else:
    use_quad = 'n'

###
nkmax=args.nkmax
koutmax=args.kmax
dk=koutmax/nkmax
kout=np.linspace(dk,koutmax,nkmax)

nkmax2 = nkmax*2

### read input linear matter power spectrum
infpk=args.inf
kin = np.loadtxt(infpk,usecols=[0])
pkin = np.loadtxt(infpk,usecols=[1])

### compute growth factor
gz = cosmo_for_colossus.growthFactor(z=float(zin))
print('growth factor: ',gz)
pkin *= gz*gz

### compute growth rate
if space == 'r':
    fz = 0.
else:
    if zin > 0.01:
        gz_u=cosmo_for_colossus.growthFactor(z=(zin+0.01))
        gz_l=cosmo_for_colossus.growthFactor(z=(zin-0.01))
        fz=-np.log(gz_u/gz_l)/np.log((1.01+zin)/(0.99+zin))
    else:
        gz_u=cosmo_for_colossus.growthFactor(z=(zin+0.005))
        gz_l=cosmo_for_colossus.growthFactor(z=zin)
        fz=-np.log(gz_u/gz_l)/np.log((1.005+zin)/(1+zin))
    print('growth rate:fz=',fz)

### anistropy parameter beta = grate (linear bias is set to be unity)
beta = fz

###interpolate linear P(k) in logarithmic scale
logkin = np.log(kin)
logpkin = np.log(pkin)
interp1d_pkl = interpolate.interp1d(logkin,logpkin,kind='cubic')
pklinout=np.exp(interp1d_pkl(np.log(kout)))

print(kout)
print(pklinout)

### set the range of wavenumber k in a broad range
logkmin=-3.99
logkmax=2.

kmin=10.**logkmin
kmax=10.**logkmax

def win_cube(p):

    f = lambda s, t: np.sinc(p*np.sin(t)*np.cos(s))*np.sinc(p*np.sin(t)*np.sin(s))*np.sinc(p*np.cos(t))
    win = integrate.dblquad(f,0,0.5*np.pi,lambda t: 0, lambda t: 0.5*np.pi)

    return win[0]

def win_top(p):

    return 3./p**3*(np.sin(p)-p*np.cos(p))

def win_gauss(k):

    return np.exp(-0.5*(k*Rs)**2)

### monopole and quadpoles of redshift-space power spectra
Legendre0=legendre(0)
Legendre2=legendre(2)

### compute multipole power spectra: P_l (l=0,2)
pkl0=integrate.quad(lambda mu:(1+fz*mu*mu)**2*Legendre0(mu)*1,0,1)[0]
pkl2=integrate.quad(lambda mu:(1+fz*mu*mu)**2*Legendre2(mu)*5,0,1)[0]

pkl0_lin = pkl0 * pklinout
pkl2_lin = pkl2 * pklinout
pkl_lin = np.append(pkl0_lin,pkl2_lin)

### take into account the finite boxsize and resolutio
kboxmin = max(2.*np.pi/lbox,kmin)
kboxmax = min(2.*np.pi/lbox*npix/2,kmax)

#### calculate Gaussian covariance of P0 & P2  (shot noise is neglected)
nmode=4./3*np.pi*((kout+dk/2)**3-(kout-dk/2)**3)*vol/(2*np.pi)**3
cov_part00 = 2./nmode*((1+4./3.*fz+6./5.*fz**2+4./7.*fz**3+1./9.*fz**4)*(pklinout)**2 \
                + 2./nmode*(1+2./3.*fz+1./5.*fz**2)*(pklinout)+1./nmode**2)
cov_part02 = 2./nmode*((8./3.*fz+24./7.*fz**2+40./21.*fz**3+40./99*fz**4)*(pklinout)**2 \
                + 2./nmode*(4./3.*fz+4./7.*fz**2)*(pklinout))
cov_part22 = 2./nmode*((5.+220./21.*fz+90./7.*fz**2+1700./231.*fz**3+2075./1287.*fz**4)*(pklinout)**2 \
                + 2./nmode*(5+110./21.*fz+15./7.*fz**2)*(pklinout)+5./nmode**2)

cov_gauss=np.concatenate([np.concatenate([np.diag(cov_part00),np.diag(cov_part02)]),np.concatenate([np.diag(cov_part02),np.diag(cov_part22)])],axis=1)

### tree-level non-Gaussian covarariane
fname_covtree_head='output_cov/cov_tree_rs{:}_z{:}_kmax{:}'.format(int(Rs),zin,koutmax)
print(fname_covtree_head)
    
if calc_covtree == 'y':
    for il in range(3):
        fname_covtree = fname_covtree_head + '_lcomb{:}.txt'.format(il)
    
        nphi = 3
        dphi = np.pi/nphi

        cov_tree = np.zeros((nkmax,nkmax))

        for ik in range(nkmax):
            for jk in range(nkmax):
                k = kout[ik] 
                pk1 = np.exp(interp1d_pkl(np.log(k)))
                kr = kout[jk]
                pk2 = np.exp(interp1d_pkl(np.log(kr)))
                    
                r = kr/k
                rmin = kboxmin/k
                rmax = kboxmax/k
                xmin=np.maximum(-1,(1+r*r-rmax*rmax)/2./r)
                xmax=np.minimum(1,(1+r*r-rmin*rmin)/2./r)

                ftermintmu = np.zeros(3)
                def fterm_int_mu(mu):
                    fmu2 = fz*mu*mu
                    ftermintx = np.zeros(3)
                    def fterm_int_x(x):
                        ks = k*np.sqrt(1+r*r-2*r*x)
                        Wp = win_gauss(kr)
                        Wk = win_gauss(k)
                        Ws = win_gauss(ks)
                        pks = np.exp(interp1d_pkl(np.log(ks)))
                        for iphi in range(nphi):
                            phi = -0.5 * np.pi + dphi * (iphi + 0.5)
                            mupz = x*mu+np.sqrt((1-x*x)*(1-mu*mu))*np.sin(phi)
                            pz = kr*mupz
                        
                            if il == 0:
                                lterm = 1.
                            elif il == 1:
                                lterm = (5*(3*mu*mu-1)/2.+5*(3*mupz*mupz-1)/2.)/2.
                            elif il == 2:
                                lterm = 5*(3*mu*mu-1)/2.*5*(3*mupz*mupz-1)/2.

                            if Rs == 0:
                                f2term1 = (7*(fz*pz*mu)**2-7*fz*k*pz*mu*(1-2*r*x+fmu2)+k*kr*(-7*x*(1+fmu2)+r*(-3+10*x*x+6*fmu2*(x*x-1))))**2 \
                                      /(196*(ks*r)**4)
                                f2term2 = -(k*kr*kr*(-3-7*r*x+10*x*x)-7*(fz*pz)**2*mu*(pz-k*mu)+fz*k*pz*(pz*(-6-7*r*x+6*x*x)-7*kr*(r-2*x)*mu)) \
                                      *(-7*(fz*pz*mu)**2+7*fz*k*pz*mu*(1-2*r*x+fmu2)+k*kr*(3*r+7*x-10*r*x*x+fmu2*(7*x-6*r*(x*x-1)))) \
                                      /(196*ks**4*kr*r)
                            else:
                                f2term1 = (k**2*kr**2*(1+r**2-2*r*x)*(-7*(-1+Wp)*x+r*(3-7*Ws+7*r*(-Wp+Ws)*x+2*(-5+7*Wp)*x**2))+7*fz**2*pz*(-pz+k*mu) \
                                       *(pz**2*(Wp*x+r*(Ws-r*Ws*x+Wp*(r-2*x)*x))+k*pz*(-(Wp*x)-r*Wp*(r-2*x)*x+r*Ws*(-1+r*x))*mu+k**2*r*(1+r**2-2*r*x)*mu**2) \
                                       +fz*k**2*(7*pz**2*(1+2*r*(r-x))*(-(Wp*x)-r*Wp*(r-2*x)*x+r*Ws*(-1+r*x))+7*k*pz*r*(1+r*(r+2*r*Ws+2*(-2+Wp+r**2*(-1+Wp-Ws))*x-4*r*(-1+Wp)*x**2))*mu \
                                        +kr**2*(-7*(-1+Wp)*x-6*r**3*(-1+x**2)+r**2*x*(-5-7*Wp+7*Ws+12*x**2)+r*(6-7*Ws+2*(-10+7*Wp)*x**2))*mu**2))**2/(196.*k**2*kr**6*(1+r**2-2*r*x)**4)
                                f2term2 = ((-(k**4*r*(1+r**2-2*r*x)*(r*(-3+7*Ws)+7*(r**2*(-1+Wk)+Wk-Ws)*x+2*r*(5-7*Wk)*x**2))-7*fz**2*k*mu*(-pz+k*mu) \
                                        *(pz**2*(1+r**2-2*r*x)-k*pz*r*(r*Ws+(Wk+r**2*Wk-Ws)*x-2*r*Wk*x**2)*mu+k**2*r*(r*Ws+(Wk+r**2*Wk-Ws)*x-2*r*Wk*x**2)*mu**2) \
                                        +fz*k**2*(pz**2*(6-7*r**3*(-1+Wk)*x-6*x**2+r*x*(-5-7*Wk+7*Ws+12*x**2)+r**2*(6-7*Ws+2*(-10+7*Wk)*x**2)) \
                                        +7*k*pz*r*(r+r**3+2*r*Ws+2*(-1+r**2*(-2+Wk)+Wk-Ws)*x-4*r*(-1+Wk)*x**2)*mu-7*k**2*r*(2+r**2-2*r*x)*(r*Ws+(Wk+r**2*Wk-Ws)*x-2*r*Wk*x**2)*mu**2)) \
                                       *(k**4*r**2*(1+r**2-2*r*x)*(-7*(-1+Wp)*x+r*(3-7*Ws+7*r*(-Wp+Ws)*x+2*(-5+7*Wp)*x**2)) \
                                         +7*fz**2*pz*(-pz+k*mu)*(pz**2*(Wp*x+r*(Ws-r*Ws*x+Wp*(r-2*x)*x))+k*pz*(-(Wp*x)-r*Wp*(r-2*x)*x+r*Ws*(-1+r*x))*mu+k**2*r*(1+r**2-2*r*x)*mu**2) \
                                         +fz*k**2*(7*pz**2*(1+2*r*(r-x))*(-(Wp*x)-r*Wp*(r-2*x)*x+r*Ws*(-1+r*x))+7*k*pz*r*(1+r*(r+2*r*Ws+2*(-2+Wp+r**2*(-1+Wp-Ws))*x-4*r*(-1+Wp)*x**2))*mu \
                                            +k**2*r**2*(-7*(-1+Wp)*x-6*r**3*(-1+x**2)+r**2*x*(-5-7*Wp+7*Ws+12*x**2)+r*(6-7*Ws+2*(-10+7*Wp)*x**2))*mu**2)))/(196.*k**8*r**3*(1+r**2-2*r*x)**4)
                            f3term = (-6*fz*kr*pz*x*mu*(7+7*fmu2+r**4*(19-12*x*x+7*fmu2)-2*r*r*(-7-fmu2+2*x*x*(7+4*fmu2))) \
                                  -3*(fz*pz*mu)**2*(7+7*fmu2+r**4*(19-12*x*x+7*fmu2)+r*r*(26+14*fmu2-4*x*x*(10+7*fmu2))) \
                                  +k*k*(-21*(r*x)**2*(1+fmu2)+2*r**4*(5+15*fmu2-2*x*x*(11+12*fmu2)+x**4*(38+30*fmu2)) \
                                    +r**6*(10+30*fmu2+4*x**4*(7+3*fmu2)-x*x*(59+63*fmu2))))/(126.*(kr*r)**2*(1+r**4+2*r*r*(1-2*x*x)))
                            if Rs > 0:
                                f3term += (-7*k*Wp**2*(fz*k*pz**2*r*x+kr**3*x)**2*k*(1+fmu2)+(kr**3*(fz*pz**2+kr**2)*Wp*x*(k**3*r*(7*x+7*r**4*x-2*r*(5+9*x**2)-2*r**3*(5+9*x**2) \
                                        +r**2*x*(34+8*x**2))+7*fz**2*pz*(1+r**2-2*r*x)*mu*(pz-k*mu)**2+fz*k*(pz**2*(-7+21*r*x+7*r**3*x-r**2*(13+8*x**2)) \
                                            +k*pz*(7+7*r**4-28*r*x-28*r**3*x+2*r**2*(13+8*x**2))*mu+k**2*r*(-7*r**3+7*x+21*r**2*x-r*(13+8*x**2))*mu**2))) \
                                           /(1+r**2-2*r*x)-(2*k**3*r**4*(fz*pz**2+kr**2)*Ws*(-1+r*x)*(k**3*r*(7*x+7*r**4*x-2*r*(5+9*x**2)-2*r**3*(5+9*x**2)+r**2*x*(34+8*x**2)) \
                                        +7*fz**2*pz*(1+r**2-2*r*x)*mu*(pz-k*mu)**2+fz*k*(pz**2*(-7+21*r*x+7*r**3*x-r**2*(13+8*x**2))+k*pz*(7+7*r**4-28*r*x-28*r**3*x+2*r**2*(13+8*x**2))*mu \
                                            +k**2*r*(-7*r**3+7*x+21*r**2*x-r*(13+8*x**2))*mu**2)))/(1+r**2-2*r*x)**2+(kr**3*(fz*pz**2+kr**2)*Wp*x*(fz*kr*(7*x+7*r**2*x+r*(6+8*x**2))*(pz+k*mu)**2 \
                                            +(1+r**2+2*r*x)*(7*k**3*r*x+7*kr**3*x+2*k*kr**2*(5+2*x**2)+7*fz**2*pz*mu*(pz+k*mu)**2+7*fz*k*(pz+k*mu)*(pz+k*r**2*mu))))/(1+r**2+2*r*x))/(42.*kr**8)

                            pkf1 = (1+fmu2)*pk1
                            pkf2 = (1+fz*mupz**2)*pk2
                            dfterm = (k*k*ddk)*(kr*kr*ddk2)*(2*dphi)*(4*np.pi)
                            ftermintphi += lterm*(12*f3term*pkf1*pkf2*pkf2+8*pks*f2term1*pkf2**2+8*pks*f2term2*pkf1*pkf2)*dfterm
                        return ftermintphi
                    return quad(lambda x: fterm_int_x(x),xmin,xmax)[0]
                ftermintmu = quad(lambda mu: fterm_int_mu(mu),0,1)[0]
                cov_tree[ik,jk] = ftermintmu/(4.*np.pi*kout[ik]**2*dk)/(4.*np.pi*kout[jk]**2*dk)/vol

        ### add cyclic term
        cov_tree = cov_tree+cov_tree.transpose()
        np.savetxt(fname_covtree,cov_tree)
else:
    print('no calc cov')
    fname_covtree = fname_covtree_head + '_lcomb0.txt'
    print(fname_covtree)
    cov_tree0=np.loadtxt(fname_covtree)
    if space == 'z':
        fname_covtree = fname_covtree_head + '_lcomb1.txt'
        cov_tree1=np.loadtxt(fname_covtree)
        fname_covtree = fname_covtree_head + '_lcomb2.txt'
        cov_tree2=np.loadtxt(fname_covtree)
    else:
        cov_tree1 = cov_tree0
        cov_tree2 = cov_tree0

cov_tree = np.concatenate([np.concatenate([cov_tree0,cov_tree1]),np.concatenate([cov_tree1,cov_tree2])],axis=1)


print(len(cov_tree))

### super-sample covariance

if ssc == 'y':

### derivative of Pl0 and Pl2 (if only Kaiser effect (scale-independent), logPl0 = logPl2)
    dlnk = 0.01
    logPl0_u=interp1d_pkl(np.log(kout)+dlnk/2)
    logPl0_d=interp1d_pkl(np.log(kout)-dlnk/2)
    logpl0_deriv = 3 + (logPl0_u-logPl0_d)/dlnk
    logpl2_deriv = logpl0_deriv

### variance of background overdensity
    sigmal2 = 0.
    num_k = 50
    lbox_large = 4000.
    lnkmin=np.log(2*np.pi/lbox_large)
    #lnkmax=np.log(10.)
    lnkmax=np.log(1.)
    dlnk = (lnkmax-lnkmin)/num_k
    lnk_tab = np.linspace(lnkmin+dlnk*0.5,lnkmax-dlnk*0.5,num_k)
    k = np.exp(lnk_tab)
    Plin_tab = np.exp(interp1d_pkl(lnk_tab))
    p = k * lbox / (2. * np.pi)

    #dsigdlnk = np.zeros(len(p))
    #for i in range(len(p)):
    #    dsigdlnk[i] = k[i]**3 / (2. * np.pi**2) * Plin_tab[i] * win_cube(p[i])**2
    #    print(p[i],dsigdlnk[i])
    dsigdlnk = np.array([k[i]**3 / (2. * np.pi**2) * Plin_tab[i] * win_cube(p[i])**2 for i in range(len(p))])
    sigmal2 = np.sum(0.5*(dsigdlnk[1:]+dsigdlnk[:-1])*dlnk)

    G0 = (68./21.*(1+fz)+164./105.*fz*fz+4./15.*fz**3)/(1+2*fz/3.+fz*fz/5.)
    D0 = -((1+fz)/3.+fz*fz/5.+fz**3/7.)/(1+2*fz/3.+fz*fz/5.)
    if space == 'r':
        G2 = 0
        D2 = 0
    else:
        G2 = (122./21.*fz+656./147.*fz*fz+58./63.*fz**3)/(4./3.*fz+4./7.*fz*fz)
        D2 = -(2*fz/3+4*fz*fz/7+10*fz**3/63)/(4./3.*fz+4./7.*fz*fz)

    Pl0_res = pkl0_lin * (G0 + D0 * logpl0_deriv - (2+2./3.*fz))
    Pl2_res = pkl2_lin * (G2 + D2 * logpl2_deriv - (2+2./3.*fz))

    Pl_res = np.append(Pl0_res,Pl2_res)
    cov_ssc=sigmal2*np.dot(Pl_res.reshape(nkmax2,1),Pl_res.reshape(1,nkmax2))

### calculate S/N for Gaussian

cov = cov_gauss
arr = np.zeros(nkmax2,dtype=int)
SN2_GA = np.zeros(nkmax)
for j in range(nkmax):
    if use_mono == 'y':
        arr[j]=1
    if use_quad == 'y':
        arr[j+nkmax]=1
    indices = np.where(arr==1)[0]
    cov_sliced = cov[np.ix_(indices, indices)]
    cholesky_transform = cholesky(cov_sliced, lower=True)
    yt = solve_triangular(cholesky_transform, pkl_lin[indices], lower=True)
    SN2_GA[j] = yt.dot(yt)

print(SN2_GA)

### calculate S/N for Gaussian + tree NG + SSC

SN2_NG = np.zeros(nkmax)
cov = cov_gauss + cov_tree
if ssc == 'y':
    cov += cov_ssc

arr = np.zeros(nkmax2,dtype=int)
for j in range(nkmax):
    if use_mono == 'y':
        arr[j]=1
    if use_quad == 'y':
        arr[j+nkmax]=1
    indices = np.where(arr==1)[0]
    cov_sliced = cov[np.ix_(indices, indices)]
    cholesky_transform = cholesky(cov_sliced, lower=True)
    yt = solve_triangular(cholesky_transform, pkl_lin[indices], lower=True)
    SN2_NG[j] = yt.dot(yt)

print(SN2_NG)

fig = plt.figure()
if Rs > 0:
    plt.title("post-rec S/N ($R_s$={:}Mpc/h)".format(Rs),fontsize=16)
else:
    plt.title("pre-rec S/N".format(Rs),fontsize=16)
plt.xlabel("$k_{max}$ [Mpc/h]",fontsize=14)
if ssc == 'y':
    plt.ylabel("S/N with SSC",fontsize=14)
else:
    plt.ylabel("S/N",fontsize=14)
plt.tick_params(labelsize=12)
plt.plot(kout,np.sqrt(SN2_GA),color='black',linestyle="dotted",label='Gaussian Covariance')
plt.plot(kout,np.sqrt(SN2_NG),color='red',label='Gaussian + tree-level NG')
plt.savefig(args.outf)
