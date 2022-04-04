import numpy as np
from io import StringIO
import time,os, importlib
#from tqdm import tqdm
np.set_printoptions(linewidth=160)
from . import io as kio
from . import GT
from scipy.sparse import save_npz,load_npz, diags, eye, csr_matrix,bmat
from scipy.sparse.linalg import eigs,inv,spsolve
import scipy as sp
import scipy.linalg as spla

def es_compute_passage_stats(A_sel, B_sel, pi, K, dopdf=True,rt=None):
	r"""Compute the A->B and B->A first passage time distribution,
	first moment, and second moment using eigendecomposition of a CTMC
	rate matrix.

	Parameters
	----------
	A_sel : (N,) array-like
		boolean array that selects out the A nodes
	B_sel : (N,) array-like
		boolean array that selects out the B nodes
	pi : (N,) array-like
		stationary distribution
	K : (N, N) array-like
		CTMC rate matrix

	dopdf : bool, optional
		Do we calculate full fpt distribution or just the moments. Defaults=True.
	rt: array, optional
		Vector of times to evaluate first passage time distribution in multiples
		of :math:`\left<t\right>` for A->B and B->A. If ``None``, defaults to a logscale
		array from :math:`0.001\left<t\right>` to :math:`1000\left<t\right>`
		in 400 steps, i.e. ``np.logspace(-3,3,400)``.
		Only relevant if ``dopdf=True``

	Returns
	-------
	tau : (4,) array-like
		First and second moments of first passage time distribution for A->B and B->A [:math:`\mathcal{T}_{\mathcal{B}\mathcal{A}}`, :math:`\mathcal{V}_{\mathcal{B}\mathcal{A}}`, :math:`\mathcal{T}_{\mathcal{A}\mathcal{B}}`, :math:`\mathcal{V}_{\mathcal{A}\mathcal{B}}`]
	pt : ( len(rt),4) array-like
		time and first passage time distribution p(t) for A->B and B->A

	"""
	#multiply by negative 1 so eigenvalues are positive instead of negative
	Q=-K
	if rt is None:
		rt = np.logspace(-3,3,400)
	#<tauBA>, <tau^2BA>, <tauAB>, <tau^2AB>
	tau = np.zeros(4)
	if dopdf:
		# time*tau_range, p(t) (first 2: A->B, second 2: B->A)
		pt = np.zeros((4,len(rt)))

	#A -> B
	#P(0) is initialized to local boltzman of source community A
	rho = pi * A_sel
	rho /= rho.sum()
	#B is absorbing, so we want Q in space of A U I
	M = Q[~B_sel,:][:,~B_sel]
	x = spsolve(M,rho[~B_sel])
	y = spsolve(M,x)
	# first moment tau(A->B) = 1.Q^{-1}.rho(A) = 1.x
	tau[0] = x.sum()
	# second moment = 2 x 1.Q^{-2}.rho = 2.0* 1.Q^{-1}.x
	tau[1] = 2.0*y.sum()
	if dopdf:
		#time in multiples of the mean first passage time
		pt[0] = rt*tau[0]
		#nu=eigenvalues, v=left eigenvectors, w=right eigenvectors
		nu,v,w = spla.eig(M.todense(),left=True)
		#normalization factor
		dp = np.sqrt(np.diagonal(w.T.dot(v))).real
		#dot product (v.P(0)=rho)
		v = (v.real.dot(np.diag(1.0/dp))).T.dot(rho[~B_sel])
		#dot product (1.T.w)
		w = (w.real.dot(np.diag(1.0/dp))).sum(axis=0)
		nu = nu.real
		#(v*w/nu).sum() is the same as <tau>, the first bit is the pdf p(t)
		pt[1] = (v*w*nu)@np.exp(-np.outer(nu,pt[0]))*(v*w/nu).sum()

	#B -> A
	rho = pi * B_sel
	rho /= rho.sum()
	M = Q[~A_sel,:][:,~A_sel]
	x = spsolve(M,rho[~A_sel])
	y = spsolve(M,x)
	tau[2] = x.sum()
	tau[3] = 2.0*y.sum()
	if dopdf:
		pt[2] = rt*tau[2]
		nu,v,w = spla.eig(M.todense(),left=True)
		dp = np.sqrt(np.diagonal(w.T.dot(v))).real
		v = (v.real.dot(np.diag(1.0/dp))).T.dot(rho[~A_sel])
		w = (w.real.dot(np.diag(1.0/dp))).sum(axis=0)
		nu = nu.real
		pt[3] = (v*w*nu)@np.exp(-np.outer(nu,pt[2]))*(v*w/nu).sum()
		return tau, pt.T
	else:
		return tau

