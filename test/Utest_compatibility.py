#!/usr/bin/env python3
import sys,os,imp
sys.path.append("/net/server/homes/sep/ettore/research/packages/pySolver/GenericSolver/python")
sys.path.append("/net/server/homes/sep/ettore/research/packages/acoustic_isotropic_operators/local/lib/python")
try:
	imp.find_module('genericIO')
	import genericIO
	SepVector=genericIO.SepVector
except ImportError:
	print("Cannot load required module: \"genericIO\". Quitting the program.")
	quit()
import numpy as np
import pyOperator as Op
import pyProblem as Prblm
import pyStopperBase as Stopper
import pyLCGsolver as LCG
import pyVector

class MatMult_SepVector(Op.Operator):
	"""Operator class to perform matrix-vector multiplication"""

	def __init__(self,A,domain,range):
		"""Constructor for the class: A = matrix to use; domain = domain vector; range = range vector"""
		if(not isinstance(domain, SepVector.Vector)): raise TypeError("ERROR! Domain vector not a Vector object")
		if(not isinstance(range, SepVector.Vector)): raise TypeError("ERROR! Range vector not a Vector object")
		#Setting domain and range of operator and matrix to use during application of the operator
		self.setDomainRange(domain,range)
		self.A = np.matrix(A)
		return

	def forward(self,add,model,data):
		"""Method to compute d = A m"""
		self.checkDomainRange(model,data)
		if(not isinstance(model, SepVector.Vector)): raise TypeError("ERROR! Model vector not a Vector object")
		if(not isinstance(data, SepVector.Vector)): raise TypeError("ERROR! Data vector not a Vector object")
		if(not add): data.zero()
		#Converting to numpy arrays
		data_np=data.getNdArray()
		model_np=model.getNdArray()
		data_np+=np.matmul(A,model_np)
		return

	def adjoint(self,add,model,data):
		"""Method to compute m = A d"""
		self.checkDomainRange(model,data)
		if(not isinstance(model, SepVector.Vector)): raise TypeError("ERROR! Model vector not a Vector object")
		if(not isinstance(data, SepVector.Vector)): raise TypeError("ERROR! Data vector not a Vector object")
		if(not add): model.zero()
		#Converting to numpy arrays
		data_np=data.getNdArray()
		model_np=model.getNdArray()
		model_np+=np.matmul(A.H,data_np)
		return



if __name__ == '__main__':
	#Create stopper
	niter = 60
	Stop  = Stopper.BasicStopper(niter=niter)
	#Create solver
	LCGsolver = LCG.LCGsolver(Stop)
	#Create a sepVector
	nsamp=200
	model=SepVector.getSepVector(ns=[1,nsamp])#,storage="dataDouble")
	data=SepVector.getSepVector(ns=[1,nsamp])#,storage="dataDouble")
	A = np.matrix(np.zeros((nsamp,nsamp)))
	np.fill_diagonal(A, -2)
	np.fill_diagonal(A[1:], 1)
	np.fill_diagonal(A[:,1:], 1)
	#Create operator
	MatMultSym = MatMult_SepVector(A,model,data)
	#Testing operator
	model.rand()
	MatMultSym.forward(False,model,data)
	MatMultSym.adjoint(False,model,data)
	#Testing solver
	data_np = data.getNdArray()
	data_np.fill(1.)
	model.zero()
	#Create L2-norm linear problem
	L2Prob_sym = Prblm.RegularizedL2(model, data, MatMultSym, 0.01)
	LCGsolver.setDefaults(iter_sampling=35,save_obj=True,iter_buffer_size=1,save_model=True,save_grad=True,save_res=True,prefix="compatibility_inversion")
	LCGsolver.run(L2Prob_sym,True)
