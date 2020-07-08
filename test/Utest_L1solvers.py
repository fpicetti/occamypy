#!/usr/bin/env python3
# import sys,os
# sys.path.insert(0, "/net/server/homes/sep/ettore/research/packages/acoustic_isotropic_operators/local/lib/python")
# sys.path.insert(0, "/net/server/homes/sep/ettore/research/packages/pySolver/GenericSolver/python")
import pyVector as Vec
import pyLCGsolver as LCG
import pySymLCGsolver as SymLCGsolver
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger
import sep_util as sep
import numpy as np

import pyISTCsolver as ISTC
import pyISTAsolver as ISTA
from Gaussian_smoothing import Gauss_smooth_scipy as Gauss_smooth
# from spatialDerivModule import LaplacianPython

if __name__ == '__main__':
	# true_model = SepVector.getSepVector(ns=[301,601])
	true_model = Vec.VectorIC((301, 601))
	true_model_arr = true_model.getNdArray()
	#Adding spikes to the model
	true_model_arr[300,150] = 10.0
	true_model_arr[200,100] = -5.0
	true_model_arr[400,280] = 1.0
	# genericIO.defaultIO.writeVector("true_model_spike.H",true_model)
	true_model.writeVec("true_model_spike.H")
	#Instantiating operator
	sigmax = 5.0
	sigmaz = 4.0
	Gauss_op = Gauss_smooth(true_model,sigmax,sigmaz)
	#Generating data
	data = true_model.clone()
	Gauss_op.forward(False,true_model,data)
	# Lapla_op = LaplacianPython(true_model,true_model,0)
	Lapla_op = None
	# genericIO.defaultIO.writeVector("data_gauss.H",data)
	data.writeVec("data_gauss.H")

	####################################################
	#L2-norm inversions
	#Create stopper
	niter = 100
	Stop  = Stopper.BasicStopper(niter=niter)
	#Create solver
	LCGsolver = LCG.LCGsolver(Stop)
	LCGsolver.setDefaults()


	#Create L2-norm linear problem
	initial_model = true_model.clone()
	initial_model.zero()
	L2Prob = Prblm.LeastSquares(initial_model, data, Gauss_op)
	# LCGsolver.run(L2Prob,verbose=True)
	# genericIO.defaultIO.writeVector("inverted_model_L2.H",L2Prob.model)

	#Running using symmetric problem (unstable)
	SymProb = Prblm.LeastSquaresSymmetric(initial_model, data, Gauss_op)
	SLCG = SymLCGsolver.SymLCGsolver(Stop)
	# SLCG.run(SymProb,verbose=True)
	# genericIO.defaultIO.writeVector("inverted_model_L2_Sym.H",SymProb.model)

	#Regularization using Laplacian operator
	L2ProbReg = Prblm.RegularizedL2(initial_model, data, Gauss_op, 0.1, Lapla_op)
	# L2ProbReg.estimate_epsilon(True)
	# LCGsolver.run(L2ProbReg,verbose=True)
	# genericIO.defaultIO.writeVector("inverted_model_L2_Reg.H",L2ProbReg.model)

	#L1 problem
	# op_norm = Gauss_op.powerMethod(True)
	op_norm = 15790. #Estimated from the previous line using the power method
	L1LassoISTC = Prblm.Lasso(initial_model, data, Gauss_op, op_norm=op_norm)
	Stop1  = Stopper.BasicStopper(niter=500)
	ISTCsolver = ISTC.ISTCsolver(Stop1,5,cooling_start=0.01,cooling_end=0.99,logger=logger("ISTClog.txt"))
	ISTCsolver.run(L1LassoISTC,True)
	# genericIO.defaultIO.writeVector("inverted_model_L1_ISTC.H",L1LassoISTC.model)

	#Solving using the ISTA
	L1LassoISTA = Prblm.Lasso(initial_model, data, Gauss_op, op_norm=op_norm, lambda_value=0.1)
	ISTAsolver = ISTA.ISTAsolver(Stop,logger=logger("ISTAlog.txt"))
	ISTAsolver.run(L1LassoISTA,verbose=True)
	# genericIO.defaultIO.writeVector("inverted_model_L1_ISTA.H",L1LassoISTA.model)

	#Solving using the FISTA
	L1LassoFISTA = Prblm.Lasso(initial_model, data, Gauss_op, op_norm=op_norm, lambda_value=1.0)
	FISTAsolver = ISTA.ISTAsolver(Stop,fast=True,logger=logger("FISTAlog.txt"))
	FISTAsolver.run(L1LassoFISTA,verbose=True)
	# genericIO.defaultIO.writeVector("inverted_model_L1_FISTA1.H",L1LassoFISTA.model)
	L1LassoFISTA.model.writeVec("inverted_model_L1_FISTA1.H")
	FISTAsolver.stoppr.niter=3000
	for ii in range(5):
			L1LassoFISTA.lambda_value*=0.1
			FISTAsolver.run(L1LassoFISTA,verbose=True)
			# genericIO.defaultIO.writeVector("inverted_model_L1_FISTA%s.H"%(ii+2),L1LassoFISTA.model)
			L1LassoFISTA.model.writeVec("inverted_model_L1_FISTA%s.H"%(ii+2))

























#
