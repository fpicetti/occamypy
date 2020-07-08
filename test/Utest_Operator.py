#!/usr/bin/env python3
import sys,os
sys.path.append("../../python")
import pyOperator
from pyParOperator import parOperator
import pyVector
import numpy as np

#Using vectorIC
vec1=pyVector.VectorIC(np.zeros((100, 200)))
op1=pyOperator.scalingOp(vec1,10.)
op1.dotTest(True)

#Using vectorOC
vec2=pyVector.VectorOC(np.zeros((100, 200)))
op2=pyOperator.scalingOp(vec2,10.)
op2.dotTest(True)

#Testing pyParOperator
vec3=pyVector.VectorOC(np.zeros((100, 200)))
vec3.rand()
vec4=vec3.clone()
op2 = parOperator(vec3,vec3)
#Setting forward operator
operator_path = os.environ.get('REPOSITORY')+"/python_solver/python_modules/unit_tests/"+"ScaleOp.p"
op2.set_forward(operator_path,"input.H","output.H")
op2.set_adjoint(operator_path,"input.H","output.H")
op2.forward(False,vec3,vec4)
#testing with in-core vectors
vec5=pyVector.VectorIC(np.zeros((100, 200)))
vec6=vec5.clone()
vec5.rand()
op2.forward(False,vec5,vec6)
