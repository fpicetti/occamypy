import pyDaskVector
from dask_util import DaskClient
import pyDaskOperator as DaskOp
import pyOperator as pyOp
import pyVector as  Vec
import numpy as np

hostnames = ["thanos", "thanos", "friday", "friday"]
client = DaskClient(hostnames=hostnames)
# params = {"nanny": True, "job_name":"test1", "cores":32, "memory":'1GB', "queue":'xaginolimit', "walltime":'20:00:00', "interface":'ib0', "job_extra": ["-V"]}
# params = {"nanny":"0", "job_name":"test1","cores":32, "memory":'120GB', "queue":'xagigpu', "walltime":'9999:00:00', "interface":'ib0', "resource_spec":"select=1:ncpus=32:mem=120gb:ngpus=2"}
# client = DaskClient(pbs_params=params,n_workers=2,n_jobs=2)
# client = DaskClient(pbs_params=params,n_workers=2)
vec = Vec.VectorIC((10, 1))
chunks = (1,1,1,1)
vecD = pyDaskVector.DaskVector(client, vector_template=vec, chunks=chunks)

# Checking which worker has a given chunk
client.getClient().who_has(vecD.vecDask[0])
client.getClient().who_has(vecD.vecDask[1])
client.getClient().who_has(vecD.vecDask[2])
client.getClient().who_has(vecD.vecDask[3])

# Testing DaskCollectOp
vecLocal = Vec.VectorIC(np.block(vecD.getNdArray()).shape)

CollectOp = DaskOp.DaskCollectOp(vecD,vecLocal)
CollectOp.dotTest(True)
vecD.rand()
CollectOp.forward(False, vecD, vecLocal)
print(vecLocal.getNdArray()[-1][-1])
print(vecD.getNdArray()[-1][-1][-1])


vec1 = Vec.VectorIC((200, 1))
vectors = [vec1.clone(),vec1.clone()]
vec1D = pyDaskVector.DaskVector(client, vectors=vectors)

vec1D.rand()
vec1D.norm()
vecClone = vec1D.clone()
vec1D.checkSame(vecClone)
vec1D.checkSame(vec1D)

#Testing getNdArray
vec1 = Vec.VectorIC((800, 500, 1)); vec1.rand()
vec2 = Vec.VectorIC((800, 500, 1)); vec2.rand()
vec3 = Vec.VectorIC((800, 500, 1)); vec3.rand()
vec2D = pyDaskVector.DaskVector(client,vectors=(vec1,vec2,vec3))
arr = vec2D.getNdArray()
vec1.norm()**2+vec2.norm()**2+vec3.norm()**2
vec2D.norm()**2

#Testing Dask Operator
vec = Vec.VectorIC((2, 1))
chunks = (5,5,5,5)
sc = 10.0
vecD = pyDaskVector.DaskVector(client,vector_template=vec,chunks=chunks)
scal_op_args = [(vec_i,sc) for vec_i in vecD.vecDask]

scaleOp_constr = pyOp.scalingOp
dask_scaling = DaskOp.DaskOperator(client,scaleOp_constr,scal_op_args,chunks)

vecD1 = vecD.clone()
vecD.set(1.0)
dask_scaling.forward(False,vecD,vecD1)
arr = vecD1.getNdArray()

#Dot-product test
dask_scaling.dotTest(True)


#Testing Spreading/Stacking operator
S = DaskOp.DaskSpreadOp(client,vec,chunks)
S.dotTest(True)
x = S.domain.clone()
y = S.range.clone()
x.set(1.0)
S.forward(False,x,y); y.getNdArray()

#Chain of scaling and spreading operator
scale_S = pyOp.ChainOperator(S,dask_scaling)
scale_S.dotTest(True)
scale_S.forward(False,x,y); y.getNdArray()

try:
	#Testing with GenericIO
	import SepVector
	vecSep = SepVector.getSepVector(ns=[200,400])
	chunks = (1,1,1,1)
	vecD = pyDaskVector.DaskVector(client,vector_template=vecSep,chunks=chunks)
	vec1 = vecSep.clone(); vec1.rand()
	vec2 = vecSep.clone(); vec2.rand()
	vec3 = vecSep.clone(); vec3.rand()
	vec4 = vecSep.clone(); vec4.rand()
	vectors = [vec1, vec2, vec3, vec4]
	vecD1 = pyDaskVector.DaskVector(client,vectors=vectors,chunks=chunks)
	vec1.norm()**2+vec2.norm()**2+vec3.norm()**2+vec4.norm()**2
	vecD1.norm()**2
	chunks = (10,10,10,10)
	sc = 10.0
	vecD = pyDaskVector.DaskVector(client,vector_template=vecSep,chunks=chunks)
	scal_op_args = [(ivec,sc) for ivec in vecD.vecDask]
	scaleOp_constr = pyOp.scalingOp
	dask_scaling = DaskOp.DaskOperator(client,scaleOp_constr,scal_op_args,chunks)
	vecD1 = vecD.clone()
	vecD.set(1.0)
	dask_scaling.forward(False,vecD,vecD1)
	arr = vecD1.getNdArray()
	dask_scaling.dotTest(True)
	#Stacking/Spreading
	S = DaskOp.DaskSpreadOp(client,vecSep,chunks)
	x = S.domain.clone()
	y = S.range.clone()
	x.set(1.0)
	S.forward(False,x,y); y.getNdArray()
	#Chain of scaling and spreading operator
	scale_S = pyOp.ChainOperator(S,dask_scaling)
	scale_S.dotTest(True)
	scale_S.forward(False,x,y); y.getNdArray()
except ImportError:
	pass
