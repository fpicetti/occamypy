import numpy as np
import dask.distributed as daskD
from collections.abc import Iterable

from occamypy import Vector, Operator
from .vector import DaskVector
from .utils import DaskClient
from .vector import scatter_large_data


def call_constructor(constr, args, kwargs=None):
    """Function to call the constructor"""
    if kwargs is None:
        if isinstance(args, Iterable):
            op = constr(*args)
        else:
            op = constr(args)
    else:
        if isinstance(args, Iterable):
            op = constr(*args, **kwargs)
        else:
            op = constr(args, **kwargs)
    return op


def call_getDomain(opObj):
    """Function to call getDomain method"""
    domain = opObj.getDomain()
    return domain


def call_getRange(opObj):
    """Function to call getRange method"""
    range = opObj.getRange()
    return range


def call_forward(opObj, add, model, data):
    """Function to call forward operator"""
    res = opObj.forward(add, model, data)
    return res


def call_adjoint(opObj, add, model, data):
    """Function to call adjoint operator"""
    res = opObj.adjoint(add, model, data)
    return res


def getNdfuture(vecObj):
    """Function to obtain NdArray as a future object"""
    Nd = vecObj.getNdArray()
    return Nd


def call_func_name(opObj, func_name, *args):
    """Function to call a method by name"""
    fun2call = getattr(opObj, func_name)
    res = fun2call(*args)
    return res


def _add_from_NdArray(vecObj, NdArray):
    """Function to add vector values from numpy array"""
    vecObj.getNdArray()[:] += NdArray
    return


def _check_dask_error(futures):
    """Function to check error on futures related to Dask operators"""
    for idx, fut in enumerate(futures):
        if fut.status == 'error':
            print("Error for dask operator %s" % idx)
            print(fut.result())
    return


class DaskOperator(Operator):
    """
    Class to apply multiple operators in parallel through Dask and DaskVectors
    """
    
    def __init__(self, dask_client, op_constructor, op_args, chunks, **kwargs):
        """
        Dask Operator constructor
        
        :param dask_client: [no default] - DaskClient;
            client object to use when submitting tasks (see dask_util module)
        :param op_constructor: [no default] - pointer to function or list of pointers to functions;
            Pointer to constructor(s)
        :param op_args: [no default] - list;
            List containing lists of arguments to run the constructor.
            It can instantiate the same operator on multiple workers or different ones if requested
            by passing a list of list of arguments (e.g., [(arg1,arg2,arg3,...)])
            If op_kind = blocky the order is column wise
        :param chunks: [no default] - list;
            List defining how many operators wants to instantiated.
            Note, the list must contain the same number of elements as the number of
            Dask workers present in the DaskClient.
        :param op_kind: [diag] - string;
            Mode to run the Dask Operator,
            diag = block diagonal operator
            blocky = blocky opearator (note: len(op_args) must be equal to np.sum(chunks)**2)
        :param setbackground_func_name: [None] - string;
            Name of the function to set the model point on which the Jacobian is computed.
            See NonLinearOperator in operator module.
        :param spread_op: [None] - DaskSpreadOp;
            Spreading operator to distribute a model vector to the set_background functions
        :param set_aux_name: [None] - string;
            Name of the function to set the auxiliary vector. Useful for VpOperator.
        :param spread_op_aux: [None] - DaskSpreadOp;
            Spreading operator to distribute an auxiliary vector to the set_aux functions
        """
        # Client to submit tasks
        if not isinstance(dask_client, DaskClient):
            raise TypeError("Passed client is not a Dask Client object!")
        if not isinstance(op_args, list):
            raise TypeError("Passed operator arguments not a list!")
        self.dask_client = dask_client
        self.client = self.dask_client.getClient()
        wrkIds = self.dask_client.getWorkerIds()
        N_wrk = self.dask_client.getNworkers()
        # Check if number of provided chunks is the same as workers
        if len(chunks) != N_wrk:
            raise ValueError(
                "Number of provide chunks (%s) different than the number of workers (%s)" % (len(chunks), N_wrk))
        # Check whether it is a blocky or block diagonal Dask operator
        self.op_kind = kwargs.get("op_kind", "diag")
        if self.op_kind not in "diag blocky":
            raise ValueError("Unknown op_kind provided (%s)" % self.op_kind)
        # Check if many arguments are passed to construct different operators
        N_args = len(op_args)
        N_ops = int(np.sum(chunks)) if self.op_kind == "diag" else int(np.sum(chunks))**2
        if N_args > 1:
            if N_args != N_ops:
                raise ValueError(
                    "Number of lists of arguments (%s) different than the number of requested operators (%s)" % (
                        N_args, N_ops))
        else:
            if N_ops > 1:
                op_args = [op_args for ii in range(N_ops)]

        # Check if kwargs for constructor was provided:
        op_kwargs = self.set_background_name = kwargs.get("op_kwargs", None)
        if op_kwargs is not None:
            N_kwargs = len(op_kwargs)
            if N_kwargs != N_args:
                raise ValueError("Length of kwargs (%d) different than args (%d)!" % (N_kwargs, N_args))
        else:
            op_kwargs = [None] * N_args
        
        # Instantiation of the operators on each worker
        self.dask_ops = []
        self.dask_ops_adj = []
        # Check if a list of constructors has been passed
        if isinstance(op_constructor, list):
            opt_list = op_constructor
        else:
            opt_list = [op_constructor]*N_ops
        self.n_col = 1 if self.op_kind == "diag" else int(np.sum(chunks))
        # Creating list of adjoint operators
        if self.n_col > 1:
            opt_list_adj = opt_list.copy()
            op_args_adj = op_args.copy()
            op_kwargs_adj = op_kwargs.copy()
            # Creating adjoint operators
            for iwrk, wrkId in enumerate(wrkIds):
                for iop in range(chunks[iwrk]):
                    for i_col in range(self.n_col):
                        self.dask_ops_adj.append(
                            self.client.submit(call_constructor,
                                               opt_list_adj.pop(0),
                                               op_args_adj.pop(0),
                                               op_kwargs_adj.pop(0),
                                               workers=[wrkId],
                                               pure=False))
        # Creating forward operators
        for i_col in range(self.n_col):
            for iwrk, wrkId in enumerate(wrkIds):
                for iop in range(chunks[iwrk]):
                    self.dask_ops.append(
                        self.client.submit(call_constructor,
                                           opt_list.pop(0),
                                           op_args.pop(0),
                                           op_kwargs.pop(0),
                                           workers=[wrkId],
                                           pure=False))
        daskD.wait(self.dask_ops)
        # Checking for errors during operators construction
        for idx, fut in enumerate(self.dask_ops):
            if fut.status == 'error':
                print("Error for dask operator %s" % idx)
                print(fut.result())
        # Creating domain and range of the Dask operator
        dom_vecs = []  # List of remote domain vectors
        rng_vecs = []  # List of remote range vectors
        op_list = self.dask_ops
        if self.n_col > 1:
            # Dealing with a blocky operator
            op_list = np.diag(np.asarray(self.dask_ops).reshape((self.n_col, self.n_col)).T)
        for op in op_list:
            dom_vecs.append(self.client.submit(call_getDomain, op, pure=False))
            rng_vecs.append(self.client.submit(call_getRange, op, pure=False))
        daskD.wait(dom_vecs + rng_vecs)
        _check_dask_error(dom_vecs + rng_vecs)
        self.domain = DaskVector(self.dask_client, dask_vectors=dom_vecs)
        self.range = DaskVector(self.dask_client, dask_vectors=rng_vecs)
        # Set background function name "necessary for non-linear operator Jacobian"
        self.set_background_name = kwargs.get("setbackground_func_name", None)
        if self.set_background_name:
            if self.op_kind != "diag":
                raise ValueError("Set background not currently supported for blocky operators")
            if not isinstance(self.set_background_name, list):
                self.set_background_name = [self.set_background_name] * len(self.dask_ops)
            # Creating a spreading operator useful for
            self.Sprd = kwargs.get("spread_op", None)
            if self.Sprd:
                if not isinstance(self.Sprd, DaskSpread):
                    raise TypeError("Provided spread_op not a DaskSpreadOp class!")
                self.model_tmp = self.Sprd.getRange().clone()
        # Set aux function name "necessary for VP operator"
        self.set_aux_name = kwargs.get("set_aux_name", None)
        if self.set_aux_name:
            if self.op_kind != "diag":
                raise ValueError("set_aux_name not currently supported for blocky operators")
            if not isinstance(self.set_aux_name, list):
                self.set_aux_name = [self.set_aux_name] * len(self.dask_ops)
            # Creating a spreading operator useful
            self.SprdAux = kwargs.get("spread_op_aux", None)
            if self.SprdAux:
                if not isinstance(self.SprdAux, DaskSpread):
                    raise TypeError("Provided spread_op_aux not a DaskSpreadOp class!")
                self.tmp_aux = self.SprdAux.getRange().clone()
        return

    def __str__(self):
        return " DaskOp "

    def forward(self, add, model, data):
        """Forward Dask operator"""
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        # Dimensionality check
        self.checkDomainRange(model, data)
        if self.op_kind == "diag":
            add = [add] * len(self.dask_ops)
            fwd_ftr = self.client.map(call_forward, self.dask_ops, add, model.vecDask, data.vecDask, pure=False)
            daskD.wait(fwd_ftr)
            _check_dask_error(fwd_ftr)
        else:
            add = [add] * self.n_col
            ops = np.asarray(self.dask_ops).reshape((self.n_col, self.n_col)).T
            for icol in range(self.n_col):
                fwd_ftr = self.client.map(call_forward, ops[:, icol], add,
                                          [model.vecDask[icol]] * self.n_col, data.vecDask, pure=False)
                daskD.wait(fwd_ftr)
                _check_dask_error(fwd_ftr)
                add = [True] * self.n_col
        return

    def adjoint(self, add, model, data):
        """Adjoint Dask operator"""
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        # Dimensionality check
        self.checkDomainRange(model, data)
        if self.op_kind == "diag":
            add = [add] * len(self.dask_ops)
            adj_ftr = self.client.map(call_adjoint, self.dask_ops, add, model.vecDask, data.vecDask, pure=False)
            daskD.wait(adj_ftr)
            _check_dask_error(adj_ftr)
        else:
            ops = np.asarray(self.dask_ops_adj).reshape((self.n_col, self.n_col)).T
            add = [add] * self.n_col
            for icol in range(self.n_col):
                adj_ftr = self.client.map(call_adjoint, ops[icol, :], add, model.vecDask,
                                          [data.vecDask[icol]] * self.n_col, pure=False)
                daskD.wait(adj_ftr)
                _check_dask_error(adj_ftr)
                add = [True] * self.n_col
        return
    
    def set_background(self, model):
        """Function to call set_background function of each dask operator"""
        if self.set_background_name == None:
            raise NameError("setbackground_func_name was not defined when constructing the operator!")
        if self.Sprd:
            self.Sprd.forward(False, model, self.model_tmp)
            model = self.model_tmp
        setbkg_ftr = self.client.map(call_func_name,
                                     self.dask_ops,
                                     self.set_background_name,
                                     model.vecDask,
                                     pure=False)
        daskD.wait(setbkg_ftr)
        return
    
    def set_aux(self, aux_vec):
        """Function to call set_nl or set_lin_jac functions of each dask operator"""
        if self.set_aux_name == None:
            raise NameError("set_aux_name was not defined when constructing the operator!")
        if self.SprdAux:
            self.SprdAux.forward(False, aux_vec, self.tmp_aux)
            aux_vec = self.tmp_aux
        setaux_ftr = self.client.map(call_func_name,
                                     self.dask_ops,
                                     self.set_aux_name,
                                     aux_vec.vecDask,
                                     pure=False)
        daskD.wait(setaux_ftr)
        return


class DaskSpread(Operator):
    """
    Class to spread/stack single vector to/from multiple copies on different workers:
         | v1 |   | I |
    fwd: | v2 | = | I | v       adj: | v | = | I | v1 + | I | v2 + | I | v3
         | v3 |   | I |
    """
    
    def __init__(self, dask_client, domain, chunks):
        """
        Dask Operator constructor
        
        :param dask_client: [no default] - DaskClient;
            client object to use when submitting tasks (see dask_util module)
        :param domain: [no default] - vector class;
            Vector template to be spread/stack (note this is also the domain of the operator)
        :param chunks: [no default] - list;
            List defininig how many operators wants to instantiated.
            It must have the same number of elements as the number of Dask workers present in the DaskClient.
        """
        if not isinstance(dask_client, DaskClient):
            raise TypeError("Passed client is not a Dask Client object!")
        if not isinstance(domain, Vector):
            raise TypeError("domain is not a vector-derived object!")
        self.dask_client = dask_client
        self.client = self.dask_client.getClient()
        self.chunks = chunks
        self.setDomainRange(domain, DaskVector(self.dask_client, vector_template=domain, chunks=chunks))
        return
    
    def __str__(self):
        return "DaskSprd"
    
    def forward(self, add, model, data):
        """Forward operator"""
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        # Model vector checking
        if isinstance(model, DaskVector):
            # Getting the future to the first vector in the Dask vector
            modelNd = self.client.submit(getNdfuture, model.vecDask[0], pure=False)
        else:
            # Getting the numpy array to the local model vector
            modelNd = model.getNdArray()
        
        # Spreading model array to workers
        if len(self.chunks) == self.dask_client.getNworkers():
            dataVecList = data.vecDask.copy()
            for iwrk, wrkId in enumerate(self.dask_client.getWorkerIds()):
                arrD = scatter_large_data(modelNd, wrkId, self.client)
                for ii in range(self.chunks[iwrk]):
                    daskD.wait(
                        self.client.submit(_add_from_NdArray, dataVecList.pop(0), arrD, workers=[wrkId], pure=False))
        else:
            # Letting Dask handling the scattering of the data (not ideal)
            futures = self.client.map(_add_from_NdArray,
                                      data.vecDask,
                                      [modelNd] * len(data.vecDask),
                                      pure=False)
            daskD.wait(futures)
        return
    
    def adjoint(self, add, model, data):
        """Adjoint operator"""
        if not isinstance(data, DaskVector):
            raise TypeError("Data vector must be a DaskVector!")
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        if isinstance(model, DaskVector):
            arrD = self.client.map(getNdfuture, data.vecDask, pure=False)
            daskD.wait(arrD)
            sum_array = self.client.submit(np.sum, arrD, axis=0, pure=False)
            daskD.wait(sum_array)
            # Getting the future to the first vector in the Dask vector
            daskD.wait(self.client.submit(_add_from_NdArray, model, sum_array, pure=False))
        else:
            arrD_list = data.getNdArray()
            # Getting the numpy array to the local model vector
            modelNd = model.getNdArray()
            for arr_i in arrD_list:
                modelNd[:] += arr_i
        return


class DaskCollect(Operator):
    """
    Class to Collect/Scatter a Dask vector into/from a local vector
    """
    
    def __init__(self, domain, range):
        """
        Dask Collect constructor
        
        :param domain : - DaskVector;
            Dask vector to be collected from remote
        :param range : - Vector;
            Vector class to be locally stored
        """
        if not isinstance(domain, DaskVector):
            raise TypeError("domain is not a DaskVector object!")
        if isinstance(range, DaskVector):
            raise TypeError("range should not a DaskVector object!")
        if not isinstance(range, Vector):
            raise TypeError("range is not a vector-derived object!")
        if domain.size != range.size:
            raise ValueError("number of elements in domain and range is not equal!")
        super(DaskCollect, self).__init__(domain, range)
    
    def forward(self, add, model, data):
        """Forward operator: collecting dask vector array to local one"""
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        self.checkDomainRange(model, data)
        if not add:
            data.zero()
        dataNd = data.getNdArray().ravel()
        # Obtaining remove array/s
        modelNd_list = model.getNdArray()
        idx = 0
        # Adding remove arrays to local one
        for arr in modelNd_list:
            n_elem = arr.size
            dataNd[idx:idx + n_elem] += arr.ravel()
            idx += n_elem
        return
    
    def adjoint(self, add, model, data):
        """Adjoint operator: scattering/distributing local array to remote vector"""
        if not isinstance(model, DaskVector):
            raise TypeError("Model vector must be a DaskVector!")
        self.checkDomainRange(model, data)
        if not add:
            model.zero()
        dataNd = data.getNdArray().ravel()
        # Shapes of the DaskVector chunks
        shapes = model.shape
        # Scattering local array
        client = model.client
        idx_el = 0
        for idx, shape in enumerate(shapes):
            n_elem = np.prod(shape)
            wrkId = list(client.who_has(model.vecDask[idx]).values())[0]
            arrD = client.scatter(dataNd[idx_el:idx_el + n_elem], workers=wrkId)
            daskD.wait(arrD)
            daskD.wait(client.submit(_add_from_NdArray, model.vecDask[idx], arrD, pure=False))
            idx_el += n_elem
        return
