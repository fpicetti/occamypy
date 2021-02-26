import atexit
import json
import os
import random
import socket
import subprocess
from time import time

DEVNULL = open(os.devnull, "wb")
import dask.distributed as daskD
from dask_jobqueue import PBSCluster, LSFCluster, SLURMCluster
from dask_kubernetes import KubeCluster, make_pod_spec


def get_tcp_info(filename):
    """Function to obtain scheduler tcp information"""
    tcp_info = None
    with open(filename) as json_file:
        data = json.load(json_file)
    if "address" in data:
        tcp_info = data["address"]
    return tcp_info


def create_hostnames(machine_names, Nworkers):
    """
    Function to create hostnames variables (i.e., list of ip addresses)
    from machine names and number of wokers per machine
    """

    ip_adds = []
    for host in machine_names:
        ip_adds.append(socket.gethostbyname(host))

    if len(Nworkers) != len(ip_adds):
        raise ValueError("Lenght of number of workers (%s) not consistent with number of machines available (%s)"
                         % (len(Nworkers), len(ip_adds)))

    hostnames = []
    for idx, ip in enumerate(ip_adds):
        hostnames += [ip] * Nworkers[idx]
    return hostnames


def client_startup(cluster, n_jobs: int, total_workers: int):
    """
    Function to start a dask client
    """
    if n_jobs <= 0:
        raise ValueError("n_jobs must equal or greater than 1!")
    if isinstance(cluster, daskD.LocalCluster) or isinstance(cluster, KubeCluster):
        cluster.scale(n_jobs)
    else:
        cluster.scale(jobs=n_jobs)
    # Creating dask Client
    client = daskD.Client(cluster)
    workers = 0
    t0 = time()
    while workers < total_workers:
        workers = len(client.get_worker_logs().keys())
        # If the number of workers is not reached in 5 minutes raise exception
        if time() - t0 > 300.0:
            raise SystemError("Dask could not start the requested workers within 5 minutes!"
                              "Try different n_jobs.")
    WorkerIds = list(client.get_worker_logs().keys())
    return client, WorkerIds


class DaskClient:
    """
    Class useful to construct a Dask Client to be used with Dask vectors and operators
    """

    def __init__(self, **kwargs):
        """
    Constructor for obtaining a client to be used when Dask is necessary
    1) Cluster with shared file system and ssh capability:
    :param hostnames : - list; list of strings containing the host names or IP addresses of the machines that
    the user wants to use in their cluster/client (First hostname will be running the scheduler!) [None]
    :param scheduler_file_prefix : string; prefix to used to create dask scheduler-file.
    :param logging : - boolean; Logging scheduler and worker stdout to files within dask_logs folder [True]
    Must be a mounted path on all the machines. Necessary if hostnames are provided [$HOME/scheduler-]
    2) Local cluster:
    :param local_params : - dict; dictionary containing Local Cluster options (see help(LocalCluster) for help) [None]
    :param n_wrks: - int; number of workers to start [1]
    3) PBS cluster:
    :param pbs_params : - dict; dictionary containing PBS Cluster options (see help(PBSCluster) for help) [None]
    :param n_jobs : - int; number of jobs to be submitted to the cluster
    :param n_wrks: - int; number of workers per job [1]
    4) LSF cluster:
    :param lfs_params : - dict; dictionary containing LSF Cluster options (see help(LSFCluster) for help) [None]
    :param n_jobs : - int; number of jobs to be submitted to the cluster
    :param n_wrks: - int; number of workers per job [1]
    5) SLURM cluster:
    :param slurm_params : - dict; dictionary containing SLURM Cluster options (see help(SLURMCluster) for help) [None]
    :param n_jobs : - int; number of jobs to be submitted to the cluster
    :param n_wrks: - int; number of workers per job [1]
    6) Kubernetes cluster:
    :param kube_params : - dict; dictonary containing KubeCluster options
     (see help(KubeCluster) and help(make_pod_spec) for help) [None]
    :param n_wrks: - int; number of workers to scale the cluster
    Note that by default the Kubernetes pods are created using the Docker image "ettore88/occamypy:devel". To change
    the image to be use, provide the item image within the kube_params dictionary.
    """
        hostnames = kwargs.get("hostnames", None)
        local_params = kwargs.get("local_params", None)
        pbs_params = kwargs.get("pbs_params", None)
        lsf_params = kwargs.get("lsf_params", None)
        slurm_params = kwargs.get("slurm_params", None)
        kube_params = kwargs.get("kube_params", None)
        logging = kwargs.get("logging", True)
        ClusterInit = None
        cluster_params = None
        if local_params:
            cluster_params = local_params
            ClusterInit = daskD.LocalCluster
        elif pbs_params:
            cluster_params = pbs_params
            ClusterInit = PBSCluster
        elif lsf_params:
            cluster_params = lsf_params
            ClusterInit = LSFCluster
        elif slurm_params:
            cluster_params = slurm_params
            ClusterInit = SLURMCluster
        # Checking interface to be used
        if hostnames:
            if not isinstance(hostnames, list):
                raise ValueError("User must provide a list with host names")
            scheduler_file_prefix = kwargs.get("scheduler_file_prefix", os.path.expanduser("~") + "/scheduler-")
            # Random port number
            self.port = ''.join(["1"] + [str(random.randint(0, 9)) for _ in range(3)])
            # Creating logging interface
            stdout_scheduler = DEVNULL
            stdout_workers = [DEVNULL] * len(hostnames)
            if logging:
                # Creating logging folder
                try:
                    os.mkdir("dask_logs")
                except OSError:
                    pass
                stdout_scheduler = open("dask_logs/dask-scheduler.log", "w")
                stdout_workers = [open("dask_logs/dask-worker-%s.log" % (ii + 1), "w") for ii in range(len(hostnames))]
            # Starting scheduler
            scheduler_file = "%s%s" % (scheduler_file_prefix, self.port) + ".json"
            cmd = ["ssh"] + [hostnames[0]] + \
                  ["dask-scheduler"] + ["--scheduler-file"] + [scheduler_file] + \
                  ["--port"] + [self.port]
            self.scheduler_proc = subprocess.Popen(cmd, stdout=stdout_scheduler, stderr=subprocess.STDOUT)
            # Checking if scheduler has started and getting tpc information
            t0 = time()
            while True:
                if os.path.isfile(scheduler_file):
                    if get_tcp_info(scheduler_file): break
                # If the dask scheduler is not started in 5 minutes raise exception
                if time() - t0 > 300.0:
                    raise SystemError("Dask could not start scheduler! Try different first host name.")
            # Creating dask Client
            self.client = daskD.Client(scheduler_file=scheduler_file)
            # Starting workers on all the other hosts
            self.worker_procs = []
            worker_ips = []
            for ii, hostname in enumerate(hostnames):
                cmd = ["ssh"] + [hostname] + ["dask-worker"] + ["--scheduler-file"] + [scheduler_file]
                # Starting worker
                self.worker_procs.append(subprocess.Popen(cmd, stdout=stdout_workers[ii], stderr=subprocess.STDOUT))
                # Obtaining IP address of host for the started worker (necessary to resort workers)
                worker_ips.append(
                    subprocess.check_output(
                        ["ssh"] + [hostname] + ["hostname -I"] + ["| awk '{print $1}'"]).rstrip().decode("utf-8"))
            # Waiting until all the requested workers are up and running
            workers = 0
            requested = len(hostnames)
            t0 = time()
            while workers < requested:
                workers = len(self.client.get_worker_logs().keys())
                # If the number of workers is not reached in 5 minutes raise exception
                if time() - t0 > 300.0:
                    raise SystemError(
                        "Dask could not start the requested workers within 5 minutes! Try different hostnames.")
            # Resorting worker IDs according to user-provided list
            self.WorkerIds = []
            wrkIds = list(self.client.get_worker_logs().keys())  # Unsorted workers ids
            wrk_ips = [idw.split(":")[1][2:] for idw in wrkIds]  # Unsorted ip addresses
            for ip in worker_ips:
                idx = wrk_ips.index(ip)
                self.WorkerIds.append(wrkIds[idx])
                wrkIds.pop(idx)
                wrk_ips.pop(idx)
        elif kube_params:
            n_wrks = kwargs.get("n_wrks")
            if "image" not in kube_params:
                kube_params.update({"image": 'ettore88/occamypy:devel'})
            pod_spec = make_pod_spec(**kube_params)
            self.cluster = KubeCluster(pod_spec, deploy_mode="remote")
            self.client, self.WorkerIds = client_startup(self.cluster, n_wrks, n_wrks)
        elif ClusterInit:
            n_wrks = kwargs.get("n_wrks", 1)
            if n_wrks <= 0:
                raise ValueError("n_wrks must equal or greater than 1!")
            if "local_params" in kwargs:
                # Starting local cluster
                n_jobs = n_wrks
                n_wrks = 1
            else:
                # Starting scheduler-based clusters
                n_jobs = kwargs.get("n_jobs")
                if n_jobs <= 0:
                    raise ValueError("n_jobs must equal or greater than 1!")
                cluster_params.update({"processes": n_wrks})
                if n_wrks > 1:
                    # forcing nanny to be true (otherwise, dask-worker command will fail)
                    cluster_params.update({"nanny": True})
            self.cluster = ClusterInit(**cluster_params)
            self.client, self.WorkerIds = client_startup(self.cluster, n_jobs, n_jobs * n_wrks)
        else:
            raise ValueError("Either hostnames or local_params or pbs/lsf/slurm_params or kube_params must be "
                             "provided!")
        # Closing dask processes
        atexit.register(self.client.shutdown)

    def getClient(self):
        """
    Accessor for obtaining the client object
    """
        return self.client

    def getWorkerIds(self):
        """
    Accessor for obtaining the worker IDs
    """
        return self.WorkerIds

    def getNworkers(self):
        """
    Accessor for obtaining the number of workers
    """
        return len(self.getWorkerIds())
