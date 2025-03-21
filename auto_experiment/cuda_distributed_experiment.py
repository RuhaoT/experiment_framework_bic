import os
import random
import signal
import sys

import cupy
import termcolor
import torch
import torch.multiprocessing as mp
import tqdm

from . import auto_experiment


def _set_device(device, backend) -> int:
    """Set the device for a specific process.
    
    Return the device index in terms of process.
    """
    if device == "cpu":
        # disable CUDA for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if backend == "cupy":
            # throw a warning
            termcolor.cprint(
                "Warning: CuPy has limited CPU support.", "yellow", attrs=["bold"]
            )
            termcolor.cprint(
                "Please make sure your code is agnostic.", "yellow", attrs=["bold"]
            )
        return 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        return device

def _map_data_loading(data_queue, data_function):
    """Map the data loading"""
    
    data_queue.put(data_function())
    

def _map_experiment(
    parameters, dataset, device, backend: str, device_queue: mp.Queue, exp_func
):
    """
    Map the experiment process.
    """
    # set the device
    device_id = _set_device(device=device, backend=backend)

    # if parameters is a list, then it is a batch experiment
    if isinstance(parameters, list):

        # create a tqdm progress bar at device index position
        for parameter in tqdm.tqdm(
            parameters, position=device_id, desc=f"Device {device}", mininterval=1
        ):
            exp_func(parameter, dataset)
    else:
        exp_func(parameters, dataset)
    # return the device to the device queue
    device_queue.put(device)


# TODO(Ruhao Tian): Hide multiprocessing shared data handling from the user level
# TODO(Ruhao Tian): Create a derived experiment interface with device attribute
class CudaDistributedExperiment(auto_experiment.AutoExperiment):
    """
    This is the class for CUDA distributed experiments.
    """

    def __init__(
        self,
        experiment_interface: auto_experiment.ExperimentInterface,
        cuda: str | list[int] = "max",
        backend: str = "torch",
    ) -> None:
        """
        This is the constructor of the class.

        Args:
            experiment_interface (ExperimentInterface): The experiment interface.
            cuda (str | list[int]): The CUDA device(s) to use. Use "max" to use
                the maximum available CUDA device. Use "none" to use CPU. Use a
                list of CUDA device IDs to specify the CUDA devices.
            backend (str): The backend to use. Currently only "torch" and "cupy"
                are supported.
        """
        self.backend = backend
        # check backend
        if self.backend not in ["torch", "cupy"]:
            raise ValueError("Invalid backend.")

        # check and set the CUDA device(s)
        self.devices = []
        if cuda == "none":
            self.devices = ["cpu"]
        else:
            if cuda == "max":
                cuda = None
            self.devices = self._check_cuda_device(cuda)

        self.experiment_interface = experiment_interface
        self.parameter_group = self.experiment_interface.load_parameters()

        # prepare for multiprocessing
        # create a device Queue act as GPU pool
        self.device_queue = mp.Manager().Queue()
        for device in self.devices:
            self.device_queue.put(device)
        # create a process list
        self.process_list = []

    def _check_cuda_device(self, selection: int | list[int] = None) -> list:
        """Check the device selection. If no selection provided, return all devices.

        Args:
            selection (str | list[str], optional): Devices to check. Defaults to None.

        Returns:
            list[str]: available devices
        """

        # check device count
        if self.backend == "torch":
            device_count = torch.cuda.device_count()
        else:
            device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count == 0:
            raise ValueError("No CUDA device available.")

        # check selection
        if selection is None:
            # because different backend has different device name format
            # use index to represent the device and serialize later
            return [i for i in range(device_count)]

        if not isinstance(selection, list):
            selection = [selection]

        # check each device
        for device in selection:
            if device < 0 or device >= device_count:
                raise ValueError("Invalid device selection.")

        return selection

    def _sigint_handler(self, sig, frame):
        """
        Signal handler for SIGINT.
        """
        # close all processes
        termcolor.cprint(
            "SIGINT received, terminating processes...", "yellow", attrs=["bold"]
        )
        termcolor.cprint(
            "Please wait. Force quitting may result in VRAM leakage.",
            "yellow",
            attrs=["bold"],
        )
        for process in self.process_list:
            process.terminate()
        sys.exit(0)

    def run(self):
        """
        Run the experiment on the CUDA device(s) in parallel.
        """
        # TODO(Ruhao Tian): Implement the CUDA distributed experiment.
        # set the signal handler
        signal.signal(signal.SIGINT, self._sigint_handler)
        
        # spawn an isolated process to load dataset
        # loading dataset in parent process may cause deadlock
        # an issue with OpenMP backend in NumPy & PyTorch
        data_queue = mp.Queue()
        data_process = mp.Process(
            target=_map_data_loading,
            args=(data_queue, self.experiment_interface.load_dataset),
        )
        data_process.start()
        self.dataset = data_queue.get()
        data_process.join()
        

        # split the parameter group uniformly to each device
        # shuffle the parameter group
        random.shuffle(self.parameter_group)
        chunk_num = min(len(self.parameter_group), len(self.devices))
        chunk_size = len(self.parameter_group) // chunk_num

        termcolor.cprint(f"Running on {len(self.devices)} devices.", attrs=["bold"])

        # iterate over the parameter group
        for chunk in range(chunk_num):
            # get the device from the device queue
            device = self.device_queue.get()

            # get current parameter chunk
            if chunk == chunk_num - 1:
                current_chunk = self.parameter_group[chunk * chunk_size :]
            else:
                current_chunk = self.parameter_group[
                    chunk * chunk_size : (chunk + 1) * chunk_size
                ]

            # execute the experiment process
            new_process = mp.Process(
                target=_map_experiment,
                args=(
                    current_chunk,
                    self.dataset,
                    device,
                    self.backend,
                    self.device_queue,
                    self.experiment_interface.execute_experiment_process,
                ),
            )
            new_process.start()
            self.process_list.append(new_process)

        # wait for all processes to finish
        for process in self.process_list:
            process.join()

    def evaluate(self):
        """
        Evaluate the experiment.
        """
        self.experiment_interface.summarize_results()
