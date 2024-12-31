"""Auto experiment module for the experiment framework.

This module provide template classes for automatic experiments. By implementing
experiment interfrace, the user can easily create their own experiments and run
batch experiments.

"""

from abc import ABC, abstractmethod
import sys
import os
import signal
import tqdm
import torch
import torch.multiprocessing as mp
import cupy
import termcolor


class AutoExperiment(ABC):
    """
    This is the abstract class for automatic experiments.
    """

    @abstractmethod
    def __init__(self):
        """
        This is the constructor of the class.
        """

    @abstractmethod
    def run(self):
        """
        This is the method that runs the experiment.
        """
    
    @abstractmethod
    def evaluate(self):
        """
        This is the method that evaluates the experiment.
        """

class ExperimentInterface(ABC):
    """
    This is the abstract class for the experiment interface.
    """

    @abstractmethod
    def load_parameters(self):
        """
        This is the method that parameterizes the experiment.
        """

    @abstractmethod
    def load_dataset(self):
        """
        This is the method that loads the dataset.
        """

    @abstractmethod
    def execute_experiment_process(self, parameters, dataset):
        """
        This is the method that processes the experiment.
        """
    
    @abstractmethod
    def summarize_results(self):
        """
        This is the method that summarizes the results.
        """

# Derived classes
class SimpleBatchExperiment(AutoExperiment):
    """
    This is the class for simple batch experiments.
    """

    def __init__(self, experiment_interface: ExperimentInterface, batch: int) -> None:
        """
        This is the constructor of the class.
        """
        self.experiment_interface = experiment_interface
        self.batch = batch
        self.parameter_group = self.experiment_interface.load_parameters()
        self.dataset = self.experiment_interface.load_dataset()

    def run(self):
        """
        Batch run the experiment.
        """
        progress_bar = tqdm.tqdm(total=len(self.parameter_group) * self.batch)
        for parameters in self.parameter_group:
            for _ in range(self.batch):
                self.experiment_interface.execute_experiment_process(parameters, self.dataset)
                progress_bar.update(1)
    
    def evaluate(self):
        """
        Evaluate the experiment.
        """
        self.experiment_interface.summarize_results()

# TODO(Ruhao Tian): Hide multiprocessing shared data handling from the user level
# TODO(Ruhao Tian): Create a derived experiment interface with device attribute
class CudaDistributedExperiment(AutoExperiment):
    """
    This is the class for CUDA distributed experiments.
    """

    def __init__(self, experiment_interface: ExperimentInterface, cuda: str | list[int] = "max", backend: str = "torch") -> None:
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
        self.dataset = self.experiment_interface.load_dataset()
        
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
    
    def _set_device(self, device):
        """Set the device for a specific process."""
        if device == "cpu":
            # disable CUDA for this process
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            if self.backend == "cupy":
                # throw a warning
                termcolor.cprint("Warning: CuPy has limited CPU support.", "yellow", attrs=["bold"])
                termcolor.cprint("Please make sure your code is agnostic.", "yellow", attrs=["bold"])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    
    def _sigint_handler(self, sig, frame):
        """
        Signal handler for SIGINT.
        """
        # close all processes
        termcolor.cprint("SIGINT received, terminating processes...", "yellow", attrs=["bold"])
        termcolor.cprint("Please wait. Force quitting may result in VRAM leakage.", "yellow", attrs=["bold"])
        for process in self.process_list:
            process.terminate()
        sys.exit(0)

    
    def _map_experiment(self, parameters, dataset, device):
        """
        Map the experiment process.
        """
        # set the device
        self._set_device(device)
        self.experiment_interface.execute_experiment_process(parameters, dataset)
        # return the device to the device queue
        self.device_queue.put(device)

    def run(self):
        """
        Run the experiment on the CUDA device(s) in parallel.
        """
        # TODO(Ruhao Tian): Implement the CUDA distributed experiment.
        # set the signal handler
        signal.signal(signal.SIGINT, self._sigint_handler)
        
        # create a progress bar
        progress_bar = tqdm.tqdm(total=len(self.parameter_group))
        
        # iterate over the parameter group
        for parameters in self.parameter_group:
            # get the device from the device queue
            device = self.device_queue.get()
            # execute the experiment process
            new_process = mp.Process(target=self._map_experiment, args=(parameters, self.dataset, device))
            new_process.start()
            self.process_list.append(new_process)
            # update the progress bar
            progress_bar.update(1)
        
        # wait for all processes to finish
        for process in self.process_list:
            process.join()
            
    
    def evaluate(self):
        """
        Evaluate the experiment.
        """
        self.experiment_interface.summarize_results()
