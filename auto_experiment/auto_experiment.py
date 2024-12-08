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
        
class CudaDistributedExperiment(AutoExperiment):
    """
    This is the class for CUDA distributed experiments.
    """

    def __init__(self, experiment_interface: ExperimentInterface, cuda: str | list[str] = "max") -> None:
        """
        This is the constructor of the class.
        
        Args:
            experiment_interface (ExperimentInterface): The experiment interface.
            cuda (str | list[str]): The CUDA device(s) to use. Use "max" to use
                the maximum available CUDA device. Use "none" to use CPU. Use a
                list of CUDA device IDs to specify the CUDA devices.
        """
        
        # check and set the CUDA device(s)
        self.devices = []
        self.device_count = 0
        if cuda == "none":
            self.devices = ["cpu"]
        elif cuda == "max":
            # get device count
            self.device_count = torch.cuda.device_count()
            # check and append all available devices
            if self.device_count == 0:
                # raise error if no CUDA device is available
                raise ValueError("No CUDA device is available.")
            for i in range(self.device_count):
                self.devices.append(f"cuda:{i}")
        elif isinstance(cuda, list):
            self.devices = [f"cuda:{i}" for i in cuda]
        
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
        with torch.cuda.device(device):
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
