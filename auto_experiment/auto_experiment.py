"""Auto experiment module for the experiment framework.

This module provide template classes for automatic experiments. By implementing
experiment interfrace, the user can easily create their own experiments and run
batch experiments.

"""

from abc import ABC, abstractmethod
import tqdm
import torch


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

    def run(self):
        """
        Run the experiment on the CUDA device(s) in parallel.
        """
        # TODO(Ruhao Tian): Implement the CUDA distributed experiment.
    
    def evaluate(self):
        """
        Evaluate the experiment.
        """
        self.experiment_interface.summarize_results()
