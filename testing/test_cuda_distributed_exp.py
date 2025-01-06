import numpy as np
import torch.multiprocessing as mp
import auto_experiment.auto_experiment as auto_experiment
import auto_experiment.cuda_distributed_experiment as cuda_distributed_experiment


manager = mp.Manager()

# Create an instance of the ExperimentInterface class
class experiment_interface(auto_experiment.ExperimentInterface):
    
    def __init__(self):
        self.params = np.arange(1000).tolist()
        self.results = manager.list()
    
    def load_parameters(self):
        return self.params

    def load_dataset(self):
        return "dataset"
    
    def execute_experiment_process(self, parameters, dataset):
        
        self.results.append(parameters)
        
    
    def summarize_results(self):
        # sort self.results
        self.results = list(self.results)
        self.results.sort()
        
        self.params.sort()
        
def test_cuda_distributed_experiment():
    
    experiment = cuda_distributed_experiment.CudaDistributedExperiment(experiment_interface(), cuda="max")
    experiment.run()
    experiment.evaluate()
    
    assert experiment.experiment_interface.results == experiment.experiment_interface.params