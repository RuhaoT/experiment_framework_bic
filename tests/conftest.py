import pytest
import logging
import shutil
from pathlib import Path

# clear/create the log directory for each test run
@pytest.fixture(scope='session', autouse=True)
def setup_session_logging_dir(pytestconfig) -> Path:
    log_dir = pytestconfig.rootpath / 'logs'
    # if the log directory exists, remove it
    if log_dir.exists():
        shutil.rmtree(log_dir)
    # create a new log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

# module-level fixture to set up logging directory
@pytest.fixture(scope='module')
def setup_module_logging_dir(setup_session_logging_dir: Path, request) -> Path:
    
    module_name = request.module.__name__
    module_dir = setup_session_logging_dir / module_name
    module_dir.mkdir(parents=True, exist_ok=True)
    return module_dir

# test-case-level fixture to handle logging
@pytest.fixture(scope='function', autouse=True)
def case_log_handler(setup_module_logging_dir: Path, request):
    # Create a logger for the test case
    root_logger = logging.getLogger()
    # backup logging level
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

    # Create a file handler for the log file
    log_file = setup_module_logging_dir / f"{request.node.name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    root_logger.addHandler(file_handler)
    
    root_logger.info(f"Starting test case: {request.node.name}")

    yield root_logger
    
    root_logger.info(f"Finished test case: {request.node.name}")

    # Remove the handler after the test case is done
    root_logger.removeHandler(file_handler)
    
    # Restore the original logging level
    root_logger.setLevel(original_level)