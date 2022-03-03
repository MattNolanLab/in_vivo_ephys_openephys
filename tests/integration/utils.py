import subprocess
import shutil
from pathlib import Path
from functools import wraps
from typing import Callable


def is_running_in_docker() -> bool:
    """Returns true if the script is being ran in a Docker container"""
    return Path('/.dockerenv').exists()


def require_docker(func: Callable) -> Callable:
    """A decorator that throws an EnvironmentError if the wrapped function is not called from a Docker environment"""
    @wraps(func)
    def wrap(*args, **kwargs) -> Callable:
        if is_running_in_docker():
            return func(*args, **kwargs)
        else:
            raise EnvironmentError('These integration test functions are potentially destructive and should only be run in a temporary Docker container/CI')
    return wrap


def set_up(raw_output_path: str, data_url: str, test_data_name:str) -> None:

    output_path = Path(raw_output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # With the folders set up, we're ready to fetch the test data.
        # This cryptic shell command downloads data (curl), decrypts it, and unzips it (tar). We don't have to worry about creating
        # intermediary files because the pipes (| characters) pass the data between each command using the magic of linux.
        subprocess.check_call(
            f'curl {data_url} | tar xzfv - -C {output_path}',
            shell=True)
    except subprocess.CalledProcessError as ex:
        raise Exception(f'Encountered an error while fetching the integration test data') from ex

    try:
        subprocess.check_call(f'bash -lc "./runSnake.py {output_path}/{test_data_name}"', 
        shell=True)
    except subprocess.CalledProcessError as ex:
        raise Exception('Failed to successfully run the sorting pipeline') from ex

def tear_down(raw_output_path: str) -> None:
    """Perform cleanup operations

    This copies result files to the artifacts directory so we can look at them later
    """
    output_path = Path(raw_output_path)
    shutil.rmtree(output_path)
    