import sys
import os
import signal
import platform


def clear_terminal():
    # Check the operating system
    system_name = platform.system()

    if system_name == "Windows":
        os.system("cls")  # Windows command to clear the terminal
    else:
        os.system("clear")  # UNIX-based systems (Linux, macOS) use this

def check_version():
    # Get the current Python version
    current_version = sys.version_info

    # Check if the Python version is either 3.9 or 3.10
    if current_version.major != 3 or current_version.minor not in [9, 10]:
        raise EnvironmentError("[error] This script requires Python 3.9 or 3.10. Please use the correct Python version.")

    # Rest of your code goes here
    print("Python version is valid. Continuing with the script...")
