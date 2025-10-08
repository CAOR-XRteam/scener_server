# Server (3D Scene Generation Platform)

## Overview

This document provides detailed instructions for the setup and operation of the server component for the [3D Scene Generation Platform](https://github.com/arteume/Scener).

The server is a Python-based application responsible for:
*   Managing persistent WebSocket connections with clients.
*   Receiving and interpreting requests for 3D scene generation/modification.
*   Orchestrating and dispatching tasks to various AI generation pipelines.
*   Structuring and transmitting responses back to the client.

## Prerequisites

Before proceeding, ensure your system meets the following requirements.

### Hardware
*   **GPU:** A GPU with a minimum of **CALCULATE** is required to run all AI pipelines simultaneously.
*   **System:** A Linux-based operating system (Ubuntu 20.04+ is recommended).

### Software
*   **Git:** For cloning the repository.
*   **Conda / Miniconda:** For Python environment management.
*   **Sudo/root access:** Required for installing system-level dependencies.

## Installation

Two installation paths are provided: a pre-configured Conda environment file or a full manual installation of all python dependencies. We strongly recommend users with NVIDIA RTX 5090 GPUs use the provided Conda environment, since the manual package installation is complicated due to incompatible PyTorch/CUDA version required by the TRELLIS library.

---

### Method A: Automatic Installation (TODO)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/CAOR-XRteam/scener_server
    cd Scener
    ```

2.  **Create the Conda Environment:**
    For NVIDIA RTX 5090 GPUs:
    ```bash
    conda env create -f environment-5090.yml
    ```

   For lower generations graphic cards:
    ```bash
    conda env create -f environment-rest.yml
    ```
    
3.  **Activate the Environment:**
    You must activate this environment in any new terminal session before running the server.
    ```bash
    conda activate scener
    ```

---

### Method B: Manual Installation

1.  **Clone this Repository:**
    ```bash
    git clone https://github.com/CAOR-XRteam/scener_server
    cd Scener
    ```

2.  **Install the TRELLIS Library:**
    This project depends heavily on the `TRELLIS` image-to-3D library. You must first follow the official installation instructions from the Microsoft repository:
    
    **[https://github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)**
    
    **Take note of the Conda environment you create during their setup process, as you will need to install this project's dependencies into that same environment.**

3.  **Install Project Dependencies:**
    After successfully installing TRELLIS, activate its Conda environment. Then, from the root of this project's directory (`Scener`), run the following command to install the remaining packages:
    ```bash
    pip install -e . -c constraints.txt
    ```
    *   `-e .`: Installs the project in "editable" mode.
    *   `-c constraints.txt`: Ensures that package versions adhere to the specified constraints, preventing conflicts.

## Dependency Configuration

After setting up the Python environment, you must install and configure Redis and Ollama.

### 1. Redis Server

Redis is an in-memory data structure store ([learn more](https://redis.io/)) utilized to store serialized scene JSON data.

1.1.  **Install Redis:**
    ```bash
    sudo apt update
    sudo apt install redis-server
    ```

1.2.  **Configure Redis for Network Access:**
    You must edit the configuration file to allow the server to connect from non-local addresses.
    ```bash
    sudo nano /etc/redis/redis.conf
    ```
    Make the following two changes inside the file:
    *   Find the line `bind 127.0.0.1 ::1` and change it to `bind 0.0.0.0`. This makes Redis listen on all available network interfaces.
    *   Find the line `protected-mode yes` and change it to `protected-mode no`.

    
    > Disabling protected mode without a password is not recommended for production environments.

1.3.  **Apply Changes and Enable Service:**
    Restart the Redis service to apply the new configuration and enable it to start on boot.
    ```bash
    sudo systemctl restart redis-server
    sudo systemctl enable redis-server
    ```

1.4.  **Verify Redis is Running:**
    Check that Redis is listening on port `6379` for all interfaces.
    ```bash
    sudo ss -tuln | grep 6379
    ```
    The expected output should contain `LISTEN` and `0.0.0.0:6379`.

### 2. Ollama

Ollama serves the Large Language Models (LLMs).

2.1.  **Install Ollama:**
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

2.2.  **Run the Ollama Service:**
    Open a new terminal or a `tmux` session and run the following command to start the Ollama server.
    ```bash
    ollama serve
    ```

2.3.  **Pull Required LLM Models:**
    You must download the specific models required by the project, which are defined in `config.json` under variables with `_model` suffix. Check the file for the model names. For example, if the config requires `llama3.1`, run:
    ```bash
    ollama pull llama3.1
    ```
    Repeat this for every model listed in the configuration.

## Environment Variables

The server requires API keys and network configuration, which are managed via an `.env` file.

1.  **Create the `.env` file:**
    Copy the example template to create your local configuration file.
    ```bash
    cp env_example .env
    ```

2.  **Edit the `.env` file:**
    Open the `.env` file and fill in thhttps://huggingface.co/stabilityai/stable-diffusion-3.5-mediume values for each variable:
    *   `HF_API_KEY`: Your Hugging Face API key (used for text-to-image AI models).
           You must also ask to be granted access to the stable difussion model, [see](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).
    *   `WEBSOCKET_HOST`: The host address for the server (e.g., `0.0.0.0`).
    *   `WEBSOCKET_PORT`: The port for the server (e.g., `8080`).
    *   `REDIS_HOST`: The IP address of your Redis server (e.g., `0.0.0.0`).
    *   `REDIS_PORT`: The port for your Redis server (default is `6379`).

## Running the Server

Once all previous steps are complete, you can launch the server.

1.  **Activate the Conda Environment:**
    ```bash
    conda activate scener
    ```

2.  **Run the Server using Makefile:**
    From the project root directory, execute:
    ```bash
    make run_server
    ```

The server should now be running and ready to accept WebSocket and Redis connections.

## Technical Architecture

