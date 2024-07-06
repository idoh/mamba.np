## Prerequisites

-   **Python**: This project requires Python version 3.8 or higher. It has been tested on Python 3.10.

### Environment Setup

1. **Clone the Repository**

    First, clone the repository to your local machine:

    ```bash
    git clone git@github.com:idoh/mamba.np.git
    cd mamba.np
    ```

2. **Set Up Virtual Environment**

    It is recommended to use a virtual environment to manage dependencies. You can set up a virtual environment using `venv`:

    ```bash
    python3 -m venv venv
    ```

3. **Activate Virtual Environment**

    - On **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    - On **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4. **Install Required Packages**

    Install the required packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

5. **Install PyTorch**

    Install PyTorch from the official [PyTorch website](https://pytorch.org/get-started/locally/). Choose the appropriate configuration for your system. For example, for a basic CPU-only installation on Windows/Mac, you can use:

    ```bash
    pip install torch
    ```

### Notes

-   Ensure your Python version is at least 3.8. You can check your Python version by running:
    ```bash
    python --version
    ```
