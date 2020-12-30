# Dowwin
The new Project Dowwin.

## Documentation
See our [Coda](https://coda.io/d/Project-Dowwin-v1-0_dP7-RL3GJyj/Project-Planning_surrt#_luF_I) page for documentation.

## Running the project

### Setting up the environment

Necessary packages are contained in the [core.yml](./tradebot.yml) and [core.txt](./core.txt). The core environment is managed by Anaconda. Some packages is managed by pip. To set up this environment, first make sure that you have the appropriate version of Anaconda installed.

Run the following command to create the conda environment:

```shell
conda env create -f core.yml
```

Assuming that no changes are made to the *core.yml* and *core.txt* files, &lt;environment name&gt; should be *dowwin*.

Some packages are managed by pip. After running the previous command, activate the newly created Anaconda environment and install these additional dependencies with the following command:

```shell
conda activate <environment name>
pip install -r core.txt
```

You should now have an environment that can run the functionality in the core module.

To run any functionality contained in the core module, ensure that you activate the environment first using the following command:

```shell
conda activate <environment names>
```

After you are done, deactivate the environment using the following command:
```shell
conda deactivate
```

### Additional notess

- **Windows 10 users are required to install the [Microsoft Message Passing Interface](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) functionality.** This is required by one of our dependencies, *stable-baselines*.
- **You should run** *conda* **on the default terminal** (example: Windows command prompt, Mac OS terminal, Unix bash). To use *conda* on your preferred terminal, you will need to initialize conda for that terminal. One way to do this is to use the following command:

    ```shell
    conda init <your terminal name>
    ```

    Example:

    ```shell
    conda init powershell
    ```

    let windows users access conda on PowerShell.

    **You will need to restart your terminal for the changes to take effect.**

    Please consider other sources if the above solution does not work for you.
