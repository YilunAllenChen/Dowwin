# Dowwin
The new Project Dowwin.

## Documentation
See our [Coda](https://coda.io/d/Project-Dowwin-v1-0_dP7-RL3GJyj/Project-Planning_surrt#_luF_I) page for documentation.

## Running the project

### Setting up the environment

Necessary packages are contained in the [tradebot.yml](./tradebot.yml). To 
create an environment using this configuration file, ensure that you have a 
package manager installed on your machine (example: Anaconda/Miniconda, pip).
Run the following commands:

*Anaconda:*

```shell
conda env create -f tradebot.yml
```

**Note: Windows 10 users are required to install the [Microsoft Message Passing Interface](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) functionality.** This is required by one of our dependencies, *stable-baselines*.
