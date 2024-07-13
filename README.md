# mofu-chan: Personal Investing Planner
This is part of the document for prototype of “HumanAIze Hackathon <FinTech Edition>”. Please find more detail of the idea submission [here](https://medium.com/@npatamawadee/humanaize-hackathon-mofu-chan-personal-investing-planner-3acf93659efb).


## Environment Setup
- Create docker image:
    ```
    make build #build docker image name mofuchan
    ```
- Update the docker environment
    1. conda environment
        - Edit `environment.yml`
        - `make conda_lock` to update the conda lock file. (Need to install conda-lock in your local)
    2. Update the dependencies:
        -  `poetry add <library>` (Need to install poetry in your local)
    3. Rebuild the docker image

## Run docker container
Please add all the environment variables in `.env` file.
- Run with bash:
    ```
    make run-bash OPTS="<add_parameter>" #create container to run the notebook
    ```
    ** Add addtional parameter to run the container in <add_parameter>
- Run the notebook:
    ```
    make run-notebook OPTS="<add_parameter>" #create container to run the notebook
    ```
- Delete the notebook:
    ```
    make rm-bash
    ```

## Structure
The structure of the source code
```
.
├── .jupyter                     # configuration file for jupyter notebook
├── dockerfiles
├── notebook                     # jupyter notebook files 
├── src                          # main source code
|    ├── ....  
├── tests                        # unittest (if needed)
├── Makefile  
└── README.md
```



## Pre-commit

To automatically format please use pre-commit.

1. Install pre-commit

```commandline
pip install pre-commit
```

2. Install git hook scripts

```commandline
pre-commit install
```