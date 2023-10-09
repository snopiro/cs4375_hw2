# CS4375 Assignment 2
GitHub Repo: https://github.com/snopiro/cs4375_hw2

## Project Description
This is a an assignment for my Machine Learning class.

## How to Run
There are several ways you can run this project. I have provided docker files to provide an execution environment if you have Docker installed. Alternatively, you can also set up a Github Codespaces environment and run the docker files from there. You can also run the project locally if you have python installed on your machine.

After running the project, it will create log files and plots in the src/files directory.

### Github Codespaces (Recommended)

If you would like to use Github Codespaces, click the green button at the top of this repository that says "Code" and select "Create codespace on master". This will open a new Codespaces environment with the project already cloned. Once the environment is ready, navigate to the root directory of this project, where the docker-compose.yml file is located and run the following command in the terminal:
```
docker compose up
```

### Running the project locally

If you have python already installed on your machine, as well as the libraries
```
numpy
pandas
matplotlib
```
you can run the project locally by navigating to the src directory of this project and running the following command:
```
python main.py
```
