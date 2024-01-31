# Comp442/542 Assigment

This assignment is adapted from [Stanford Course cs231n](http://cs231n.stanford.edu/).

## Setup Instructions

**Installing Anaconda:** If you decide to work locally, we recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/), which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version, which currently installs Python 3.8.

**Anaconda Virtual environment:** Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run (in a terminal)

`conda create -n comp442 python=3.8`

to create an environment called comp442.

Then, to activate and enter the environment, run

`conda activate comp442`

To exit, you can simply close the window, or run

`conda deactivate comp442`

Note that every time you want to work on the assignment, you should run `conda activate comp442` (change to the name of your virtual env).

You may refer to [this page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.

#### Package Dependencies
If you have `pip` installed on your system (normally conda does it by default), you may use `pip` to install the
necessary python packages conveniently. From the project root, type the following:

`pip install -r requirements.txt`

This command will install the correct versions of the dependency packages listed in the requirements.txt file.


## Download data:

Make sure `wget` is installed on your machine before running the commands below. Run the following from the dataset directory:

```
cd datasets
./download_dataset.sh
```

## Grading
### Q1: 20
### Q2: 80

## For those who might run into troubles running PyTorch locally, we'd recommend working on the Jupyter notebook using Google Colab 
    
## Submission

Name the jupyter notebook with the format `username_studentid_assignment1.ipynb`.
Upload only the jupyter notebook to blackboard. Do not include large files in the submission (for
instance data files).!!!!

## Notes

NOTE 1: Make sure that your homework runs successfully. Otherwise, you may get a zero grade from the assignment.

NOTE 2: There are # *****START OF YOUR CODE***** and *****END OF YOUR CODE***** tags denoting the start and end of code sections you should fill out. Take care to not delete or modify these tags, or your assignment may not be properly graded.

NOTE 3: The assignment2 code has been tested to be compatible with python version 3.8 (it may work with other versions of 3.x, but we haven't tested them). You will need to make sure that during your virtual environment setup that the correct version of python is used. You can confirm your python version by (1) activating your environment and (2) running which python.
