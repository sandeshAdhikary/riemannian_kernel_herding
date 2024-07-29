# Riemannian Herding


- Install the latest version of Anaconda for your system
- Create a new environment
```
conda create -n riemannian-herding python=3.7 
```
- Activate the environment whenever working on the project
```
conda activate riemannian-herding
```
- Install (or update) required packages into the environment
```
conda env update -n riemannian-herding -f environment.yml
```

- Install the project as a package (in editable state) using ``setup.py``. This step lets us
 do relative imports from sibling directories without messing with the 
 ``sys.path``. Since we will be installing the project in editable state (the ``-e`` flag), any changes
 we make to python files will automatically be updated in the installed package. 
 Check this [SO post](https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944) for more info. 
 From the root folder, run the following (Note that there is a
 ``.`` at the end here indicating the ``setup.py`` lives in the current dir)
```
pip install -e .
```