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

- Install the project as a package (in editable state) using ``setup.py``. Note the ``.`` at the end here indicating the ``setup.py`` lives in the current dir)
```
pip install -e .
```
