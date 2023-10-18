# Pip and pipenv
To upgrade pip:

'''
python -m pip install --upgrade pip
'''
   
To install pipenv:
'''
pip install pipenv
'''
   
To get version of pipenv:
'''
pipenv --version
'''
   
To install a specific version of a library:
'''
pipenv install scikit-learn==1.3.1
'''
Once done it will create an environment with the installed libraries in it.
In the directory where you executed this, you can find the Pipfile and pipfile.lock files

you can now create the virtual environment
'''
pipenv shell
'''


If you clone the code on a different computer you can just type 
'''
pipenv install 
'''
And you get the same environment
   
# Conda 
Create a conda enviroment:
'''
conda create -n mlzoomcamp python=3.8
'''
   
Activate the conda environment
'''
conda activate mlzoomcamp
'''
    
Install libraries available in conda
'''
conda install numpy pandas scikit-learn seaborn jupyter
'''
   
# Docker
docker pull an image:
'''
docker pull svizor/zoomcamp-model:3.10.12-slim
'''
   
docker list images:
'''
docker images
'''

docker inspect image:
'''
docker inspect  svizor/zoomcamp-model:3.10.12-slim
'''

docker list all containers:
'''
docker ps -a
'''
