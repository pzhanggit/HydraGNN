# GCNN

# Packages
Installing required packages:
(For running on machine with only cpu)

'pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html'
'pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html'
'pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html'
'pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html'
'pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.6.0.html'
'pip install torch-geometric'


(For serialization and storing the processed data for later sessions)

'pip install pickle5'

(For proper formatting of code)

'pip install git+git://github.com/psf/black'


# Code formatting
Before commiting and pushing changes, run command black . from the main directory of the project(GCNN).
