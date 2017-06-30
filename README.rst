multiplety
=========================

Multiplet calculation of XAS and RIXS spectra

Installation Instructions
=========================


1. Create a python environment to work in (optional).

    Download and install `anaconda <https://www.continuum.io/downloads>`_.

    Create a conda environment:
    ::
        conda create --name <name_of_enviroment>
    where <name_of_enviroment> is what you want to call the environment. N.B. python 3 is required, which should be the default, but can be explicitly requested by appending ``python=3`` to the ``conda create`` command above.


    Activate the environment:
    ::
        source activate <name_of_enviroment>

2. Install package.

    Download and extract `multiplety package <https://github.com/gfabbris/multiplety>`_.

    Change directory and install pyrixs and additional lmfit package:
    ::
        cd multiplety-master
        python setup.py install
        conda install pandas

3. Launch analysis session.

    Open jupyter:
    ::
        jupyter notebook

    Navigate to *notebooks* and click *example_xas*, *example_rixs* or *example_rixs_dichroism*.
