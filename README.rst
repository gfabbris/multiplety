multiplety
=========================

Multiplet calculation of XAS and RIXS spectra using the Cowan's and Racer codes. These codes are not distributed here but are essential, please contact me for more information.

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

3. Install gfortran.
    Download and install `gfortran <https://gcc.gnu.org/wiki/GFortranBinaries>`_.

3. Install package.

    Download and extract `multiplety package <https://github.com/gfabbris/multiplety>`_.

    Edit the atomic_calculation.py module to correct the folder for Cowan's and Racer codes. This module is located at *multiplety-master/multiplety*. The global parameter *install_folder* in line 50 needs to contain the correct folder for these atomic codes.

    Change directory and install pyrixs:
    ::
        cd multiplety-master
        python setup.py install
        conda install pandas

4. Launch analysis session.

    Open jupyter:
    ::
        jupyter notebook

    Navigate to *notebooks* and click *example_xas*, *example_rixs* or *example_rixs_dichroism*.
