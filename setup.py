
from setuptools import setup, find_packages

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setup(
    name="dmosopt", 
    version="0.11.0",
    author="Ivan Raikov",
    author_email="ivan.g.raikov@gmail.com",
    description="Distributed controller for MO-ASMO multi-objective surrogate optimization algorithm.",
#    long_description=long_description,
#    long_description_content_type="text/markdown",
    url="https://github.com/iraikov/dmosopt",
    packages=['dmosopt'],
    entry_points={
        'console_scripts': [ 'dmosopt_analyze=dmosopt.dmosopt_analyze:main',
                             'dmosopt_onestep=dmosopt.dmosopt_onestep:main',
                             'dmosopt_train=dmosopt.dmosopt_train:main',
                             'dmosopt_plot=dmosopt.dmosopt_plot:main',
                             ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=[
        'mpi4py', 'numpy', 'h5py', 'scipy', 'scikit-learn', 'distwq>=0.0.8', 
    ],
    extras_require = {
        "gpflow": ["gpflow"],
    }

)
