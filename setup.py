import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="dmosopt", 
    version="0.0.1",
    author="Ivan Raikov",
    author_email="ivan.g.raikov@gmail.com",
    description="Distributed controller for MO-ASMO multi-objective surrogate optimization algorithm.",
#    long_description=long_description,
#    long_description_content_type="text/markdown",
    url="https://github.com/iraikov/dmosopt",
    py_modules=["dmosopt", "GLP", "MOASMO", "NSGA2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=[
        'mpi4py', 'numpy', 'h5py', 'distwq>=0.0.3', 
    ],

)
