import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme:
    long_description = readme.read()

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='staNMF',
    version='1.0',
    packages=['staNMF'],
    include_package_data=True,
    license='',
    description='python 3 implementation of stability NMF (Siqi Wu 2016)',
    long_description=long_description,
    url='https://github.com/Yu-Group/staNMF',
    author='',
    author_email='',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'sklearn',
        'spams',
        'torch',
    ],
)
