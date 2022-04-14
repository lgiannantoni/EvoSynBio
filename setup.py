from setuptools import setup, find_namespace_packages, find_packages, PEP420PackageFinder

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coherence",
    version="0.0.0",
    author='Leonardo Giannantoni',
    author_email='leonardo.giannantoni@gmail.com',
    description='A useful module for blablabla',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    # packages=['coeherence'],  #same as name
    packages=PEP420PackageFinder.find(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        #"Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9'
)
