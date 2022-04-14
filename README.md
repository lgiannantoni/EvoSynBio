Coherence - co-simulation-based optimization of biofabrication protocols
========================================================================

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Release Date: March 2022](https://img.shields.io/badge/release%20date-march%202022-orange.svg)
![Status: Actrive](https://img.shields.io/badge/status-active-brightgreen.svg)
![Language: python](https://img.shields.io/badge/language-python-blue.svg)

Coherence would like to be a distributed multiscale framework for ontogenesis simulation and generation of optimized biofabrication protocols.

## Table of Contents

* [Overview](#overview)
  * [Methodological information](#methodological-information)
* [Coherence container](#coherence-container)
  * [Setup](#setup)
  * [Usage](#usage)
    * [Exploring the container](#exploring-the-container) 
    * [Shutting down the container](#shutting-down-the-container)
    * [Running an experiment](#running-an-experiment)
* [Coherence from source](#coherence-from-source)
  * [Setup](#setup-1)
  * [Usage](#usage-1)
    * [Running an experiment](#running-an-experiment-1)
    * [Batch mode](#batch-mode)
* [Author](#author)
* [Publications](#publications)
* [Recommended citation for this project](#recommended-citation-for-this-project)
* [Copyright](#copyright)

## Overview
Biofabrication processes are complex and often unsatisfactory. Trial-and-error methods are costly and yield only incremental innovation, starting from sub-optimal and poorly represented existing processes. Although computational techniques might support efficient process design to find optimal process configurations, intelligent computational approaches must comprise biological complexity to provide meaningful insights. Coherence is a novel prototypal co-simulation-based optimization methodology for the systematic design of protocols for cell culture and biofabrication. The proposed strategy integrates evolutionary computation and simulation for efficient design space exploration and assessment of candidate protocols. A generic library supports the modular and flexible composition of multiscale and multidomain co-simulation scenarios. The feasibility of the presented approach was demonstrated in the automatic generation of protocols for the biofabrication of an epithelial cell monolayer (see [Publications](#publications)).

### Methodological information
Coherence includes a Design Space Exploration (DSE) engine for the generation of biofabrication protocols and a simulation engine for testing them. The framework receives the high-level specification of the target product and iteratively computes a biofabrication protocol optimized to grow it. The DSE assembles potential biofabrication protocols and feeds them to simulation instances. Simulation results are compared against the specifications of the target product used to rank the corresponding protocols and generate new ones at the next iteration. This procedure continues until an optimal protocol is produced, a predetermined number of iterations is reached, or the protocol performance stalls for a given number of iterations.

Additional details and experimental results on a selected use-case can be found in [our publications](#publications).

## Coherence container
The following instructions help you build a Singularity container for running coherence.

### Setup
1. If Singularity is not installed in your system, open a shell and run
   ~~~~ bash
   wget https://raw.githubusercontent.com/smilies-polito/coherence/master/singularity_install.sh && \
   sudo chmod +x singularity_install.sh && \
   sudo ./singularity_install.sh
   ~~~~

2. Then get the container definition and build it.
   > This step is going to take a few minutes to complete.
   ~~~~ bash
   wget https://raw.githubusercontent.com/smilies-polito/coherence/master/coherence_container.def && \
   singularity build --fakeroot coherence_container.sif coherence_container.def
   ~~~~

   You will obtain a Singularity container (`coherence_container.sif`) with all the libraries and the code to run experiments with Coherence.        


3. Create a workspace folder for the experiments. This folder will be bound to the `experiments` folder in the container, so that results will be written locally.
   ~~~~ bash
   mkdir ~/workspace
   ~~~~
   
   You can [download the provided examples](https://raw.githubusercontent.com/smilies-polito/coherence/master/experiments.zip) and unzip them into your `workspace` folder. Alternatively, you can set up your own experiments based on the [README](experiments/README).

### Usage

   #### Starting up a container instance
   ~~~~ bash       
   singularity instance start --bind ~/workspace/:/packages/coherence/experiments coherence_container.sif coherence
   ~~~~
   > A container instance named `coherence` is now running in background.

   #### Exploring the container
   You can open a shell into the container with
   ~~~~ bash
   singularity shell instance://coherence
   ~~~~

   A few predefined environment variables are available to the user: 
* `$COHERENCE` points to coherence install directory `packages/coherence`
* `$EXPERIMENTS` expands to `packages/coherence/experiments`, where your local `~/workspace` directory is mounted.

You can manually run your experiments from inside the container following [the same instructions provided for the container-less version](#running-an-experiment-1).
   
   #### Shutting down the container
   To stop the instance use
   ~~~~ bash
   singularity instance stop coherence
   ~~~~

   #### Running an experiment

   You can automagically run an experiment without dealing with `coherence` code 
   ~~~~ bash
   singularity run instance://coherence half
   ~~~~

   This command executes the `singularity` script in the `coherence_container.sif` image root, which is a dump of the code defined in the `%runscript` section of the `coherence_container.def` file.

   > Substitute `half` in the command above with any other `${experiment name}` in your `~/workspace`.
   
   > Instructions on how to define new experiments are available in the `experiments/README` file.
 
   The results will be collected locally, in the `~/workspace/${experiment name}/results` folder.

   > If you are running the same experiment again, rename the `results` folder first.
       

## Coherence from source

The following sections guide you through the installation of coherence and all its dependencies directly on your system (i.e. the execution is not confined into a container).
   
   ### Setup
   1. Install system dependencies 
      
      If your system is Ubuntu >= 20.04 LTS, run:
      ~~~~ bash
      sudo add-apt-repository ppa:rock-core/qt4 && \
      sudo apt-get update && \
      sudo apt-get -y install build-essential apt-utils cmake wget bzip2 git curl binutils qt4-default unzip python3-venv python3-virtualenv screen
      ~~~~
      
      Otherwise, run: 
      ~~~~ bash
      sudo apt-get -y install build-essential apt-utils cmake wget bzip2 git curl binutils qt4-default unzip python3-venv python3-virtualenv screen
      ~~~~
      
      
   2. Install python 3.9.5
      ~~~~ bash
      wget https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz && \
      tar -xf Python-3.9.5.tgz && \
      cd Python-3.9.5 && \
      ./configure --enable-optimizations && \
      make -j 8 && \
      make altinstall && \
      cd .. && \
      rm -rf Python*
      ~~~~

   3. Get coherence
      ~~~~ bash
      git clone https://github.com/smilies-polito/coherence.git && \
      virtualenv -p /usr/bin/python3.9 coherence/venv && \
      source coherence/venv/bin/activate && \
      python -m pip install -r coherence/requirements.txt
      ~~~~
          
   4. Install custom libraries
      > N.B. First check the virtualenv created above is active.
      
      **[CoSimo](https://github.com/lgiannantoni/CoSimo)**   
        ~~~~ bash
        git clone https://github.com/leonardogian/CoSimo.git && \
        python -m pip install -e CoSimo
        ~~~~

      **[PyBoolNet](https://github.com/hklarner/pyboolnet)**
        ~~~~ bash
        git clone -b 2.31.0 --depth 1 https://github.com/hklarner/pyboolnet.git && \
        python -m pip install -e PyBoolNet
        ~~~~

      **[microgp3](https://github.com/squillero/microgp3)**
        ~~~~ bash
        git clone https://github.com/squillero/microgp3.git && \
        cd microgp3/src && \
        sed -i '34 a #include <functional>' ./Libs/EvolutionaryCore/Evaluator.h && \
        sed -i '40 a #include <functional>' ./Libs/EvolutionaryCore/CandidateSelection.h && \
        cmake -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_INSTALL_PREFIX:PATH=$HOME . && \
        make && make install && \
        mv $HOME/bin/ugp3 /opt && \
        mv $HOME/bin/ugp3-extractor /opt && \
        cd .. && \
        rm -rf microgp3
        ~~~~
          
      > N.B. Make sure `/opt` is in your `$PATH`! 
    
   ### Usage
   #### Running an experiment

   From inside the `coherence` folder, run
   ~~~~ bash
   ./coherence.sh experiments/half
   ~~~~

   Substitute `half` in the command above with any other `${experiment name}` in your `coherence/experiments` folder.
   
   The results will be stored in the `coherence/experiments/${experiment name}/results` folder.

   > If you are running the same experiment again, rename the `results` folder first.

   #### Batch mode
   From inside the `coherence/experiments` folder, run
   ~~~~ bash
   ./run_batch.sh
   ~~~~

   > Edit the `experiment` variable in `./run_batch.sh` with the list of experiments (i.e. folder names in `coherence/experiments`) to be run.


   > Each experiment will be run in a dedicated detached screen named after ${experiment name}.
   >  
   > Each ${experiment name} results will be collected in their corresponding `coherence/experiments/${experiment name}/results` folder.



## Author
Leonardo Giannantoni

Politecnico di Torino - Control and Computer Engineering Department <br/>
PhD Candidate @ SMILIES group

<img src="https://www.smilies.polito.it/email_signature/smilies.png" alt="smilies research group logo"/>
   
**E**:    leonardo.giannantoni@polito.it <br/>
**P**:    +39 011 090 7191 <br/>
**M**:    +39 377 283 4499 <br/>
**A**:    Corso Duca degli Abruzzi 24 10129 Torino Italy <br/>
**W**:    https://www.smilies.polito.it     <br/>       

## Publications
1. Leonardo Giannantoni, Roberta Bardini, and Stefano Di Carlo. "[A methodology for co-simulation-based optimization of biofabrication protocols](https://doi.org/10.1101/2022.01.28.478198)". IWBBIO 2022.

The experiment presented in this paper is available in the [*half*](experimental_results/half) folder, together with the results.

## Recommended citation for this project

**APA**
   
      Giannantoni, L. (2022). Coherence (Version 1.0.0) [Computer software]. https://github.com/smilies-polito/coherence

**BibTeX**

      @software{Giannantoni_Coherence_2022,
      author = {Giannantoni, Leonardo},
      month = {4},
      title = {{coherence}},
      url = {https://github.com/smilies-polito/coherence},
      version = {1.0.0},
      year = {2022}
      }

## Copyright

Coherence is licensed under the MIT license. A copy of this license is included in the file LICENSE.

