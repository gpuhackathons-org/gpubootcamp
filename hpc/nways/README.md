# Nways to GPU programming
This repository contains mini applications for GPU Bootcamps (**Tested on NVIDIA driver 440.82**). This labs comprises Nways to GPU programming implemented with the following programning approaches:

**C programming language**
  - OpenACC
  - Kokkos
  - PSTL
  - OpenMP
  - CUDA C
  
  
  
**Fortran programming language**
  - do-concurrent
  - OpenACC
  - OpenMP
  - CUDA Fortran
  
  
**Python programming language**
  - CuPy
  - Numba

We showcase above ways using mini applications in MD domain and CFD.

## Prerequisites:
To run this tutorial you will need a machine with NVIDIA GPU.

- Install the [Docker](https://docs.docker.com/get-docker/) or [Singularity](https://sylabs.io/docs/]).
- Install Nvidia toolkit, [Nsight Systems (latest version)](https://developer.nvidia.com/nsight-systems) and [compute (latest version)](https://developer.nvidia.com/nsight-compute).

## Creating containers
To start with, you will have to build a Docker or Singularity container.

### Docker Container
To build a docker container for **C & Fortran**, run:

`sudo docker build -t <imagename>:<tagnumber> .`

For instance :


`sudo docker build -t myimage:1.0 .`

While in the case of **Python** you have to specify the dockerfile name using flag **"-f"**, therefore run:

`sudo docker build -f <dockerfile name> -t <imagename>:<tagnumber> .`

For example :

`sudo docker build -f Dockerfile_python -t myimage:1.0 .`


For C, Fortran, and Python, the code labs have been written using Jupyter notebooks and a Dockerfile has been built to simplify deployment. In order to serve the docker instance for a student, it is necessary to expose port 8000 from the container, for instance, the following command would expose port 8000 inside the container as port 8000 on the lab machine:

`sudo docker run --rm -it --gpus=all -p 8888:8888 myimage:1.0`

When this command is run, you can browse to the serving machine on port 8888 using any web browser to access the labs. For instance, from if they are running on the local machine the web browser should be pointed to http://localhost:8888. The `--gpus` flag is used to enable `all` NVIDIA GPUs during container runtime. The `--rm` flag is used to clean an temporary images created during the running of the container. The `-it` flag enables killing the jupyter server with `ctrl-c`. This command may be customized for your hosting environment.


Then, inside the container launch the Jupyter notebook assigning the port you opened:

`jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root`


Once inside the container, open the jupyter notebook in browser: http://localhost:8888, and start the lab by clicking on the `START_nways.ipynb` notebook.

### Singularity Container

To build the singularity container for **C & Fortran**, run: 

`singularity build nways.simg Singularity`

While in the case of **Python**, run:

`singularity build nways.simg Singularity_python`

Thereafter, for C, Fortran, and Python, copy the files to your local machine to make sure changes are stored locally:

`singularity run nways.simg cp -rT /labs ~/labs`

Then, run the container:

`singularity run --nv nways.simg jupyter notebook --notebook-dir=~/labs`

Once inside the container, open the jupyter notebook in browser: http://localhost:8888, and start the lab by clicking on the `nways_start.ipynb` notebook.


## Questions?
Please join [OpenACC Slack Channel](https://openacclang.slack.com/messages/openaccusergroup) for questions.
