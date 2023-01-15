# StokesFlow.jl

[![CI action](https://github.com/dastrm/StokesFlow.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/dastrm/StokesFlow.jl/actions/workflows/CI.yml)

Multi-xPU solver for the 2D Stokes & continuity equations, with variable density and viscosity, written in Julia. Material Properties are advected using Marker-in-Cell techniques.

## Introduction

The motion of viscous fluids can be modelled by solving the Navier–Stokes equations numerically. With a focus on the 2D Stokes equations with continuity assumption, a common simplification in computational fluid dynamics is explored in this project. For example, processes in the earth's mantle are commonly studied using similar assumptions, usually with more sophisticated coupling (e.g temperature) which is not implemented here. Additionally, we demonstrate the use of packages in Julia which allow reasonable scaling from one CPU core to potentially thousands of GPU nodes with ease.

To avoid an unreasonably dense grid and still ensure no (purely numerical) diffusion of material properties, the marker-in-cell technique was chosen over implicit advection on the grid.

Also, every major function is tested by an extensive test suite.

## Content

- [StokesFlow.jl](#stokesflowjl)
  - [Introduction](#introduction)
  - [Content](#content)
  - [Script list](#script-list)
  - [Usage](#usage)
  - [2D Stokes and continuity](#2d-stokes-and-continuity)
  - [Implementation](#implementation)
  - [Results](#results)
  - [Discussion](#discussion)
  - [TODO](#todo)

## Script list
TODO update

The [scripts](/scripts/) folder contains the following Julia scripts:
- [`StokesFlow2D_multigpu.jl`](scripts/StokesFlow2D_multigpu.jl), main script
- [`StokesSolver_multixpu.jl`](scripts/StokesSolver_multixpu.jl), standalone multi-xPU Stokes solver
- [`viz.jl`](scripts/viz.jl), visualization script

## Usage

We provide a [`Project.toml`](Project.toml) file which installs all requirements upon activation. Once all packages are installed, the main script can be launched from the command line as follows:

```sh
mpiexecjl -n 2 julia --project=. scripts/multi_process/StokesFlow2D_multixpu.jl
```

Note that the `-n` flag controls the amount of MPI processes to be spawned, which in our exemplary case is set to two. During execution, relevant output is written to a newly created folder `viz_out`. In order to visualize the output upon completion, the visualization script can be run as follows:

```sh
julia --project=. scripts/viz.jl
```

The resulting figures can be found in the folder `viz_out`.

Additionally, the script allows for live visualization during computation if desired.

## 2D Stokes and continuity

The **2D Stokes and continuity** equations, assuming earth's gravity in positive $y$-direction, are:
$$
\frac{\partial \tau_{xx}}{\partial x} + \frac{\partial \tau_{xy}}{\partial y} - \frac{\partial P}{\partial x} = 0
\newline
\frac{\partial \tau_{yx}}{\partial x} + \frac{\partial \tau_{yy}}{\partial y} - \frac{\partial P}{\partial y} = -\rho g y
\newline
\nabla \cdot V = 0,
$$
with
$$
\tau_{xx} = 2 \mu \frac{\partial V_x}{\partial x} 
\newline
\tau_{yy} = 2 \mu \frac{\partial V_y}{\partial y} 
\newline
\tau_{xy} = \mu \left(\frac{\partial V_x}{\partial y} + \frac{\partial V_y}{\partial x}  \right).
$$

The material properties (carried by markers) are
* $\mu$ : viscosity
* $\rho$ : density,

the other variables describe
* $V = [V_x, V_y]^T$ : velocity
* $P$ : pressure
* $g$ : earth gravity
* $\tau$ : deviatoric stress tensor.


## Implementation

### General structure and methods

TODO: general & grid (Simon)

### Details of Marker Methods

TODO: Continuity-based velocity interpolation (Simon)

### Details of Stokes Solver

TODO: Free surface stabilization, array sizes (Simon)

## Results

| ![markersref](figures/markersref.gif) |
| :-----------------------------------: |
|           Fig. 1: Markers.            |

| ![gridref](figures/gridref.gif) |
| :-----------------------------: |
|          Fig. 2: Grid.          |

<figure>
    <img src="figures/markersref.gif">
    <figcaption>Fig. 3: Markers.<figcaption>
<figure>


## Discussion

## Open Issues and Further Work

* The Stokes solver converges rather slow, caused by the big range of material properties which can span several orders of magnitude (e.g. density of *sticky air* vs. *mantle*). Some improvement can certainly be achieved with more parameter tuning, slightly smoothing the material property arrays, or even altogether implementing the iterations differently. However one must pay special attention to avoid blowups of the solution, which is a delicate balance.

* For the specific parameter combination of **both** `USE_GPU=true` **and** `--check-bounds=yes`, the results of the Stokes Solver are different, and thus the reference tests don't pass. This is very strange, especially since no bounds check ever fails.

* To model actually interesting phenomena, the physics could be extended in various ways. For example, coupling with the heat equation, different heating mechanisms and heat-dependent changes in material properties could be implemented.

## TODO

* do some performance evaluations: what is slowest etc.
* write README:
    * Equations, Methods,... with references
    * How to run the code
    * Figures, Discussion of results and performance
    * further work
