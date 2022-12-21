# StokesFlow.jl

[![CI action](https://github.com/dastrm/StokesFlow.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/dastrm/StokesFlow.jl/actions/workflows/CI.yml)

Multi-xPU solver for the 2D Stokes equations with continuity written in Julia.

## Introduction

The motion of viscous fluids can be modelled by solving the Navier–Stokes equations numerically. With a focus on the 2D Stokes equations with continuity assumption, a common simplification in computational fluid dynamics is explored in this project. Additionally, we demonstrate the use of packages in Julia which allow reasonable scaling from one CPU core to potentially thousands of GPU nodes with ease.

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

The [scripts](/scripts/) folder contains the following Julia scripts:
- [`StokesFlow2D_multigpu.jl`](scripts/StokesFlow2D_multigpu.jl), main script
- [`StokesSolver_multixpu.jl`](scripts/StokesSolver_multixpu.jl), standalone multi-xPU Stokes solver
- [`viz.jl`](scripts/viz.jl), visualization script

## Usage

We provide a [`Project.toml`](Project.toml) file which installs all requirements upon activation. Once all packages are installed, the main script can be launched from the command line as follows:

```sh
mpiexecjl -n 2 julia --project=. scripts/StokesFlow2D_multigpu.jl
```

Note that the `-n` flag controls the amount of MPI processes to be spawned, which in our exemplary case is set to two. During execution, relevant output is written to a newly created folder `viz_out`. In order to visualize the output upon completion, the visualization script can be run as follows:

```sh
julia --project=. scripts/viz.jl
```

The resulting figures can be found in the folder `viz_out`.

## 2D Stokes and continuity

## Implementation

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

Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes Navier–Stokes 

## Discussion

## TODO

* do some performance evaluations: what is slowest etc.
* write README:
    * Intro
    * Equations, Methods,... with references
    * How to run the code
    * Figures, Discussion of results and performance
    * further work
* write tests
* CI & CI-badge
* polish scripts: docstrings, comments,...
* optional: multi-xPU with MPI
