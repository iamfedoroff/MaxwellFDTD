# MaxwellFDTD


## Overview

Numerical solver for Maxwell’s equations using the Finite-Difference Time-Domain (FDTD)
method.
It targets time-domain electromagnetic simulations with support for linear, dispersive, and
nonlinear materials on CPU and GPU architectures.


## Features

- 1D, 2D, 3D Cartesian geometries
- Perfectly Matched Layers (PML)
- Soft, hard, and TFSF sources
- Linear and dispersive material models
- Kerr and plasma nonlinearities
- CPU execution in single-thread or multi-thread mode
- GPU acceleration


## Installation

Add the repository to your Julia environment using a direct GitHub link:
```julia
] add https://github.com/iamfedoroff/MaxwellFDTD
```


## Usage

See the `examples/` directory for representative simulation setups and usage patterns.

For lightweight visualization, it is recommended to install the
[MaxwellPlots](https://github.com/iamfedoroff/MaxwellPlots) package.
