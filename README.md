# InvertibleNetworks.jl

| **Documentation** | **Build Status**  |  **JOSS paper**  |
|:-----------------:|:-----------------:|:----------------:|
|[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://slimgroup.github.io/InvertibleNetworks.jl/stable/) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://slimgroup.github.io/InvertibleNetworks.jl/dev/)| [![CI](https://github.com/slimgroup/InvertibleNetworks.jl/actions/workflows/runtests.yml/badge.svg)](https://github.com/slimgroup/InvertibleNetworks.jl/actions/workflows/runtests.yml)|  [![DOI](https://joss.theoj.org/papers/10.21105/joss.06554/status.svg)](https://doi.org/10.21105/joss.06554)

Building blocks for invertible neural networks in the [Julia] programming language.

## Overview

InvertibleNetworks.jl provides memory-efficient building blocks for invertible neural networks with hand-derived gradients, Jacobians, and log-determinants. The package is designed for high-performance scientific computing and machine learning applications.

### Key Features

- **Memory Efficient**: Hand-derived gradients, Jacobians J, and log|J| for optimal memory usage
- **Flux Integration**: Seamless integration with Flux.jl for automatic differentiation
- **AD Support**: Support for [Zygote] and [ChainRules] automatic differentiation
- **GPU Support**: Full GPU support via CuArray
- **Comprehensive Examples**: Various examples of invertible neural networks, normalizing flows, variational inference, and uncertainty quantification

## Installation

InvertibleNetworks is registered and can be added like any standard Julia package:

```julia
using Pkg
Pkg.add("InvertibleNetworks")
```

Or from the REPL:

```julia
] add InvertibleNetworks
```

## Quick Start

### Basic Usage

```julia
using InvertibleNetworks, Flux

# Create a simple activation normalization layer
an = ActNorm(10; logdet=true)

# Forward pass
X = randn(Float32, 64, 64, 10, 4)
Y, logdet = an.forward(X)

# Inverse pass
X_reconstructed = an.inverse(Y)

# Test invertibility
@assert norm(X - X_reconstructed) < 1e-6
```

### GPU Support

```julia
using InvertibleNetworks, Flux

# Move data to GPU
X = randn(Float32, 64, 64, 10, 4) |> gpu
AN = ActNorm(10; logdet=true) |> gpu

# Forward pass on GPU
Y, logdet = AN.forward(X)
```

## Building Blocks

### Core Layers

- **ActNorm**: Activation normalization (Kingma and Dhariwal, 2018)
- **Conv1x1**: 1x1 Convolutions using Householder transformations
- **ResidualBlock**: Invertible residual blocks
- **CouplingLayerGlow**: Invertible coupling layer from Dinh et al. (2017)
- **CouplingLayerHINT**: Invertible recursive coupling layer HINT from Kruse et al. (2020)
- **CouplingLayerHyperbolic**: Invertible hyperbolic layer from Lensink et al. (2019)
- **CouplingLayerIRIM**: Invertible coupling layer from Putzky and Welling (2019)

### Activation Functions

- **ReLU**: Rectified Linear Unit
- **LeakyReLU**: Leaky Rectified Linear Unit
- **Sigmoid**: Sigmoid activation with optional scaling
- **Sigmoid2**: Modified sigmoid activation
- **GaLU**: Gated Linear Unit
- **ExpClamp**: Exponential with clamping

### Utilities

- **Parameter Management**: Efficient parameter handling with gradients
- **Objective Functions**: Mean squared error, log-likelihood
- **Dimensionality Operations**: Squeeze/unsqueeze, split/cat
- **Jacobian Computation**: Hand-derived Jacobians for memory efficiency

## Network Architectures

### Pre-built Networks

- **NetworkGlow**: Generative flow with invertible 1x1 convolutions
- **NetworkHINT**: Multi-scale HINT networks
- **NetworkHyperbolic**: Hyperbolic networks
- **NetworkIRIM**: Invertible recurrent inference machines
- **NetworkConditionalGlow**: Conditional Glow networks
- **NetworkConditionalHINT**: Conditional HINT networks

### Example: Creating a Glow Network

```julia
using InvertibleNetworks, Flux

# Network parameters
n_in = 3      # Input channels
n_hidden = 64 # Hidden dimensions
L = 4         # Number of scales
K = 2         # Number of flow steps per scale

# Create Glow network
G = NetworkGlow(n_in, n_hidden, L, K)

# Forward pass
X = randn(Float32, 64, 64, n_in, 4)
Y, logdet = G.forward(X)

# Inverse pass
X_reconstructed = G.inverse(Y)
```

## Uncertainty-aware Image Reconstruction

InvertibleNetworks.jl has been particularly successful at Bayesian posterior sampling with simulation-based inference due to its memory scaling. 

### Example: MNIST Inpainting

```julia
# See examples/applications/conditional_sampling/amortized_glow_mnist_inpainting.jl
# for a complete example of conditional sampling for MNIST inpainting
```

![mnist_sampling_cond](docs/src/figures/mnist_sampling_cond.png)

## Examples

The package includes comprehensive examples organized by application:

### Applications
- **Conditional Sampling**: MNIST inpainting, banana distribution sampling
- **Non-conditional Sampling**: Banana distribution, seismic data
- **Denoising**: HINT-based denoising

### Benchmarks
- **Performance**: Memory usage comparisons
- **Differentiation**: ForwardDiff vs ManualDiff comparisons

### Layer Examples
- **Individual Layers**: Detailed examples for each layer type
- **Network Composition**: How to combine layers into networks

### Network Examples
- **Complete Networks**: End-to-end examples for each network type
- **Training**: Examples with Flux integration

## Documentation

- **API Documentation**: [Stable](https://slimgroup.github.io/InvertibleNetworks.jl/stable/) | [Development](https://slimgroup.github.io/InvertibleNetworks.jl/dev/)
- **Examples**: See the `examples/` directory for comprehensive usage examples
- **Tests**: The `test/` directory contains extensive unit tests

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```julia
using Pkg
Pkg.develop("InvertibleNetworks")
```

### Running Tests

```julia
using Pkg
Pkg.test("InvertibleNetworks")
```

## Citation

If you use InvertibleNetworks.jl in your research, please cite:

```bibtex
@article{Orozco2024, 
    doi = {10.21105/joss.06554}, 
    url = {https://doi.org/10.21105/joss.06554}, 
    year = {2024}, 
    publisher = {The Open Journal}, 
    volume = {9}, 
    number = {99}, 
    pages = {6554}, 
    author = {Rafael Orozco and Philipp Witte and Mathias Louboutin and Ali Siahkoohi and Gabrio Rizzuti and Bas Peters and Felix J. Herrmann}, 
    title = {InvertibleNetworks.jl: A Julia package for scalable normalizing flows}, 
    journal = {Journal of Open Source Software} 
}
```

## Related Publications

The following publications use InvertibleNetworks.jl:

- **["Reliable amortized variational inference with physics-based latent distribution correction"]**
    - Paper: [https://arxiv.org/abs/2207.11640](https://arxiv.org/abs/2207.11640)
    - Code: [ReliableAVI.jl]

- **["Learning by example: fast reliability-aware seismic imaging with normalizing flows"]**
    - Paper: [https://arxiv.org/abs/2104.06255](https://arxiv.org/abs/2104.06255)
    - Code: [ReliabilityAwareImaging.jl]

- **["Enabling uncertainty quantification for seismic data pre-processing using normalizing flows"]**
    - Paper: [https://slim.gatech.edu/Publications/Public/Conferences/SEG/2021/kumar2021SEGeuq/kumar2021SEGeuq.pdf]
    - Code: [WavefieldRecoveryUQ.jl]

- **["Preconditioned training of normalizing flows for variational inference in inverse problems"]**
    - Paper: [https://arxiv.org/abs/2101.03709](https://arxiv.org/abs/2101.03709)
    - Code: [FastApproximateInference.jl]

- **["Parameterizing uncertainty by deep invertible networks, an application to reservoir characterization"]**
    - Paper: [https://arxiv.org/abs/2004.07871](https://arxiv.org/abs/2004.07871)

## Authors

- **Rafael Orozco** - Georgia Institute of Technology [rorozco@gatech.edu]
- **Philipp Witte** - Georgia Institute of Technology (now Microsoft)
- **Gabrio Rizzuti** - Utrecht University
- **Mathias Louboutin** - Georgia Institute of Technology
- **Ali Siahkoohi** - Georgia Institute of Technology

## Acknowledgments

This package uses functions from:
- [NNlib.jl](https://github.com/FluxML/NNlib.jl)
- [Flux.jl](https://github.com/FluxML/Flux.jl)
- [Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[Flux]: https://fluxml.ai
[Julia]: https://julialang.org
[Zygote]: https://github.com/FluxML/Zygote.jl
[ChainRules]: https://github.com/JuliaDiff/ChainRules.jl
[InvertibleNetworks.jl]: https://github.com/slimgroup/InvertibleNetworks.jl
[ReliableAVI.jl]: https://github.com/slimgroup/ReliableAVI.jl
[ReliabilityAwareImaging.jl]: https://github.com/slimgroup/Software.SEG2021/tree/main/ReliabilityAwareImaging.jl
[WavefieldRecoveryUQ.jl]: https://github.com/slimgroup/Software.SEG2021/tree/main/WavefieldRecoveryUQ.jl
[FastApproximateInference.jl]: https://github.com/slimgroup/Software.siahkoohi2021AABIpto
