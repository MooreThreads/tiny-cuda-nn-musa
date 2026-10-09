# tiny-cuda-nn MUSA integration

The port adds MUSA headers, kernels, and build/runtime compilation integration
for the encoding and neural-network paths while retaining the upstream CUDA
implementation and bindings. Select the MUSA compiler, SDK, and architecture
through the provided CMake configuration; toolkits are external prerequisites.
