# CPP MNIST

This project implment a neural network that recognizes hand-written digits

### Build

The executable can be built using the following command:

```sh
cmake -B build -S .
cmake --build build -j$(nproc)
```

### Testing

Execute the following commands to test components of the project

> The project must be built before testing

```sh
cd build
ctest
```
