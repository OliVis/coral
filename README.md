# Signal Processing with Coral Edge TPU 

This project uses the Coral Edge TPU and TensorFlow Lite to perform signal processing functions, specifically the Fast Fourier Transform (FFT), on the Coral Edge TPU device. By leveraging the power of the Edge TPU, we aim to accelerate these computations for real-time or high-performance signal processing tasks.

## Getting Started

### Prerequisites

Ensure that you have the following installed on your system:

- Python with TensorFlow 2.16.1
- CMake 3.18+
- gasket-dkms
- librtlsdr-dev

### Installation

To install this project, follow these steps:

1. Download and install the Edge TPU runtime library `libedgetpu1-std` and `libedgetpu-dev`. You can obtain them from [here](https://github.com/feranick/libedgetpu/releases/tag/16.0TF2.16.1-1) or you can build `libedgetpu` from [source](https://github.com/google-coral/libedgetpu). Make sure to choose the appropriate package version for your device. Note that Google's [packages](https://coral.ai/docs/m2/get-started/#2-install-the-pcie-driver-and-edge-tpu-runtime) are outdated and won't work with this project.

2. Clone the project repository:
```bash
git clone https://github.com/OliVis/coral.git
cd coral
```

3. Create a build directory and run CMake to configure the project. Make sure the TensorFlow version matches the installed Python TensorFlow version.
```bash
mkdir build
cd build
cmake ..
```

4. Build the executable using make:
```bash
make -j$(nproc)
```
Use the `-j$(nproc)` option to use multiple processing units and speed up the build process.

## Usage
***TODO***
