#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <filesystem>
#include <readerwritercircularbuffer.h>
#include "coral/interpreter.h"
#include "coral/rtlsdr.h"

// Type alias for the circular buffer that holds the data
using CircularBuffer = moodycamel::BlockingReaderWriterCircularBuffer<std::vector<uint8_t>>;

const std::string MODEL_SUFFIX = "_edgetpu.tflite";

// Struct to hold program properties
struct ProgramProperties {
    size_t read_size;
    int fft_size;
    int samples;
    std::string model_script;
    std::string output_file;
    int output_samples;
};

// Callback function to handle received data from the RTL-SDR
void callback(uint8_t* buf, uint32_t len, void* ctx) {
    CircularBuffer* queue = reinterpret_cast<CircularBuffer*>(ctx);

    // Measure time between callbacks
    static auto last_callback = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_callback = current_time - last_callback;
    last_callback = current_time;
    std::cerr << std::fixed << "Time between callbacks: " << elapsed_callback.count() << " seconds" << std::endl;

    // Create a movable object to hold the buffer data
    std::vector<uint8_t> data(buf, buf + len);

    // Place the data in the queue
    if (!queue->try_enqueue(std::move(data)))
        std::cerr << "# Queue full, sample dropped!" << std::endl;
}

void process(EdgeTPUInterpreter& interpreter, CircularBuffer& queue, const ProgramProperties properties) {
    // Input tensor info
    const QuantizationParams input_quant = interpreter.input_tensor_info().quantization;
    int8_t* input_tensor = interpreter.input_tensor_info().data_ptr;

    // Output tensor info
    const QuantizationParams output_quant = interpreter.output_tensor_info().quantization;
    int8_t* output_tensor = interpreter.output_tensor_info().data_ptr;

    // Precompute constants for float conversion
    const float scale = 1.0f / 128.0f;
    const float zero_offset = -127.4f;

    // Buffers to hold the data
    std::vector<uint8_t> input_data(properties.read_size);
    float* output_data = new float[properties.read_size];

    std::ofstream ofs(properties.output_file, std::ios::binary);

    int samples_written = 0;
    while (properties.output_samples == -1 || samples_written < properties.output_samples) {
        auto start_batch = std::chrono::high_resolution_clock::now();

        // Retrieve data from the queue when available
        queue.wait_dequeue(input_data);

        // Ensure that the correct number of bytes are read
        if (input_data.size() < properties.read_size) {
            std::cerr << "# Lost bytes, got less data than expected." << std::endl;
            continue;
        }

        // Preprocess each value in the buffer
        for (size_t index = 0; index < properties.read_size; ++index) {
            // Convert unsigned byte to float and apply quantization
            float value = (input_data[index] + zero_offset) * scale;
            input_tensor[index] = static_cast<int8_t>((value / input_quant.scale) + input_quant.zero_point);
        }
        // Invoke the EdgeTPU interpreter for inference
        interpreter.invoke();

        // Process and store the output tensor values
        for (size_t index = 0; index < properties.read_size; ++index) {
            // Dequantize the output value
            output_data[index] = (output_tensor[index] - output_quant.zero_point) * output_quant.scale;
        }
        // Write the buffer to file
        ofs.write(reinterpret_cast<char*>(output_data), properties.read_size * sizeof(float));

        auto end_batch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_batch = end_batch - start_batch;
        std::cerr << std::fixed << "Time taken for batch: " << elapsed_batch.count() << " seconds" << std::endl;

        samples_written += properties.samples;
    }
    // Clean up
    ofs.close();
    delete[] output_data;
}

void usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " -f <fft_size> -s <samples> -o <output_file> [-n <num_output_samples>] [-m <model_script>]" << std::endl;
    std::cerr << " - fft_size: The size of the FFT, must be a power of 2." << std::endl;
    std::cerr << " - samples: The number of samples to batch process." << std::endl;
    std::cerr << " - output_file: The binary file to output the results to." << std::endl;
    std::cerr << " - output_samples: (optional) The number of samples in the output file. If not provided, the program will keep running indefinitely." << std::endl;
    std::cerr << " - model_script: (optional) The Python script used for model creation. Default is 'fft_model.py'." << std::endl;
    std::cerr << std::endl;
    std::cerr << " '2 * fft_size * samples' bytes are read from the USB per callback." << std::endl;
    std::cerr << " That number must be a multiple of 256 and is recommended to be a multiple of 16384 (USB urb size)." << std::endl;
}

int main(int argc, char* argv[]) {
    ProgramProperties properties;

    // Default values for optional arguments
    properties.model_script = "fft_model.py";
    properties.output_samples = -1;

    char opt;
    while ((opt = getopt(argc, argv, "f:s:o:m:n:")) != -1) {
        switch (opt) {
            case 'f':
                properties.fft_size = std::stoi(optarg);
                break;
            case 's':
                properties.samples = std::stoi(optarg);
                break;
            case 'o':
                properties.output_file = optarg;
                break;
            case 'n':
                properties.output_samples = std::stoi(optarg);
                break;
            case 'm':
                properties.model_script = optarg;
                break;
            default:
                usage(argv[0]);
                exit(-1);
        }
    }

    // Check if all required arguments are provided
    if (properties.fft_size == 0 || properties.samples == 0 || properties.output_file.empty()) {
        std::cerr << "Missing required arguments." << std::endl;
        usage(argv[0]);
        exit(-1);
    }

    // Calculate the USB read size
    properties.read_size = 2 * properties.fft_size * properties.samples;

    // Create a unique model name based on the FFT size and number of samples
    const std::string model_name = "fft_model_" + std::to_string(properties.fft_size) + "_" + std::to_string(properties.samples);

    // Check if the EdgeTPU TensorFlow Lite model exists
    if (!std::filesystem::exists(model_name + MODEL_SUFFIX)) {
        // Create command to execute Python script for model creation
        std::stringstream command;
        command << "python3 " << properties.model_script  // The script to create the model
                << " " << properties.fft_size             // Size of the FFT
                << " " << properties.samples              // Number of samples
                << " " << model_name;                     // Name for the model to be created
        if (std::system(command.str().c_str()) < 0)
            throw std::runtime_error("Error: Failed to build model.");
    }

    // Create a circular buffer to hold RTL-SDR data
    CircularBuffer queue(1000);

    // Initialize EdgeTPUInterpreter with model path
    EdgeTPUInterpreter interpreter(model_name + MODEL_SUFFIX);

    // Start a thread to process data on the EdgeTPU
    std::thread tpu_thread([&interpreter, &queue, &properties]() {
        process(interpreter, queue, properties);
    });

    // Configure RTL-SDR settings
    RtlSdr sdr;
    sdr.set_sample_rate(2'040'000);  // 2.04MHz
    sdr.set_center_freq(80'000'000); // 80Mhz
    sdr.set_gain(6.0);               // 6dB

    // Start a thread to read data asynchronously from RTL-SDR
    std::thread sdr_thread([&sdr, &queue, &properties]() {
        sdr.read_async(callback, &queue, properties.read_size);
    });

    // Wait for threads to finish execution
    tpu_thread.join();
    sdr.cancel_async();
    sdr_thread.join();

    return 0;
}
