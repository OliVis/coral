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

// Struct to hold the properties for the processing thread
struct ProcessProperties {
    size_t read_size;
    int iterations;
    std::string output_file;
    std::string samples_file;
};

// Callback function to handle received data from the RTL-SDR
void callback(uint8_t* buf, uint32_t len, void* ctx) {
    CircularBuffer* queue = reinterpret_cast<CircularBuffer*>(ctx);

    // Measure time between callbacks
    static auto last_callback = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_callback = current_time - last_callback;
    last_callback = current_time;
    std::cout << std::fixed << "< " << elapsed_callback.count() << std::endl;

    // Create a movable object to hold the buffer data
    std::vector<uint8_t> data(buf, buf + len);

    // Place the data in the queue
    if (!queue->try_enqueue(std::move(data))) {
        std::cout << "! Queue full, sample dropped" << std::endl;
    }
}

void process(EdgeTPUInterpreter& interpreter, CircularBuffer& queue, const ProcessProperties properties) {
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

    // Open output file streams
    std::ofstream ofs(properties.output_file, std::ios::binary);
    std::ofstream sfs;
    if (!properties.samples_file.empty()) {
        sfs.open(properties.samples_file, std::ios::binary);
    }

    int iter_count = 0;
    while (properties.iterations == -1 || iter_count < properties.iterations) {
        // Retrieve data from the queue when available
        queue.wait_dequeue(input_data);

        auto start_batch = std::chrono::high_resolution_clock::now();

        // Ensure that the correct number of bytes are read
        if (input_data.size() < properties.read_size) {
            std::cerr << "! Lost bytes, got less data than expected." << std::endl;
            continue;
        }

        // Write the raw SDR samples to the input file
        if (sfs.is_open()) {
            sfs.write(reinterpret_cast<const char*>(input_data.data()), input_data.size());
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
        // Write the processed samples buffer to the output file
        ofs.write(reinterpret_cast<const char*>(output_data), properties.read_size * sizeof(float));

        auto end_batch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_batch = end_batch - start_batch;
        std::cout << std::fixed << "> " << elapsed_batch.count() << std::endl;

        iter_count++;
    }
    // Clean up
    delete[] output_data;
    ofs.close();
    if (sfs.is_open()) {
        sfs.close();
    }
}

void usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " -s <fft_size> -b <batch_size> -o <output_file> [options]\n\n"
              << "Required arguments:\n"
              << "  -s <fft_size>       Size of the FFT (must be a power of 2).\n"
              << "  -b <batch_size>     Number of FFTs to compute per batch.*\n"
              << "  -o <output_file>    File to store the processed samples.\n\n"
              << "Optional arguments:\n"
              << "  -i <iterations>     Number of batches to process (default: run indefinitely).\n"
              << "  -r <sample_rate>    Sample rate of the SDR in Hz (default: 2,040,000Hz).\n"
              << "  -f <frequency>      Center frequency of the SDR in Hz (default: 80,000,000Hz).\n"
              << "  -g <gain>           Gain value of the SDR (default: automatic).\n"
              << "  -d <samples_file>   File to store the raw SDR samples.\n"
              << "  -m <model_script>   Python script used for model creation (default: 'fft_model.py').\n\n"
              << "(*) '2 * fft_size * batch_size' bytes are read from the SDR per callback.\n"
              << "    This must be a multiple of 512, and it's recommended to use multiples of 16,384 (USB URB size)."
              << std::endl;
}

int main(int argc, char* argv[]) {
    ProcessProperties properties;
    properties.iterations = -1; // Default to run indefinitely

    // Required arguments
    int fft_size = 0;   // Size of the FFT
    int batch_size = 0; // Number of FFTs to compute per batch

    // Default RTL-SDR settings
    int sample_rate = 2'040'000; // Default sample rate (2.04 MHz)
    int frequency = 80'000'000;  // Default center frequency (80 MHz)
    float gain = -1;             // Default gain (automatic if set to -1)

    // Optional arguments
    std::string model_script = "fft_model.py"; // Default script for the FFT model creation
    std::string samples_file;                  // Optional file to store the raw SDR samples

    char opt;
    while ((opt = getopt(argc, argv, "s:b:o:i:r:f:g:d:m:")) != -1) {
        switch (opt) {
            case 's':
                fft_size = std::stoi(optarg);
                break;
            case 'b':
                batch_size = std::stoi(optarg);
                break;
            case 'o':
                properties.output_file = optarg;
                break;
            case 'i':
                properties.iterations = std::stoi(optarg);
                break;
            case 'r':
                sample_rate = std::stoi(optarg);
                break;
            case 'f':
                frequency = std::stoi(optarg);
                break;
            case 'g':
                gain = std::stof(optarg);
                break;
            case 'd':
                properties.samples_file = optarg;
                break;
            case 'm':
                model_script = optarg;
                break;
            default:
                usage(argv[0]);
                exit(-1);
        }
    }

    // Check if all required arguments are provided
    if (fft_size == 0 || batch_size == 0 || properties.output_file.empty()) {
        std::cerr << "Missing required arguments." << std::endl;
        usage(argv[0]);
        exit(-1);
    }

    // Calculate the USB read size
    properties.read_size = 2 * fft_size * batch_size;

    // Create a unique model name based on the FFT size and the batch size
    const std::string model_name = "fft_model_" + std::to_string(fft_size) + "_" + std::to_string(batch_size);

    // Check if the EdgeTPU TensorFlow Lite model exists
    if (!std::filesystem::exists(model_name + MODEL_SUFFIX)) {
        // Create command to execute Python script for model creation
        std::stringstream command;
        command << "python3 " << model_script  // The script to create the model
                << " -s " << fft_size          // Size of the FFT
                << " -b " << batch_size        // Number of FFTs per batch
                << " -n " << model_name;       // Name for the model to be created
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
    sdr.set_sample_rate(sample_rate);
    sdr.set_center_freq(frequency);
    if (gain < 0) {
        sdr.set_auto_gain();
    } else {
        sdr.set_gain(gain);
    }

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
