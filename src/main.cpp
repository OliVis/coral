#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <vector>
#include <readerwritercircularbuffer.h>
#include "coral/interpreter.h"
#include "coral/rtlsdr.h"

// Type alias for the circular buffer that holds the data
using CircularBuffer = moodycamel::BlockingReaderWriterCircularBuffer<std::vector<uint8_t>>;

const std::string MODEL_SUFFIX = "_edgetpu.tflite";

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

void process(EdgeTPUInterpreter& interpreter, CircularBuffer& queue, const size_t read_size) {
    const QuantizationParams input_quant = interpreter.input_tensor_info().quantization;
    int8_t* input_tensor = interpreter.input_tensor_info().data_ptr;

    // Precompute constants for float conversion
    const float scale = 1.0f / 128.0f;
    const float zero_offset = -127.4f;

    std::vector<uint8_t> data;
   
    while (true) {
        auto start_batch = std::chrono::high_resolution_clock::now();

        // Retrieve data from the queue when available
        queue.wait_dequeue(data);

        // Ensure that the correct number of bytes are read
        if (data.size() < read_size) {
            std::cerr << "# Lost bytes, got less data than expected." << std::endl;
            continue;
        }

        // Preprocess each value in the buffer
        for (size_t index = 0; index < read_size; ++index) {
            // Convert unsigned byte to float and apply quantization
            float value = (data[index] + zero_offset) * scale;
            input_tensor[index] = static_cast<int8_t>((value / input_quant.scale) + input_quant.zero_point);
        }
        // Invoke the EdgeTPU interpreter for inference
        interpreter.invoke();

        auto end_batch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_batch = end_batch - start_batch;
        std::cerr << std::fixed << "Time taken for batch: " << elapsed_batch.count() << " seconds" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default model script to use
    std::string model_script = "fft_model.py";

    // Check if the correct number of arguments is provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <fft_size> <samples> [model_script]" << std::endl;
        std::cerr << " - fft_size: The size of the FFT, must be a power of 2." << std::endl;
        std::cerr << " - samples: The number of samples to batch process." << std::endl;
        std::cerr << " - model_script: (optional) The Python script used for model creation. Default is 'fft_model.py'." << std::endl;
        std::cerr << std::endl;
        std::cerr << " '2 * fft_size * samples' bytes are read from the USB per callback." << std::endl;
        std::cerr << " That number must be a multiple of 256 and is recommended to be a multiple of 16384 (USB urb size)." << std::endl;
        return 1;
    }

    // Use different model script if provided
    if (argc >= 4)
        model_script = argv[3];

    // Parse the command line arguments
    const int fft_size = std::stoi(argv[1]);
    const int samples = std::stoi(argv[2]);

    // Calculate the USB read size
    const size_t read_size = 2 * fft_size * samples;

    // Create a unique model name based on the FFT size and number of samples
    const std::string model_name = "fft_model_" + std::to_string(fft_size) + "_" + std::to_string(samples);

    // Check if the EdgeTPU TensorFlow Lite model exists
    if (!std::filesystem::exists(model_name + MODEL_SUFFIX)) {
        // Create command to execute Python script for model creation
        std::stringstream command;
        command << "python3 " << model_script  // The script to create the model
                << " " << fft_size             // Size of the FFT
                << " " << samples              // Number of samples
                << " " << model_name;          // Name for the model to be created
        if (std::system(command.str().c_str()) < 0) {
            throw std::runtime_error("Error: Failed to build model.");
        }
    }

    // Create a circular buffer to hold RTL-SDR data
    CircularBuffer queue(1000);

    // Initialize EdgeTPUInterpreter with model path
    EdgeTPUInterpreter interpreter(model_name + MODEL_SUFFIX);

    // Start a thread to process data on the EdgeTPU
    std::thread tpu_thread([&interpreter, &queue, read_size]() {
        process(interpreter, queue, read_size);
    });

    // Configure RTL-SDR settings
    RtlSdr sdr;
    sdr.set_sample_rate(2'040'000);  // 2.04MHz
    sdr.set_center_freq(80'000'000); // 80Mhz
    sdr.set_gain(6.0);               // 6dB

    // Start a thread to read data asynchronously from RTL-SDR
    std::thread sdr_thread([&sdr, &queue, read_size]() {
        sdr.read_async(callback, &queue, read_size);
    });

    // Wait for threads to finish execution
    sdr_thread.join();
    tpu_thread.join();

    return 0;
}
