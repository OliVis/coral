#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <sstream>
#include <filesystem>
#include "coral/interpreter.h"
#include "coral/rtlsdr.h"
#include "coral/buffer.h"

const std::string MODEL_NAME = "FFT";
const size_t FFT_SIZE = 1024;
const size_t READ_SIZE = 16384;

// Callback function to handle received data from the RTL-SDR
void callback(uint8_t* buf, uint32_t len, void* ctx) {
    CircularBuffer* queue = reinterpret_cast<CircularBuffer*>(ctx);

    // Place sample in the queue
    queue->put(buf);
}

void process(EdgeTPUInterpreter& interpreter, CircularBuffer& queue) {
    const QuantizationParams input_quant = interpreter.input_tensor_info().quantization;
    int8_t* input_tensor = interpreter.input_tensor_info().data_ptr;

    // Precompute constants for float conversion
    const float scale = 1.0f / 128.0f;
    const float zero_offset = -127.4f;

    uint8_t buffer[READ_SIZE];
   
    while (true) {
        // Retrieve data from the queue
        queue.get(buffer);

        // Split up into samples
        for (size_t sample_index = 0; sample_index < READ_SIZE; sample_index += FFT_SIZE) {
            // Preprocess each value in the sample
            for (size_t value_index = 0; value_index < FFT_SIZE; ++value_index) {
                // Convert unsigned byte to float and apply quantization
                float value = (buffer[sample_index + value_index] + zero_offset) * scale;
                input_tensor[value_index] = static_cast<int8_t>((value / input_quant.scale) + input_quant.zero_point);
            }
            // Invoke the EdgeTPU interpreter for inference
            interpreter.invoke();
            std::cerr << "Completed." << std::endl;
        }
    }
}

int main() {
    const std::string model_path = MODEL_NAME + "_edgetpu.tflite";

    // Check if the EdgeTPU TensorFlow Lite model exists
    if (!std::filesystem::exists(model_path)) {
        std::stringstream command;

        // Create a new model
        command << "python3 minimum_model.py " << FFT_SIZE << " " << MODEL_NAME;
        if (std::system(command.str().c_str()) < 0) {
            throw std::runtime_error("Error: Failed to build model.");
        }
    }

    // Create a circular buffer to hold RTL-SDR data
    CircularBuffer queue(READ_SIZE, 1000);

    // Initialize EdgeTPUInterpreter with model path
    EdgeTPUInterpreter interpreter(model_path);

    // Start a thread to process data on the EdgeTPU
    std::thread tpu_thread([&interpreter, &queue]() {
        process(interpreter, queue);
    });

    // Configure RTL-SDR settings
    RtlSdr sdr;
    sdr.set_sample_rate(2'040'000);  // 2.04MHz
    sdr.set_center_freq(80'000'000); // 80Mhz
    sdr.set_gain(6.0);               // 6dB

    // Start a thread to read data asynchronously from RTL-SDR
    std::thread sdr_thread([&sdr, &queue]() {
        sdr.read_async(callback, &queue, READ_SIZE);
    });

    // Wait for threads to finish execution
    sdr_thread.join();
    tpu_thread.join();

    return 0;
}
