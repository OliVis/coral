#include <iostream>
#include <cstdint>
#include <thread>
#include "coral/interpreter.h"
#include "coral/rtlsdr.h"
#include "coral/buffer.h"

#define READ_SIZE 16384

// Callback function to handle received data from the RTL-SDR
void callback(uint8_t* buf, uint32_t len, void* ctx) {
    CircularBuffer* queue = reinterpret_cast<CircularBuffer*>(ctx);

    // Place sample in the queue
    queue->put(buf);
}

void process(EdgeTPUInterpreter& interpreter, CircularBuffer& queue) {
    const size_t input_size = interpreter.input_tensor_info().size;
    const QuantizationParams input_quant = interpreter.input_tensor_info().quantization;
    int8_t* input_tensor = interpreter.input_tensor_info().data_ptr;

    // Precompute constants for quantization
    const float scale = 1.0f / 128.0f;
    const float zero_offset = -127.4f * scale;

    uint8_t buffer[READ_SIZE];
   
    while (true) {
        // Retrieve data from the queue
        queue.get(buffer);

        // Split up into samples
        for (size_t sample_index = 0; sample_index < READ_SIZE; sample_index += input_size) {
            // Preprocess each value in the sample
            for (size_t value_index = 0; value_index < input_size; ++value_index) {
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
    // Create a circular buffer to hold RTL-SDR data
    CircularBuffer queue(READ_SIZE, 10000);

    // Initialize EdgeTPUInterpreter with model path
    EdgeTPUInterpreter interpreter("../test_edgetpu.tflite");

    // Start a thread to process data on the EdgeTPU
    std::thread tpu_thread([&interpreter, &queue]() {
        process(interpreter, queue);
    });

    // Configure RTL-SDR settings
    RtlSdr sdr;
    sdr.set_sample_rate(2'040'000);  // 2.04MHz
    sdr.set_center_freq(80'000'000); // 80Mhz
    sdr.set_gain(60);                // 6dB

    // Start a thread to read data asynchronously from RTL-SDR
    std::thread sdr_thread([&sdr, &queue]() {
        sdr.read_async(callback, &queue, READ_SIZE);
    });

    // Wait for threads to finish execution
    sdr_thread.join();
    tpu_thread.join();

    return 0;
}
