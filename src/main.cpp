#include <iostream>
#include <cstdint>
#include <limits>
#include "coral/interpreter.h"
#include "coral/rtlsdr.h"

// Convert Hz to MHz
#define MHz *1000000

/* Output from test.py
Input:
[[-0.2961429   0.72014636]
 [-0.11538965  0.49974927]
 [ 0.07229356 -0.27514696]
 [-0.5882218   0.5396765 ]
 [ 0.27872697  0.6879593 ]
 [-0.4475756  -0.3872029 ]
 [-0.07416397 -0.49681303]
 [-0.98385125 -0.53814745]]

NumPy:
[-2.15432464+0.7502211j   0.99124136-0.76388042j  0.09547181+1.17095787j
  0.63763191-0.20103192j  2.11575194+0.52207026j -1.69764896+0.53533946j
 -0.12656288+3.18917345j -2.23070378+0.55832105j]

Model:
[[-2.1543248   0.750221  ]
 [ 0.99124134 -0.76388043]
 [ 0.09547177  1.1709579 ]
 [ 0.6376319  -0.20103206]
 [ 2.115752    0.52207017]
 [-1.6976489   0.5353394 ]
 [-0.12656283  3.1891732 ]
 [-2.2307038   0.55832124]]
*/

int main() {
    // Initialize EdgeTPUInterpreter with model path
    EdgeTPUInterpreter interpreter("../test_edgetpu.tflite");

    const size_t input_size = interpreter.getInputTensorInfo().size;

    // Input data from test.py (example values)
    float input_data[input_size] = {
        -0.2961429,   0.72014636,
        -0.11538965,  0.49974927,
         0.07229356, -0.27514696,
        -0.5882218,   0.5396765,
         0.27872697,  0.6879593,
        -0.4475756,  -0.3872029,
        -0.07416397, -0.49681303,
        -0.98385125, -0.53814745
    };
    
    // Quantize the input data and copy to interpreter's input data buffer
    int8_t* input_ptr = interpreter.getInputTensorInfo().data_ptr;
    QuantizationParams input_quant = interpreter.getInputTensorInfo().quantization;
    std::cout << "Quantized data:" << std::endl;
    for (int i = 0; i < input_size; ++i) {
        float value = (input_data[i] / input_quant.scale) + input_quant.zero_point;

        // Check for int8 conversion clipping
        if (value < std::numeric_limits<int8_t>::min() || 
            value > std::numeric_limits<int8_t>::max()) {
            std::cerr << "Quantization clipping!" << std::endl;
        }

        // Convert and store the quantized value
        *input_ptr = static_cast<int8_t>(value);
        std::cout << static_cast<int>(*input_ptr) << " ";

        ++input_ptr; // Increment data pointer
    }
    std::cout << std::endl;

    // Invoke the interpreter for inference
    interpreter.invoke();

    // Dequantize and print each element of the output data
    int8_t* output_ptr = interpreter.getOutputTensorInfo().data_ptr;
    QuantizationParams output_quant = interpreter.getOutputTensorInfo().quantization;
    std::cout << "Output data:" << std::endl;
    for (int i = 0; i < input_size; ++i) {
        float value = (static_cast<float>(*output_ptr) - output_quant.zero_point) * output_quant.scale;

        // Print the dequantized output value
        std::cout << value << " ";

        ++output_ptr; // Increment data pointer
    }
    std::cout << std::endl;

    // Test the RTL-SDR class
    RtlSdr sdr;
    sdr.set_sample_rate(2.04 MHz);
    sdr.set_center_freq(80 MHz);
    sdr.set_gain(60);

    return 0;
}
