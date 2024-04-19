#include <iostream>
#include <cstdint>
#include <limits>
#include "coral/interpreter.h"

#define FFT_SIZE 8
#define INPUT_SIZE 2 * FFT_SIZE

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

    // Input data from test.py (example values)
    float input_data[INPUT_SIZE] = {
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
    int8_t* input_ptr = interpreter.input_data_ptr;
    std::cout << "Quantized data:" << std::endl;
    for (int i = 0; i < INPUT_SIZE; ++i) {
        float value = (input_data[i] / interpreter.input_quant.scale) +
                        interpreter.input_quant.zero_point;

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
    int8_t* output_ptr = interpreter.output_data_ptr;
    std::cout << "Output data:" << std::endl;
    for (int i = 0; i < INPUT_SIZE; ++i) {
        float value = (static_cast<float>(*output_ptr) - interpreter.output_quant.zero_point) *
                        interpreter.output_quant.scale;

        // Print the dequantized output value
        std::cout << value << " ";

        ++output_ptr; // Increment data pointer
    }
    std::cout << std::endl;

    return 0;
}
