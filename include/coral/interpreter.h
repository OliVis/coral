#ifndef INTERPRETER_H
#define INTERPRETER_H

#include <cstdint>
#include <memory>
#include <edgetpu.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Struct to hold quantization parameters
struct QuantizationParams {
    float scale;
    int zero_point;
};

class EdgeTPUInterpreter {
public:
    EdgeTPUInterpreter(const char* model_path);
    ~EdgeTPUInterpreter();

    // Invoke method to run inference
    TfLiteStatus invoke();

    // Pointers to input and output tensor data
    int8_t* input_data_ptr;
    int8_t* output_data_ptr;

    QuantizationParams input_quant;
    QuantizationParams output_quant;

private:
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
    std::unique_ptr<tflite::Interpreter> interpreter;
};

#endif // INTERPRETER_H
