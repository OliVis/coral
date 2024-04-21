#ifndef INTERPRETER_H
#define INTERPRETER_H

#include <cstdint>
#include <vector>
#include <memory>
#include <edgetpu.h>
#include "tensorflow/lite/interpreter.h"

struct QuantizationParams {
    float scale;
    int zero_point;
};

struct TensorInfo {
    size_t size;
    std::vector<int> shape;
    int8_t* data_ptr;
    QuantizationParams quantization;
};

class EdgeTPUInterpreter {
public:
    EdgeTPUInterpreter(const char* model_path);
    ~EdgeTPUInterpreter();

    // Run inference on the loaded model
    TfLiteStatus invoke();

    // Getters for input and output tensor information
    const TensorInfo& getInputTensorInfo() const;
    const TensorInfo& getOutputTensorInfo() const;

private:
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
    std::unique_ptr<tflite::Interpreter> interpreter;

    TensorInfo input_tensor_info;
    TensorInfo output_tensor_info;

    void fillTensorInfo(const TfLiteTensor* tensor, TensorInfo& tensor_info);
};

#endif // INTERPRETER_H
