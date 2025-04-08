#ifndef INTERPRETER_H
#define INTERPRETER_H

#include <string>
#include <cstdint>
#include <vector>
#include <memory>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

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

/**
 * @brief A wrapper class for running TensorFlow Lite models on an Edge TPU device.
 */
class EdgeTPUInterpreter {
public:
    /**
     * @brief Constructs an EdgeTPUInterpreter object with the specified model path.
     * @param model_path The path to the TensorFlow Lite model file optimized for the Edge TPU.
     */
    EdgeTPUInterpreter(const std::string model_path);

    /**
     * @brief Destructs the EdgeTPUInterpreter object, releasing associated resources.
     */
    ~EdgeTPUInterpreter();

    /**
     * @brief Invokes the interpreter to run the inference graph.
     * @return Status of success or failure.
     */
    TfLiteStatus invoke();

    /**
     * @brief Retrieves information about the input tensor.
     * @return A reference to the TensorInfo structure containing input tensor information.
     */
    const TensorInfo& input_tensor_info() const;

    /**
     * @brief Retrieves information about the output tensor.
     * @return A reference to the TensorInfo structure containing output tensor information.
     */
    const TensorInfo& output_tensor_info() const;

private:
    TfLiteDelegate* gpu_delegate;
    std::unique_ptr<tflite::Interpreter> interpreter;

    TensorInfo input_tensor_info_;
    TensorInfo output_tensor_info_;

    void fillTensorInfo(const TfLiteTensor* tensor, TensorInfo& tensor_info);
};

#endif // INTERPRETER_H
