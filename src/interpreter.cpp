#include <iostream>
#include <stdexcept>
#include "coral/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

EdgeTPUInterpreter::EdgeTPUInterpreter(const char* model_path) {
    // Load the compiled Edge TPU model as a FlatBufferModel
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path);

    // Create the EdgeTpuContext object
    edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    // Build the interpreter with Edge TPU custom op resolver
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    // Bind the context with the interpreter
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
    interpreter->SetNumThreads(1);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Error: Failed to allocate tensors.");
    }

    // Initialize input and output tensor information
    fillTensorInfo(interpreter->input_tensor(0), input_tensor_info_);
    fillTensorInfo(interpreter->output_tensor(0), output_tensor_info_);
}

EdgeTPUInterpreter::~EdgeTPUInterpreter() {
    // Releases interpreter instance to free up resources associated with this custom op
    interpreter.reset();

    // Closes the Edge TPU
    edgetpu_context.reset();
}

TfLiteStatus EdgeTPUInterpreter::invoke() {
    // Invoke the interpreter and return the status
    return interpreter->Invoke();
}

void EdgeTPUInterpreter::fillTensorInfo(const TfLiteTensor* tensor, TensorInfo& tensor_info) {
    // Set tensor size
    tensor_info.size = tensor->bytes / sizeof(int8_t);

    // Set tensor shape
    tensor_info.shape.clear();
    for (int i = 0; i < tensor->dims->size; ++i) {
        tensor_info.shape.push_back(tensor->dims->data[i]);
    }

    // Set tensor data pointer
    tensor_info.data_ptr = reinterpret_cast<int8_t*>(tensor->data.data);

    // Set tensor quantization parameters
    tensor_info.quantization.scale = tensor->params.scale;
    tensor_info.quantization.zero_point = tensor->params.zero_point;
}

const TensorInfo& EdgeTPUInterpreter::input_tensor_info() const {
    return input_tensor_info_;
}

const TensorInfo& EdgeTPUInterpreter::output_tensor_info() const {
    return output_tensor_info_;
}
