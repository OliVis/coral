#include <iostream>
#include <edgetpu.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

int main() {
    // Create the EdgeTpuContext object
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
        edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    // Load the compiled Edge TPU model as a FlatBufferModel
    const std::string model_path = "../test_edgetpu.tflite";
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    
    // Registers Edge TPU custom op handler with Tflite resolver
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    
    // Build the interpreter
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    // Bind the context with the interpreter
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());

    interpreter->SetNumThreads(1);
    
    if (interpreter->AllocateTensors() != kTfLiteOk)
        std::cerr << "Failed to allocate tensors." << std::endl;

    // Releases interpreter instance to free up resources associated with this custom op
    interpreter.reset();

    // Closes the Edge TPU
    edgetpu_context.reset();
    
    return 0;
}