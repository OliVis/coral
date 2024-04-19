#include <iostream>
#include "coral/interpreter.h"

EdgeTPUInterpreter::EdgeTPUInterpreter(const char* model_path)
{
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
        std::cerr << "Failed to allocate tensors." << std::endl;
    }

    // Get input tensor quantization parameters
    input_quant.scale = interpreter->input_tensor(0)->params.scale;
    input_quant.zero_point = interpreter->input_tensor(0)->params.zero_point;

    // Get output tensor quantization parameters
    output_quant.scale = interpreter->output_tensor(0)->params.scale;
    output_quant.zero_point = interpreter->output_tensor(0)->params.zero_point;

    // Get pointers to input and output tensors
    input_data_ptr = interpreter->typed_input_tensor<int8_t>(0);
    output_data_ptr = interpreter->typed_output_tensor<int8_t>(0);
}

EdgeTPUInterpreter::~EdgeTPUInterpreter()
{
    // Releases interpreter instance to free up resources associated with this custom op
    interpreter.reset();

    // Closes the Edge TPU
    edgetpu_context.reset();
}

TfLiteStatus EdgeTPUInterpreter::invoke()
{
    // Invoke the interpreter and return the status
    return interpreter->Invoke();
}
