#include <TensorFlowLite.h>

#include "model_test.h"
#include "main_functions.h"
#include "predict.h"
#include "read_serial.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/testing/micro_test.h" //May be unecessary


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
TfLiteTensor* model_input;//Global tensor to replace model_input in loop function. Change to x test train data


int inference_time = 0;
float data_array[1152];
int array_index=0;
int input_length;


constexpr int kTensorArenaSize = 100 * 1024; //Set tensor arena size
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void setup() { //Setup
  pinMode(LED_BUILTIN, OUTPUT); //For debugging purposes
  Serial.begin(115200); //Starts the the serial monitor


  //Set up logging
  static tflite::MicroErrorReporter micro_error_reporter;
  //tflite::ErrorReporter* pointer was before but not needed apparently
  error_reporter = &micro_error_reporter;

  //Load model
  //const tflite::Model* again don't think pointer necessary 
  model = ::tflite::GetModel(g_test_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return; //Maybe not necessary, ^possibly more indent needed
  }

  //Instantiate operatations resolver
  static tflite::AllOpsResolver resolver;//Maybe change to micro_op_resolver

  //Instantiate interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  //Allocate tensors
  interpreter->AllocateTensors();

  model_input = interpreter->input(0);
  //Specifies the dimensions of the input tensor that will be received.
  
  input_length = model_input->bytes / sizeof(float);
}

void loop() {
  //Wait for Serial to connect, other ways of doing
  int prediction;
  int i=0;
  Initialise(data_array); //Initialises the model input
  while(!Serial.available()){
    //Wait till input received
  }
  
  while(array_index<1152) {
    data_array[array_index++]=readSerial(); //Reads 4 byte float value
    delay(0.1);
  }
  array_index=0;//resets array counter

  copyTensor(model_input->data.f, data_array); //Copies received float array to the input tensor memory location
  int temp=millis(); //Total time elapsed before inference
  TfLiteStatus invoke_status = interpreter->Invoke(); //Inference process
  temp=millis()-temp; //Total time elapsed after inference
  
  if (invoke_status != kTfLiteOk) { //Checks if inference was successful
    while(true){ //Infite loop to display that it was not
      Serial.println("Invoke failed on index");
      Serial.println(model_input->data.f[0]); //Displays the input tensor for debugging purposes
      delay(1000);                            //at a constant rate
    } 
  }//Failed invoke condition
      
  //Obtains prediction from interpreter
  prediction = predict(interpreter->output(0)->data.f);

  //Prediction output
  Serial.println(prediction);

  //Inference time output
  Serial.println(temp);

  Initialise(data_array);

}//Ends the loop
  
