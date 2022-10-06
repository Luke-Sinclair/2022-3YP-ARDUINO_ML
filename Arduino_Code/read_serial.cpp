#include "Arduino.h"
#include "read_serial.h"

float readSerial(){
  union { //Allows the storage of these different data types
    byte data[4];
    float value;
  } convertFloat;
  while(Serial.available()<4){
    //do nothing
    }
  size_t bytesRead = Serial.readBytes(convertFloat.data,4); //Reconstructs the send float vlaue
  return convertFloat.value;
}

//Copies the data array to the input tensor location 
void copyTensor(float* input, float* data){
  for (int i=0; i<1152; i++) {
    input[i]=data[i];
  }
}

//Initalises float array
void Initialise(float* data){
  for(int i=0; i<1152; i++)
  {
    data[i]=0;
  }
}
