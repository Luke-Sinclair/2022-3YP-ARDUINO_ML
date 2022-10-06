#include "predict.h"

//Prediction function
int predict(float* output) {

  float predict = output[0];
  int highest=0;
  //algorithm to return the predicted class
  for (int i = 1; i < 6; i++) {  
      if(output[i]>predict) {
        predict=output[i];
        highest=i; //Stores the index of the highest array element
      }
    }
  return highest; //Returns the index
}
