#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <stdio.h>
#include <time.h>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace std;
#define NUMBER_THREADS  512
// CUDA Kernel to process image as array
__global__ void process(int *inputArray, int *exportArray, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // "Column" if array was 2D
    int row = blockIdx.y * blockDim.y + threadIdx.y; // "Row" if array was 2D
    int index = row * width + col; // Translate (Column, Row) into index in 1D Memory allocation
    if (row > height || col > width) { // If the row or column is out of bounds, return and exit kernel
        return;
    }
        
	if (inputArray[index] > 50) { // If the value is above a threshold,
	    exportArray[index] = 100; // Replace that value
	} else { // Otherwise,
	    exportArray[index] = 0; // Replace with other value
	}
}

// Main method
int main(void) {
	Spinnaker::SystemPtr system = System::GetInstance(); // Get camera system
	Spinnaker::CameraList camList = system->GetCameras(); // Get camera list
	Spinnaker::CameraPtr pCam = camList[0]; // Get first camera
	pCam->Init(); // Initialize camera instance
	INodeMap & nodeMap = pCam->GetNodeMap(); // Acquire node map from the camera
	CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode"); // Change acquisition mode to continuous
    if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode)) { // If the acquisiton mode can't be edited, exit program
        cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
        return -1;
    }
    // Retrieve entry node from enumeration node
    CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous"); // acquire acquisition mode
    if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous)) { // If the continuous acquisition mode isn't readable, exit program
        cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
        return -1;
    }           
    // Retrieve integer value from entry node
    int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();
    // Set integer value from entry node as new value of enumeration node
    ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous); // Set acquisition program to continuous
	pCam->BeginAcquisition(); // Begin acquisition
	int numImages = 10;
	
	for (int i = 0; i < numImages; i++) { // For n images, 
	    Spinnaker::ImagePtr pResultImage = pCam->GetNextImage(); //Get next image
	    Spinnaker::ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8); // Set the pixel values to Monotone, with 8 bits per image
	    int width = pResultImage->GetWidth(); // Find width and height of image
	    int height = pResultImage->GetHeight();
	    
	    int *imageArray, *resultArray;	// declare two Host side memory allocations
	    int *device_imageArray, *device_resultArray; // declare two Device side memory allocations
	    size_t size = NUMBER_THREADS * sizeof(convertedImage); // find amount of memory to be allocated
	    
	    cudaMalloc((int**)&device_imageArray, size); // allocate "size" amount of memory to the device pointers
	    cudaMalloc((int**)&device_resultArray, size);

	    imageArray = (int*)malloc( size ); // Allocate "size" amount of memory to the host pointers
	    resultArray = (int*)malloc( size );

        imageArray = (int*)convertedImage->GetData(); // Put the image's data into the imageArray memory allocation
	    for (int i = 0; i < 1000; i++) { // Print out image's first 100 values
	        cout << imageArray[i] << " ";
	    }
	    cout << endl; // new line
        int numBlocks = (((width*height) + NUMBER_THREADS - 1) / NUMBER_THREADS); // Find number of blocks, should be the number of blocks needed to cover the whole image

	    cudaMemcpy(device_imageArray, imageArray, size, cudaMemcpyHostToDevice); // copy the values in host side array to device array
	    double starttime = clock(); // start timer to find time needed to process on gpu
	    process<<< numBlocks, NUMBER_THREADS >>>(device_imageArray, device_resultArray, width, height); // run kernel
	    double endtime = clock(); // end timer
	    cudaMemcpy(resultArray, device_resultArray, size, cudaMemcpyDeviceToHost); // copy the values back

	    double interval = (endtime - starttime) / (double)CLOCKS_PER_SEC; // Find time taken in ms
	    printf("Execution time: %f ms \n\n", interval * 1000); // Print value
	    for(int i = 0; i < 1000; i++) { // Print out new values of image
		    cout << resultArray[i] << " ";
	    }
	    free(imageArray); free(resultArray); // free host mem
	    cudaFree(device_imageArray); // free device mem
	    cudaFree(device_resultArray); 
	    pResultImage->Release(); // Release image instances
	    convertedImage->Release();
    }        

	
	pCam->EndAcquisition(); // End acquisition
	pCam  = NULL; // Release camera
    camList.Clear(); // clear list of cameras
    system->ReleaseInstance(); // release system instance

	return 0;

}
