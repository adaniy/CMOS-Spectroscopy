#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <stdio.h>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

static size_t height;
static size_t width;
#define NThreads 512

__global__ void processData(int *device_imagePtr) {
	const int imageWidth = 5472;
	const int imageHeight = 3648;
	const int imageSize = imageWidth * imageHeight;
	int row = (blockIdx.y * blockDim.x * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x * blockDim.y) + threadIdx.x;
	int index = (row * imageWidth) + col;
	
	device_imagePtr[index] = 19;
	/*if (row < imageHeight && col < imageWidth) {
		if (device_imagePtr[index] >= 127) {
			device_imagePtr[index] = 127;
		} else {
			device_imagePtr[index] = 0;
		}
	}*/
}

void ConvertToArray(ImagePtr pImage, int* imageData) {
    ImagePtr convertedImage = pImage->Convert(PixelFormat_BGR8, NEAREST_NEIGHBOR);

    unsigned int XPadding = convertedImage->GetXPadding();
    unsigned int YPadding = convertedImage->GetYPadding();
    unsigned int rowsize = convertedImage->GetWidth();
    unsigned int colsize = convertedImage->GetHeight();

    //image data contains padding. When allocating Mat container size, you need to account for the X,Y image data padding. 
    cv::Mat cvimage = cv::Mat(colsize + YPadding, rowsize + XPadding, CV_8UC1, convertedImage->GetData(), convertedImage->GetStride());
	imageData = new int(cvimage.rows * cvimage.cols * cvimage.channels()); // Dst
	int* imageDataPointer = reinterpret_cast<int*>(cvimage.data); // Src
	// pointer from, pointer two, size in bytes
	std::memcpy(imageData, imageDataPointer, cvimage.rows * cvimage.cols * cvimage.channels() * sizeof(int));
		

}


// This function acquires and saves 10 images from a device.  
int AcquireImages(CameraPtr pCam, INodeMap & nodeMap, INodeMap & nodeMapTLDevice)
{
        int result = 0;
        
        cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;
        
        try
        {

                // Set acquisition mode to continuous

                // Retrieve enumeration node from nodemap
                CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
                if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode)) {
                        cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
                        return -1;
                }
                
                // Retrieve entry node from enumeration node
                CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
                if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous)) {
                        cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
                        return -1;
                }
                
                // Retrieve integer value from entry node
                int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();
                
                // Set integer value from entry node as new value of enumeration node
                ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);
                
                cout << "Acquisition mode set to continuous..." << endl;
                
				CEnumerationPtr ptrAdcBitDepth = nodeMap.GetNode("AdcBitDepth");
				if (!IsAvailable(ptrAdcBitDepth) || !IsWritable(ptrAdcBitDepth)) {
					cout << "Unable to set AdcBitDepth to Bit 10 (enum retrieval). Aborting...)" << endl << endl;
					return -1;
				}
				CEnumEntryPtr ptrAdcBitDepthBit10 = ptrAdcBitDepth->GetEntryByName("Bit10");
				if (!IsAvailable(ptrAdcBitDepthBit10) || !IsReadable(ptrAdcBitDepthBit10)) {
						cout << "Unable to set AdcBitDepth to Bit 10 (entry retrieval). Aborting..." << endl << endl;            
                		return -1;
				}
				int64_t adcBitDepthBit10 = ptrAdcBitDepthBit10->GetValue();
				ptrAdcBitDepth->SetIntValue(adcBitDepthBit10);
				
				cout << "AdcBitDepth set to Bit10..." << endl;
// Begin acquiring images
             
                pCam->BeginAcquisition();
                cout << "Acquiring images..." << endl;
                
                // Retrieve device serial number for filename
             
                gcstring deviceSerialNumber("");
                CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
                if (IsAvailable(ptrStringSerial) && IsReadable(ptrStringSerial))
                {
                        deviceSerialNumber = ptrStringSerial->GetValue();
                        cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
                }
                cout << endl;
                
                // Retrieve, convert, and save images
                const unsigned int k_numImages = 10;

                for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
                {
                        try
                        {
                                // Retrieve next received image
                                ImagePtr pResultImage = pCam->GetNextImage();
                                
                                // Ensure image completion
                              
                                if (pResultImage->IsIncomplete()) {
                                        
                                        cout << "Image incomplete with image status " << pResultImage->GetImageStatus() << "..." << endl << endl;
                                }
                                else {

                                        // Print image information; height and width recorded in pixels
                                        
                                        width = pResultImage->GetWidth();
                                        
                                        height = pResultImage->GetHeight();
                                        int size = width * height * sizeof(int);
										
                                        cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height << ", memory allocated = " << size << " bytes" << endl;
										
                                        // Convert image to mono 8
                                        ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8);
										int numBlocks( ((size / sizeof(int)) + NThreads - 1) / NThreads );
										// Convert to array
										//static int *finalPixelInformationPointer;
										//static int *device_finalPixelInformationPointer;
										static int *imageArrayPtr;
										static int *device_imageArrayPtr;
										
										cudaMalloc((int**)device_imageArrayPtr, size);
										imageArrayPtr = (int*)malloc( size );

										unsigned int XPadding = convertedImage->GetXPadding();
										unsigned int YPadding = convertedImage->GetYPadding();
										unsigned int rowsize = convertedImage->GetWidth();
										unsigned int colsize = convertedImage->GetHeight();

										//image data contains padding. When allocating Mat container size, you need to account for the X,Y image data padding. 
										cv::Mat cvimage = cv::Mat(colsize + YPadding, rowsize + XPadding, CV_8UC1, convertedImage->GetData(), convertedImage->GetStride());
										imageArrayPtr = reinterpret_cast<int*>(cvimage.data); // Src
										// pointer from, pointer two, size in bytes

										//ConvertToArray(convertedImage, imageArrayPtr); // Seg fault is here, data is not properly written through buffer before being read
										//finalPixelInformationHolder = (int*)malloc(pixelHolderSize);
										//cudaMalloc((int**)device_finalPixelInformationHolder, pixelHolderSize)
										
										for (int i = 0; i < 100; i++) {
											cout << imageArrayPtr[i] << " ";
										}
										
										cudaMemcpy( device_imageArrayPtr, imageArrayPtr, size, cudaMemcpyHostToDevice );
										cout << endl << "Processing..." << endl;
										//cudaMemcpy(device_finalPixelInformationHolder, finalPixelInformationHolder, pixelHolderSize, cudaMemcpyHostToDevice);
										processData<<<numBlocks, NThreads>>>(device_imageArrayPtr); // FIXME
										
										//cudaMemcpy(finalPixelInformationHolder, device_finalPixelInformationHolder, pixelHolderSize, cudaMemcpyDeviceToHost);
										cudaMemcpy( imageArrayPtr, device_imageArrayPtr, size, cudaMemcpyDeviceToHost );
										cout << endl << "Processed." << endl;										
										for (int i = 0; i < 100; i++) {
											cout << imageArrayPtr[i] << " ";
										}

										ImagePtr saveImage = Spinnaker::Image::Create(width, height, 0, 0, PixelFormat_Mono8, (void*)imageArrayPtr);

										ostringstream filename;
                                        
                                        filename << "Acquisition-";
                                        if (deviceSerialNumber != "") {
                                                        filename << deviceSerialNumber.c_str() << "-";
                                        }
                                        filename << imageCnt << ".jpg";
                                        // Save image
                                        
										saveImage->Save(filename.str().c_str());
		                                cout << "Image saved at " << filename.str() << endl;
										
										//free( imageArrayPtr );
										//cudaFree( device_imageArrayPtr );
                                }
                                // Release image
                                pResultImage->Release();

                                cout << endl;
                        }
                        catch (Spinnaker::Exception &e) {
                                cout << "Error: " << e.what() << endl;
                                result = -1;
                        }
                }
                
                // End acquisition
              
                pCam->EndAcquisition();
        }
        catch (Spinnaker::Exception &e) {
                cout << "Error: " << e.what() << endl;
                result = -1;
        }
        
        return result;
}
// This function prints the device information of the camera from the transport
int PrintDeviceInfo(INodeMap & nodeMap) {
        int result = 0;
        
        cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;
        try {
                FeatureList_t features;
                CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
                if (IsAvailable(category) && IsReadable(category)) {
                        category->GetFeatures(features);
                        FeatureList_t::const_iterator it;
                        for (it = features.begin(); it != features.end(); ++it) {
                                CNodePtr pfeatureNode = *it;
                                cout << pfeatureNode->GetName() << " : ";
                                CValuePtr pValue = (CValuePtr)pfeatureNode;
                                cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
                                cout << endl;
                        }
                }
                else
                {
                        cout << "Device control information not available." << endl;
                }
        }
        catch (Spinnaker::Exception &e) {
                cout << "Error: " << e.what() << endl;
                result = -1;
        }
        
        return result;
}
// This function acts as the body of the example; please see NodeMapInfo example 
// for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam) {
        int result = 0;
        try {
                // Retrieve TL device nodemap and print device information
                INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
                
                result = PrintDeviceInfo(nodeMapTLDevice);
                
                // Initialize camera
                pCam->Init();
                
                // Retrieve GenICam nodemap
                INodeMap & nodeMap = pCam->GetNodeMap();
                // Acquire images
                result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice);
                
                // Deinitialize camera
                pCam->DeInit();
        }
        catch (Spinnaker::Exception &e) {
                cout << "Error: " << e.what() << endl;
                result = -1;
        }
        return result;
}
// Example entry point; please see Enumeration example for more in-depth 
// comments on preparing and cleaning up the system.
int main(int /*argc*/, char** /*argv*/) {

        
	  	int result = 0;
		// Retrieve singleton reference to system object
		SystemPtr system = System::GetInstance();
		// Retrieve list of cameras from the system
		CameraList camList = system->GetCameras();
        unsigned int numCameras = camList.GetSize();
        
        cout << "Number of cameras detected: " << numCameras << endl << endl;
        cout << "Press enter to start, press ctrl+c to stop";
		getchar();
	
        // Finish if there are no cameras
        if (numCameras == 0) {
                // Clear camera list before releasing system
                camList.Clear();
                // Release system
                system->ReleaseInstance();
                cout << "Not enough cameras!" << endl;
                cout << "Done! Press Enter to exit..." << endl;
                getchar();
                
                return -1;
        }
        
        CameraPtr pCam = NULL;
        // Run example on each camera
        for (unsigned int i = 0; i < numCameras; i++) {
                // Select camera
                pCam = camList.GetByIndex(i);

                cout << endl << "Running for camera " << i << "..." << endl;
                
                // Run example
                result = result | RunSingleCamera(pCam);
                
                cout << "Camera " << i << " example complete..." << endl << endl;
        }

        // Release reference to the camera
        pCam = NULL;
        // Clear camera list before releasing system
        camList.Clear();
        // Release system
        system->ReleaseInstance();
        cout << endl << "Done. Press Enter to exit..." << endl;
        getchar();
        return result;
}
