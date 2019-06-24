// Host is CPU 
// Device is GPU
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream> 
using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

__global__ void processData(int *d_A, int numBlocks) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x*blockDim.y)+ (threadIdx.y*blockDim.x)+threadIdx.x;

	if (threadId >= 19961856)return;
	
	if (d_A[threadId] >= 127) d_A[threadId] = 255;
	else d_A[threadId] = 0;
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
                if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
                {
                        cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
                        return -1;
                }
                
                // Retrieve entry node from enumeration node
                CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
                if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
                {
                        cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
                        return -1;
                }
                
                // Retrieve integer value from entry node
                int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();
                
                // Set integer value from entry node as new value of enumeration node
                ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);
                
                cout << "Acquisition mode set to continuous..." << endl;
                
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
                                        
                                        size_t width = pResultImage->GetWidth();
                                        
                                        size_t height = pResultImage->GetHeight();
                                        int size = width * height * sizeof(int);
                                        cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height << endl;

                                        // Convert image to mono 8
                                       
                                        ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);
					// Convert to array
					int *a;
					a = static_cast<int*>( pResultImage->GetData() );
					int *d_a; // Device copy of a

					cudaMalloc( (void**)&d_a, size );
					dim3 threadsPerBlock( 16, 16 ); // Creating 16x16 threadblock, will need 456 blocks
					int numBlocks( (size + 511) / 512 );
					cudaMemcpy( &d_a, &a, size, cudaMemcpyHostToDevice );
					cout << "Processing..." << endl;
					processData<<<numBlocks,512>>>( d_a, numBlocks ); // Execute kernel with 512 threads on each block, enough blocks to cover whole image
					cout << "Processed." << endl;					
	
					// Copy result back to host
					cudaMemcpy( &a, &d_a, size, cudaMemcpyDeviceToHost );
					free( a ); // Free host memory
					cudaFree( d_a ); // Free device memory



                                        ostringstream filename;
                                        
                                        filename << "Acquisition-";
                                        if (deviceSerialNumber != "") {
                                                        filename << deviceSerialNumber.c_str() << "-";
                                        }
                                        filename << imageCnt << ".jpg";
                                        // Save image
                                        convertedImage->Save(filename.str().c_str());
                                        cout << "Image saved at " << filename.str() << endl;
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
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
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
	cin;
	
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
                cout << endl << "Running example for camera " << i << "..." << endl;
                
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
