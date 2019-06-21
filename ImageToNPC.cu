// Host is CPU 
// Device is GPU
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <cuda.h>


using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;


// Kernel
__global__ void processData( int *a, int width, int height ) {
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;


}

int main( void ) {
	//string filename = "/home/ubuntu/Documents/Blackfly_Images"; // Change later
	int exposureTime = 200; // In milliseconds
	float captureFPS = 4.9;
	int gain = 0; // ISO for digital cameras
	bool reverseX = false;
	bool reverseY = false;
	int bit = 8;
	Spinnaker::SystemPtr system = System::GetInstance();
	Spinnaker::CameraList camList = system->GetCameras();
	unsigned int numCameras = camList.GetSize();
	std::cout << "Number of cameras connected: " << numCameras << endl << endl;
	if (numCameras == 0) {
		std::cout << "No cameras detected." << endl;
		camList.Clear();
		system->ReleaseInstance();
		return -1;
	}

	CameraPtr cam = camList.GetByIndex( 0 );
	cam->Init();

	//load default config
	cam->UserSetSelector.SetValue( UserSetSelector_Default );
	cam->UserSetLoad();

	//set acquisition to continuous, turn off auto exposure, set the frame rate
	//Camera Settings
	//Set Packet Size
	cam->GevSCPSPacketSize.SetValue( 9000 );
	cam->DeviceLinkThroughputLimit.SetValue( 100000000 );
	//Set acquisition mode
	cam->AcquisitionMode.SetValue( AcquisitionMode_Continuous );
	//Set exposure time
	cam->ExposureAuto.SetValue( ExposureAuto_Off );
	cam->ExposureMode.SetValue( ExposureMode_Timed );
	cam->ExposureTime.SetValue( exposureTime * 1000 );
	//Set FPS
	cam->AcquisitionFrameRateEnable.SetValue( true );
	cam->AcquisitionFrameRate.SetValue( captureFPS );
	//set analog, gain, turn off gamma
	cam->GainAuto.SetValue( GainAuto_Off );
	cam->Gain.SetValue( gain );
	cam->GammaEnable.SetValue( false );

	cam->ReverseX.SetValue( reverseX );
	cam->ReverseY.SetValue( reverseY );
	if (bit > 8) {
		cam->AdcBitDepth.SetValue( AdcBitDepth_Bit12 );
		cam->PixelFormat.SetValue( PixelFormat_Mono12p );
	} else {
		cam->AdcBitDepth.SetValue( AdcBitDepth_Bit10 );
		cam->PixelFormat.SetValue( PixelFormat_Mono8 );
	}
	cam->BeginAcquisition();
	int i = 0;
	while ( i < 10 ) {
		int *a; // Host copy of a
		int *d_a; // Device copy of a
		ImagePtr imagePtr = cam->GetNextImage();
		int imageWidth = imagePtr->GetWidth();
		size_t imageHeight = imagePtr->GetHeight();
		size_t imageSize = imageWidth * imageHeight;
		int size = imageSize * sizeof( int );
		// Allocate space on device for copies of a, b, and c
		cudaMalloc( (void**)&d_a, size );
		//Alloc space for host copies of a, b, c and setup input values
		a = ( int* )malloc( size * sizeof(int) );
		a = static_cast<int*>( imagePtr->GetData() );
		std::cout << i << ": " << a;
		dim3 threadsPerBlock( 16, 16 ); // Creating 12x8 threadblock, will need 456 blocks
		dim3 numBlocks( (imageSize + 511) / 512 );

		cudaDeviceSynchronize();

		// Copy inputs to device
		cudaMemcpy( &d_a, &a, size, cudaMemcpyHostToDevice );
		processData<<<numBlocks,512>>>( d_a, imageWidth, imageHeight ); // Execute kernel with 512 threads on each block, enough blocks to cover whole image
		// Copy result back to host
		cudaMemcpy( &a, &d_a, size, cudaMemcpyDeviceToHost );
		free( a ); // Free host memory
		cudaFree( d_a ); // Free device memory
		imagePtr->Release();
		i = i + 1;

	}
	cam->EndAcquisition();
	cam->DeInit();
	camList.Clear(); // Clear list
	system->ReleaseInstance(); // Release system
	std::cout << endl << "Done. Press Enter to exit..." << endl;
	getchar();

	return 0;
}



