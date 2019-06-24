# PySpin-Processing

Steps to run with fresh Linux installation


1. Install nano command to edit files:

       sudo apt-get install nano

2. Add Cuda to PATH:

       sudo nano /home/username/.bashrc

      At the bottom of the file add:
  
       export PATH=/usr/local/cudax-x.x/bin${PATH:+:${PATH}}$
  
       export LD_LIBRARY_PATH=/usr/local/cuda-x.x/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
 
  b. Save and exit file:
       ctrl + x -> y -> enter

3. Restart terminal

4. Navigate to directory, then:

       cmake .

       make

       ./acq

OR:

        nvcc --version 
to ensure compiler is installed
