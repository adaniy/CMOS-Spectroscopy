# PySpin-Processing

Steps to run with fresh Linux installation with Ubuntu


1. Install nano command to edit files
  a. sudo apt-get install nano

2. Add Cuda to PATH
  a. sudo nano /home/username/.bashrc
  At the bottom of the file add:
  export PATH=/usr/local/cudax-x.x/bin${PATH:+:${PATH}}$
  export LD_LIBRARY_PATH=/usr/local/cuda-x.x/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
  b. ctrl+x -> y -> enter

3. Restart terminal

4. Navigate to directory

5. cmake .

6. make

7. ./acq

OR

Run: nvcc --version to ensure compiler is installed
