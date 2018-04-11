# Football-Offsideline-Detection

using Python with Numpy and OpenCV.

Algorithm is based on following Paper: 

https://web.stanford.edu/class/ee368/Project_Spring_1415/Reports/Cheshire_Halasz_Perin.pdf

Requirements:
* OpenCV 3.4.1
* Python 3.6.5
* Python-Packages: scipy, numpy, argparse, opencv

Usage with **python3 detect.py -v Offside_normal.mp4** <br/>
For better understanding, uncomment lines after DEBUG-comments in python-file
<br/>
<br/>
Script hsv.py can be used to determine thresholds for team colors. <br/>
Usage with **python3 hsv.py -i Offside_normal.mp4** 

## Known issues
* Currently only working with example video
* Need two field lines for offside-line calculation
* Player detection needs some improvement 
