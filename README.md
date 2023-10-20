# Vehicle Detection System README

## Overview

The Vehicle Detection System is a computer vision project engineered for the purpose of detecting and tracking vehicles within both image and video streams. This system leverages cutting-edge machine learning and computer vision methodologies to accurately identify and monitor vehicles, rendering it applicable to a wide range of use cases, including traffic monitoring, surveillance, and autonomous vehicles.

In this specific context, we are in the process of developing a program with the primary objective of automatically recognizing vehicle license plates as vehicles enter designated parking areas. The program is meticulously engineered to capture license plate numbers and the corresponding entry timestamps, thereby streamlining and enhancing the efficiency of vehicle access and parking record management.

In this implementation, we integrate a pre-trained model into the system, which is capable of extracting license plate information from video feeds. The system employs a pattern recognition algorithm, which evaluates the obtained data and selects patterns exceeding a predefined threshold of 40%. These selected values are subsequently archived within a newly generated CSV file for further reference and analysis.

## Features

1. Vehicle detection in video streams.
2. Real-time tracking of vehicles.
3. Support input sources, such as live camera feeds.
4. Performance optimization for real-time applications.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Nihal-7/codeXplorer_vehicle_recognition -b master
2. Open folder named as Licence_Plate_Recognition_YOLO_V8_EasyOCR.ipynb in Google Colab.
3. And run all the scripts one by one.

## Sample Outputs

After Running all the Scripts, Results will be saved in results.csv and Output video in runs/detect folder.

## References

1.Information about Training, Test and Valid data - https://www.youtube.com/watch?v=XCYlRBf18YI&pp=ygUodHJhaW4gdmFsaWQgYW5kIHRlc3QgaW4gbWFjaGluZSBsZWFybmluZw%3D%3D

2.The AI University : https://www.youtube.com/watch?v=JGmAbuetSmI&list=PLlH6o4fAIji76vBRv54WPQr0MHgEiipL7

3.Understanding of Easy Ocr - https://github.com/JaidedAI/EasyOCR



