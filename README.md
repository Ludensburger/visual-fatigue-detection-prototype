# visual-fatigue-detection-prototype

OpenCV Installation Error
You're trying to install OpenCV using npm (Node.js package manager), but your project is a Python application. You need to use pip instead to install the required Python packages.

Install Required Python Packages
Open a command prompt and run the following commands:

```
pip install opencv-python
pip install dlib numpy pandas scipy
pip install mediapipe
```


If you encounter issues installing dlib, you might need Visual C++ build tools:
```
pip install cmake
pip install dlib

```