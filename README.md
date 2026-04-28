# Setup
Once you have cloned the code you will need a venv - I am not going to explain how to do this as it varies depending on the system. This link explains the process quite well.
[W3 Schools - VENV](https://www.w3schools.com/python/python_virtualenv.asp)

After you have made the venv you need to install the requirements
```sh
pip install -r requirements.txt
```

# Running the programs
I would recommend running the webcam code, partially since the robot will be using a live camera and not loading from an image and also since I've commented that code. To run this simply run
```sh
python3 apriltag_from_webcam.py
```
To quit the program hit `esc`


# What is the code doing?
We are using opencv which is a wonderful, extremely powerful library for python (and C++). We use this to open an instance of the webcam, convert it to greyscale - since the apriltags are only black and white we can save some processing by using greyscale images. This gets passed to the apriltag library which does the detection using the tag family of "Circle21h7". Then we iterate over all the detected tags that are above a certain level of confidence, and draw a box around them.