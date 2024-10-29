# Vision-based Insertion Skill Learning

This repository contains the code for a project sponsored by NIST in 2024. The goal of the project is to develop a vision-based skill learning system for robotic insertion tasks. Details of the project can be found in the [project page](https://cmu-mfi.github.io/rtc/).

```
git clone --recursive https://github.com/cmu-mfi/rtc_vision_toolbox.git
```

## Important Notes

* `camera` and `robot` folders contain the code for interfacing with the camera and robot, respectively. Currently, they have interface available for a few devices but can be easily extended to other devices. Make sure to follow same class structure as in the existing interfaces.

    * Each has `tests` folder to test the interface. Run the tests to make sure the interface is working correctly.

* `demo-example` folder contains training and inference data collected while running the vision system. Rename it to `demo` and use it as a reference to collect your own data.

* `rtc_core` folder contains the core code for the vision system. Avoid changing the code in this folder unless you know what you are doing.

* `scripts` contains scripts to train and test the vision system. Use the scripts as is or modify them to suit your needs.