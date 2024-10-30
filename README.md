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


## Installation Steps

1. **Step 1: Device Interfaces**:
    - Make sure the devices are connected and working properly.
    - Pre-requisite for the vision system is two cameras, one robot arm, and one gripper.

    > Note: Notice that the relevant repositories in the device interfaces are all ROS packages. If using ROS-wrapper device interface, then they need to be started seperately. Otherwise, direct device SDK based interfaces can be directly implemented in `camera/` and `robot/` modules of the vision system. For example, 'Orbbec' camera SDK interface is directly implemented in `camera/orbbec` module and 'Robotiq' gripper SDK interface is directly implemented in `robot/robotiq` module.

2. **Step 2: Systems Check**:
    - Setup the python virtual environment and install the required packages.
        ```shell
        $ python3 -m venv venv
        $ source venv/bin/activate
        $ pip install -e .
        ```
    - Run the `test.py` script in the `camera/` and `robot/` modules to check the device interfaces.
    - If all tests pass, then the devices are ready for the vision system.
<br><br>

3. **Step 3: Device Setup: Camera-Robot Calibration**:
    - Review the script `gripper_camera_calibration.py` in the `scripts/` directory to make sure it uses the right device interfaces.
    - Run the script to calibrate the camera and robot arm.
        ```shell
        # Activate the virtual environment, if not already done.
        $ source venv/bin/activate

        # Run the calibration script
        $ python scripts/gripper_camera_calibration.py
        ```
    - Calibration will be saved in the `data/` directory.
<br><br>

4. **Step 4: Create Config Files**:
    ```shell
    # Create directory for training data to be stored
    $ mkdir -p data/demonstrations/<dd-mm>

    # Copy a config file from demo-example
    $ cp demo-exmaple/demonstrations/08-14-wp/place_object.yaml data/demonstrations/<dd-mm>/
    ```
    Update the `place_object.yaml` config file with your setup parameters.   
<br>

5. **Step 5: TEACH - Collect training data**:
    - Use `collect_demonstrations()` method in `TeachPlace` class to collect training data. `place_teach.py` script can be used to run the teach script.
        ```shell
        # Activate the virtual environment, if not already done.
        $ source venv/bin/activate

        # Run the teach script
        $ python scripts/place_teach.py --config ../data/demonstrations/<dd-mm>/place_object.yaml
        ```
        > A bash script can also be created to run the teach script for different connectors. See *.sh files in the `scripts/` directory for examples.
    - Data is collected in the `data/demonstrations/<dd-mm>` directory. Review the depth images of action and anchor objects to make sure the data is collected correctly. Adjust the parameters in `training` section in `place_object.yaml` file to get good quality data.
    - Use `scripts/view_ply.py` script to visualize the point cloud data.
<br><br>

6. **Step 6: LEARN - Train the model**:
    - **Data Preparation**. `prepare_data()` method in `LearnPlace` class is used to prepare the training data. `place_learn.py` script shows how to execute the method. <br>
        Review `training` parameter in `place_object.yaml` file to make sure the data is prepared correctly.
        ```shell
        # Activate the virtual environment, if not already done.
        $ source venv/bin/activate

        # Run the learn script
        $ python scripts/place_learn.py --config ../data/demonstrations/<dd-mm>/place_object.yaml
        ```
    - **Training the Model**. 
        - Review instructions in `model/README.md` to configure training parameters.
        - Run the training script.
        ```shell
        # (optional) Run the model training in a terminal multiplexer like tmux or screen.
        tmux new -s vision-training
        
        # Activate the virtual environment, if not already done.
        $ source venv/bin/activate

        # Run the training script
        $ cd model/taxpose
        $ CUDA_VISIBLE_DEVICES=1 python scripts/train_residual_flow.py --config-name <path/to/taxpose/training/config>
        ``` 
        - Training will take a few hours to complete. Monitor the training progress on WANDB dashboard.
        - Update the `training.model_config` parameter in the `place_object.yaml` file with the `<path/to/taxpose/training/config>`
        - Once the training is complete, save the location of the trained model in the `models/taxpose/configs/checkpoints` directory.
<br><br>

7. **Step 7: EXECUTE - Run the vision system**:
    - Review the `execution` parameter in the `place_object.yaml` file.
    - Use `execute()` method in `ExecutePlace` class to run the vision system. `place_execute.py` script can be used to run the execute script.
    ```shell
    # Activate the virtual environment, if not already done.
    $ source venv/bin/activate

    # Run the execute script
    $ python scripts/place_execute.py --config ../data/demonstrations/<dd-mm>/place_object.yaml
    ```

8. **Step 8: Validate and Retrain**:
    - To validate the system use `validate_execute()` method in `ExecutePlace` class.
    - Run it a few times to get a sample set of data.
    - Use the `notebooks/visualize_pcd.ipynb` notebook to calculate the error and visualize the point cloud data for the action and anchor objects.
    - If the error is high, then retrain the model by repeating steps 5-6-7.    
