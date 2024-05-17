from robot.robotiq.robotiq_gripper import RobotiqGripper

#Test
if True:
    grip=RobotiqGripper("/dev/ttyUSB0")
    print("init done...")
    
    input("Press Enter to continue...")
    
    grip.closeGripper()
    
    input("Press Enter to continue...")

    grip.openGripper()

    input("Press Enter to continue...")
    
    grip.calibrate(0,40)
    grip.goTomm(20,255,255)
    grip.goTomm(40,1,255)
    
    input("Ready? Press Enter to continue...")
    grip.closeGripper()