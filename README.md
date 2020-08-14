Tori is a 5'0" humanoid robot. It's currently capable only of walking short distances in a user-specified direction, but training is still in progress.
Tested on ROS Noetic.

To Use:
1. roslaunch tori_gazebo tori.launch (to view simulation, run gzclient in a separate terminal)
2. In walk_direction.py, edit last_saved_index to equal the number indicated in the most recent pkl files. If no pkl files are available, set this to 0.
3. Run walk_direction.py in one terminal and control_motion_walk_direction.py in another terminal.
4. The training will now begin, and progress will save every 1000 iterations.

Directional walking after 70000 iterations:

![70,000 iterations](https://github.com/Bentell-Robotics/Tori-Robot/blob/master/70000_iters.gif)

![real_robot](https://github.com/Bentell-Robotics/Tori-Robot/blob/master/tori_real_armless.jpg)
