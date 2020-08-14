Tori is a 5'0" humanoid robot. It's currently capable only of walking short distances in a user-specified direction, but training is still in progress.

To Use:
1. In walk_direction.py, edit last_saved_index to equal the number indicated in the most recent pkl files. If no pkl files are available, set this to 0.
2. Using ROS Noetic, run walk_direction.py in one terminal and control_motion_walk_direction.py in another terminal.
3. The training will now begin, and progress will save every 1000 iterations.

Directional walking after 70000 iterations:

![70,000 iterations](https://github.com/Bentell-Robotics/Tori-Robot/blob/master/70000_iters.gif)

![real_robot](https://github.com/Bentell-Robotics/Tori-Robot/blob/master/tori_real_armless.jpg)
