class State:
    def __init__(self, joint_states, rpy, rpy_vel, positions, position_vel, true_joint_states, y_pos):
        self.joint_states = joint_states
        self.rpy = rpy
        self.rpy_vel = rpy_vel
        self.positions = positions
        self.position_vel = position_vel
        self.true_joint_states = true_joint_states
        self.y_pos = y_pos
