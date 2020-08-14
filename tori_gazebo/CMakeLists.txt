cmake_minimum_required(VERSION 2.8.3)

project(tori_gazebo)

find_package(catkin REQUIRED COMPONENTS rospy roscpp std_msgs message_generation)



add_message_files(
  FILES
  Tori_joint_positions.msg
  ToriJointAngles.msg
  State.msg
  Replay.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

if (CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  #roslaunch_add_file_check(tests/roslaunch_test.xml)
endif()




install(DIRECTORY launch controller DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
