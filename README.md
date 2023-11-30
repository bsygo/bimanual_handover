# bimanual_handover
plan:
-have object in 2 finger gripper
-observe object using azure kinect to get pointcloud
-preprocess pointcloud to filter robot out and crop to object (implemented)
-use gpd to generate grasp poses for the shadow hand (implemented)
-filter grasp poses by removing self collision poses (implemented)
-sample transforms to move into ideal handover poses
-move to handover poses
-learn handover:
    -open hand (implemented/tested)
    -move fingers until first contact (implemented/WIP)
    -start training: (partially implemented)
        -action: small synergy changes, decision to lift (partially implemented)
        -observation: joint states, gripper tactile, sh tactiles(, force/torque readings) (implemented)
        -reward: positive if closing motion, negative if opening motion, final reward after lift decision (partially implemented)
        -end: lift decision
    -repeat process for different objects until gerneralized
-perform handover
-move shadow hand to predetermined position and open to drop object

Workspace installation:
- create virtualenv with: virtualenv name 
- pip install required packages
- install gpd:
	- follow instructions on gpd git
	- modify CMAKE_INSTALL_PREFIX in CMakeCache.txt to point to desired library path
	- execute make install in build folder
	- add specified library path to LD_LIBRARY_PATH by writing in .bashrc(virtualenv activate):
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/library
- make gpd_ros ready to build:
	- change PATHS in find_library(GPD_LIB ...) to gpd library without the lib subdirectory
	- change include_idrectories to the include directory related to the gpd library instead of GPD_LIB_INCLUDE_DIR
- make orocos_kinematics_dynamics build:
	- go into python_orocos_kdl
	- comment out find_package(pybind11 ...) and related lines in CMakeLists.txt
	- only keep add subdirectory(pybind11) and comment out the rest until pybind11_add_module
- build workspace 

Model Information:
All models before 04.10.2023 were run on MultiInputPolicy for PPO using the MimicEnv with all jo    int values in joint spaces as observation space in a dict. They further had 4 values as action,     the first 3 for PCA values and the last one for finishing, which wasn't used.
The model 29_09_2023 is the best, where fingers are closing and only the thumb is an issue. It w    as trained for 100000 steps.
The model 02_10_2023 was trained for 1000000 steps and is useless despite longer training.
The previous models don't work either and were trained for 10000 steps or fewer.

The models 04_10_2023 and after are trained for 100000 steps with the MlpPolicy for PPO and usin    g the updated MimicEnv, which has only 3 values as the joint values encoded in PCA space as obse    rvation space as a Box. They also have only 3 PCA values as action.
04_10_2023 failed due to a bug, where the current pca values would be falsely updated.
06_10_2023 works very well, if the joints 4 for each finger except the thumb are limited to rema    in in the current configurations (otherwise collisions occur).
