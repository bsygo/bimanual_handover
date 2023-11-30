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
.move shadow hand to predetermined position and open to drop object
