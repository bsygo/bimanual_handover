#!/bin/sh
wdir="$PWD"; [ "$PWD" = "/" ] && wdir=""
case "$0" in
  /*) scriptdir="${0}";;
  *) scriptdir="$wdir/${0#./}";;
esac
scriptdir="${scriptdir%/*}"
pcfile="$scriptdir/pc_full.pcd"
echo "$pcfile"
rosrun pcl_ros pcd_to_pointcloud $pcfile 5 _frame_id:="azure_kinect_rgb_camera_link_urdf"

