#include <ros/ros.h>

#include <rosa/rosa_main.h>
#include <rosa/ssd_adapter.h>

using namespace fast_planner;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ssd_adapter_demo");
  ros::NodeHandle nh("~");

  ROSA_main rosa;
  rosa.init(nh);
  rosa.main();

  const SsdExportData result = SsdResultAdapter::Convert(rosa.P);

  ROS_INFO("[SSD Adapter] vertices=%zu edges=%zu branches=%zu", result.graph.vertices.size(), result.graph.edges.size(), result.graph.branches.size());
  ROS_INFO("[SSD Adapter] segments=%zu subspaces=%zu original_points=%d", result.cloud.segment_clouds.size(), result.cloud.subspace_clouds.size(), result.stats.original_cloud_size);

  return 0;
}
