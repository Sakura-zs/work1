#ifndef _SSD_ADAPTER_H_
#define _SSD_ADAPTER_H_

#include <map>
#include <vector>

#include <Eigen/Eigen>

#include <rosa/rosa_main.h>

namespace fast_planner
{

struct SsdGraphData
{
  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector2i> edges;
  std::vector<std::vector<int>> branches;
  std::vector<std::vector<int>> branch_segment_pairs;
  std::map<int, Eigen::Vector2i> segments;
};

struct SsdCloudData
{
  std::map<int, std::vector<Eigen::Vector3d>> segment_clouds;
  std::vector<std::vector<Eigen::Vector3d>> subspace_clouds;
};

struct SsdStatsData
{
  std::map<int, double> segment_inner_distance;
  int original_cloud_size = 0;
};

struct SsdExportData
{
  SsdGraphData graph;
  SsdCloudData cloud;
  SsdStatsData stats;
};

class SsdResultAdapter
{
public:
  static SsdExportData Convert(const Pcloud& source);
};

} // namespace fast_planner

#endif
