#include <rosa/ssd_adapter.h>

namespace fast_planner
{

namespace
{

Eigen::Vector3d ToVec3(const pcl::PointXYZ& point)
{
  return Eigen::Vector3d(point.x, point.y, point.z);
}

std::vector<Eigen::Vector3d> ConvertCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  std::vector<Eigen::Vector3d> output;
  if (!cloud)
  {
    return output;
  }

  output.reserve(cloud->points.size());
  for (const auto& point : cloud->points)
  {
    output.push_back(ToVec3(point));
  }
  return output;
}

} // namespace

SsdExportData SsdResultAdapter::Convert(const Pcloud& source)
{
  SsdExportData output;

  output.graph.vertices.reserve(source.realVertices.rows());
  for (int row = 0; row < source.realVertices.rows(); ++row)
  {
    output.graph.vertices.emplace_back(source.realVertices(row, 0),
                                       source.realVertices(row, 1),
                                       source.realVertices(row, 2));
  }

  output.graph.edges.reserve(source.outputEdges.rows());
  for (int row = 0; row < source.outputEdges.rows(); ++row)
  {
    output.graph.edges.emplace_back(source.outputEdges(row, 0), source.outputEdges(row, 1));
  }

  output.graph.branches = source.branches;
  output.graph.branch_segment_pairs = source.branch_seg_pairs;
  for (const auto& pair : source.segments)
  {
    if (pair.second.size() < 2)
    {
      continue;
    }
    output.graph.segments[pair.first] = Eigen::Vector2i(pair.second[0], pair.second[1]);
  }

  for (const auto& pair : source.seg_clouds_scale)
  {
    output.cloud.segment_clouds[pair.first] = ConvertCloud(pair.second);
  }

  output.cloud.subspace_clouds.reserve(source.sub_space_scale.size());
  for (const auto& subspace_cloud : source.sub_space_scale)
  {
    output.cloud.subspace_clouds.push_back(ConvertCloud(subspace_cloud));
  }

  output.stats.segment_inner_distance = source.inner_dist_set;
  if (source.ori_pts_)
  {
    output.stats.original_cloud_size = static_cast<int>(source.ori_pts_->points.size());
  }

  return output;
}

} // namespace fast_planner
