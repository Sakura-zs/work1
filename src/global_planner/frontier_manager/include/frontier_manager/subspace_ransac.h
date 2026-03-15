#ifndef _FRONTIER_SUBSPACE_RANSAC_H_
#define _FRONTIER_SUBSPACE_RANSAC_H_

#include <string>
#include <vector>

#include <Eigen/Eigen>

namespace fast_planner {

struct SubspaceFitConfig {
  bool enable = true;
  bool auto_classify = true;
  int model_points = 2000;

  double voxel_size = 0.0;
  int sor_k = 0;
  double sor_std = 2.0;

  double plane_dist = 0.01;
  int plane_iters = 3000;

  double geom_plane_ratio = 0.6;
  double geom_plane_min_ratio = 0.3;
  double geom_plane_thickness_max = 0.1;
  double geom_cyl_rstd_max = 0.12;
  double geom_cyl_mad_k = 3.0;
  double geom_cyl_inlier_ratio_min = 0.4;
  double geom_cyl_min_height_ratio = 1.0;

  bool fit_radius = true;
  double radius_mad_k = 3.0;
  double radius_min = -1.0;
  double radius_max = -1.0;
  double min_radius_inlier_ratio = 0.2;

  int min_points_plane = 30;
  int min_points_cylinder = 50;
  int min_points_box = 20;
};

struct PlaneParams {
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  double d = 0.0;
};

struct CylinderParams {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  double r = 0.0;
  double h = 0.0;
};

struct BoxParams {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d dims = Eigen::Vector3d::Zero();
  Eigen::Vector4d quat_wxyz = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
};

struct SubspaceFitResult {
  int subspace_id = -1;
  std::string shape_type = "unknown";
  int shape_idx = -1;
  bool success = false;

  double inlier_ratio = 0.0;
  double score = 0.0;

  PlaneParams plane;
  CylinderParams cylinder;
  BoxParams box;

  std::vector<Eigen::Vector3d> model_cloud;
};

struct OverallFitGateConfig {
  bool enable = true;
  double clamp_distance_m = 0.20;
  double accept_threshold_m = 0.10;
  int min_pred_points = 200;
  int min_gt_points = 200;
};

struct OverallFitGateResult {
  bool evaluated = false;
  bool accepted = false;
  double tmnd_pred_to_gt = 0.0;
  double tmnd_gt_to_pred = 0.0;
  double tmnd_bidir = 0.0;
  int pred_points = 0;
  int gt_points = 0;
};

class SubspaceRansacFitter {
public:
  static std::vector<SubspaceFitResult> FitAll(
      const std::vector<std::vector<Eigen::Vector3d>>& subspaces,
      const SubspaceFitConfig& config);

  static OverallFitGateResult EvaluateOverallHardThreshold(
      const std::vector<SubspaceFitResult>& fits,
      const std::vector<Eigen::Vector3d>& gt_points,
      const OverallFitGateConfig& config);
};

} // namespace fast_planner

#endif
