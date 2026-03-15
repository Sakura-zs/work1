#include <frontier_manager/subspace_ransac.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace fast_planner {
namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

Cloud::Ptr toCloud(const std::vector<Eigen::Vector3d>& points) {
  Cloud::Ptr cloud(new Cloud);
  cloud->points.reserve(points.size());
  for (const auto& point : points) {
    cloud->points.emplace_back(point.x(), point.y(), point.z());
  }
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

std::vector<Eigen::Vector3d> toVec(const Cloud::Ptr& cloud) {
  std::vector<Eigen::Vector3d> points;
  if (!cloud) {
    return points;
  }
  points.reserve(cloud->points.size());
  for (const auto& point : cloud->points) {
    points.emplace_back(point.x, point.y, point.z);
  }
  return points;
}

Cloud::Ptr preprocessCloud(const Cloud::Ptr& input, const SubspaceFitConfig& cfg) {
  if (!input) {
    return Cloud::Ptr(new Cloud);
  }
  Cloud::Ptr out(new Cloud(*input));

  if (cfg.voxel_size > 1e-6) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    Cloud::Ptr down(new Cloud);
    vg.setInputCloud(out);
    vg.setLeafSize(cfg.voxel_size, cfg.voxel_size, cfg.voxel_size);
    vg.filter(*down);
    out = down;
  }

  if (cfg.sor_k > 0 && out->points.size() >= static_cast<size_t>(cfg.sor_k)) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    Cloud::Ptr filtered(new Cloud);
    sor.setInputCloud(out);
    sor.setMeanK(cfg.sor_k);
    sor.setStddevMulThresh(cfg.sor_std);
    sor.filter(*filtered);
    out = filtered;
  }

  out->width = out->points.size();
  out->height = 1;
  out->is_dense = true;
  return out;
}

Eigen::Vector3d normalize(const Eigen::Vector3d& vector) {
  const double norm = vector.norm();
  if (norm < 1e-12) {
    return Eigen::Vector3d::UnitX();
  }
  return vector / norm;
}

bool fitPlane(const Cloud::Ptr& cloud, const SubspaceFitConfig& cfg, PlaneParams& params, double& inlier_ratio) {
  if (!cloud || cloud->points.size() < static_cast<size_t>(cfg.min_points_plane)) {
    return false;
  }

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(cfg.plane_dist);
  seg.setMaxIterations(cfg.plane_iters);
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coeffs);

  if (inliers->indices.empty() || coeffs->values.size() < 4) {
    return false;
  }

  Eigen::Vector3d normal(coeffs->values[0], coeffs->values[1], coeffs->values[2]);
  const double n_norm = normal.norm();
  if (n_norm < 1e-9) {
    return false;
  }
  normal /= n_norm;
  params.normal = normal;
  params.d = coeffs->values[3] / n_norm;
  inlier_ratio = static_cast<double>(inliers->indices.size()) / static_cast<double>(cloud->points.size());
  return true;
}

void fitPcaAxisHeight(const std::vector<Eigen::Vector3d>& points,
                      Eigen::Vector3d& center,
                      Eigen::Vector3d& axis,
                      double& height) {
  const int n = static_cast<int>(points.size());
  center.setZero();
  for (const auto& point : points) {
    center += point;
  }
  center /= std::max(1, n);

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& point : points) {
    const Eigen::Vector3d diff = point - center;
    covariance += diff * diff.transpose();
  }
  covariance /= std::max(1, n);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  axis = normalize(solver.eigenvectors().col(2));

  std::vector<double> projection;
  projection.reserve(points.size());
  for (const auto& point : points) {
    projection.push_back((point - center).dot(axis));
  }
  std::sort(projection.begin(), projection.end());

  const auto pick_percentile = [&](double p) {
    const double idx = p * static_cast<double>(projection.size() - 1);
    const int i0 = static_cast<int>(std::floor(idx));
    const int i1 = std::min(i0 + 1, static_cast<int>(projection.size() - 1));
    const double t = idx - static_cast<double>(i0);
    return projection[i0] * (1.0 - t) + projection[i1] * t;
  };

  const double t_min = pick_percentile(0.02);
  const double t_max = pick_percentile(0.98);
  height = std::max(0.0, t_max - t_min);
  center = center + axis * (0.5 * (t_min + t_max));
}

std::vector<double> pointToAxisDistance(const std::vector<Eigen::Vector3d>& points,
                                        const Eigen::Vector3d& center,
                                        const Eigen::Vector3d& axis) {
  const Eigen::Vector3d unit_axis = normalize(axis);
  std::vector<double> distance;
  distance.reserve(points.size());
  for (const auto& point : points) {
    const Eigen::Vector3d diff = point - center;
    const double t = diff.dot(unit_axis);
    const Eigen::Vector3d radial = diff - t * unit_axis;
    distance.push_back(radial.norm());
  }
  return distance;
}

bool robustRadius(const std::vector<double>& raw_distance,
                  const SubspaceFitConfig& cfg,
                  double& radius,
                  double& inlier_ratio) {
  std::vector<double> distance;
  distance.reserve(raw_distance.size());
  for (double value : raw_distance) {
    if (cfg.radius_min > 0.0 && value < cfg.radius_min) {
      continue;
    }
    if (cfg.radius_max > 0.0 && value > cfg.radius_max) {
      continue;
    }
    distance.push_back(value);
  }
  if (distance.size() < 20) {
    return false;
  }

  auto mid_it = distance.begin() + distance.size() / 2;
  std::nth_element(distance.begin(), mid_it, distance.end());
  const double median = *mid_it;

  std::vector<double> abs_dev;
  abs_dev.reserve(distance.size());
  for (double value : distance) {
    abs_dev.push_back(std::abs(value - median));
  }
  auto mad_it = abs_dev.begin() + abs_dev.size() / 2;
  std::nth_element(abs_dev.begin(), mad_it, abs_dev.end());
  const double mad = std::max(1e-12, *mad_it);

  std::vector<double> inlier_values;
  inlier_values.reserve(distance.size());
  for (double value : distance) {
    if (std::abs(value - median) < cfg.radius_mad_k * mad) {
      inlier_values.push_back(value);
    }
  }

  inlier_ratio = static_cast<double>(inlier_values.size()) / static_cast<double>(distance.size());
  if (inlier_values.empty()) {
    return false;
  }

  auto rad_it = inlier_values.begin() + inlier_values.size() / 2;
  std::nth_element(inlier_values.begin(), rad_it, inlier_values.end());
  radius = *rad_it;
  return true;
}

bool fitCylinder(const std::vector<Eigen::Vector3d>& points,
                 const SubspaceFitConfig& cfg,
                 CylinderParams& params,
                 double& inlier_ratio,
                 double& score) {
  if (!cfg.fit_radius) {
    return false;
  }
  if (points.size() < static_cast<size_t>(cfg.min_points_cylinder)) {
    return false;
  }

  Eigen::Vector3d center, axis;
  double height = 0.0;
  fitPcaAxisHeight(points, center, axis, height);
  const auto radius_samples = pointToAxisDistance(points, center, axis);

  double radius = 0.0;
  double radius_inlier_ratio = 0.0;
  if (!robustRadius(radius_samples, cfg, radius, radius_inlier_ratio)) {
    return false;
  }

  double mean = 0.0;
  for (double value : radius_samples) {
    mean += value;
  }
  mean /= std::max<size_t>(1, radius_samples.size());

  double variance = 0.0;
  for (double value : radius_samples) {
    const double diff = value - mean;
    variance += diff * diff;
  }
  variance /= std::max<size_t>(1, radius_samples.size());
  const double stddev = std::sqrt(variance);
  const double r_std_ratio = stddev / std::max(1e-9, mean);
  const double h_ratio = height / std::max(1e-9, 2.0 * radius);

  if (r_std_ratio > cfg.geom_cyl_rstd_max && radius_inlier_ratio < cfg.geom_cyl_inlier_ratio_min) {
    return false;
  }
  if (h_ratio < cfg.geom_cyl_min_height_ratio) {
    return false;
  }
  if (radius_inlier_ratio < cfg.min_radius_inlier_ratio) {
    return false;
  }

  params.center = center;
  params.axis = normalize(axis);
  params.h = height;
  params.r = radius;
  inlier_ratio = radius_inlier_ratio;
  score = 1.0 / std::max(1e-6, stddev);
  return true;
}

Eigen::Vector4d rotationToQuatWxyz(const Eigen::Matrix3d& rotation) {
  Eigen::Quaterniond quaternion(rotation);
  quaternion.normalize();
  return Eigen::Vector4d(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
}

bool fitBox(const std::vector<Eigen::Vector3d>& points,
            const SubspaceFitConfig& cfg,
            BoxParams& params,
            double& score) {
  if (points.size() < static_cast<size_t>(cfg.min_points_box)) {
    return false;
  }

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& point : points) {
    mean += point;
  }
  mean /= std::max<size_t>(1, points.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& point : points) {
    const Eigen::Vector3d diff = point - mean;
    covariance += diff * diff.transpose();
  }
  covariance /= std::max<size_t>(1, points.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  Eigen::Matrix3d rotation = solver.eigenvectors();
  if (rotation.determinant() < 0.0) {
    rotation.col(2) = -rotation.col(2);
  }

  Eigen::Vector3d mn = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector3d mx = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
  for (const auto& point : points) {
    const Eigen::Vector3d local = rotation.transpose() * (point - mean);
    mn = mn.cwiseMin(local);
    mx = mx.cwiseMax(local);
  }

  params.dims = (mx - mn).cwiseMax(1e-6);
  const Eigen::Vector3d center_local = 0.5 * (mn + mx);
  params.center = mean + rotation * center_local;
  params.quat_wxyz = rotationToQuatWxyz(rotation);

  const double volume = params.dims.x() * params.dims.y() * params.dims.z();
  score = 1.0 / std::max(1e-6, volume);
  return true;
}

void orthonormalBasisFromAxis(const Eigen::Vector3d& axis,
                              Eigen::Vector3d& u,
                              Eigen::Vector3d& v,
                              Eigen::Vector3d& w) {
  w = normalize(axis);
  Eigen::Vector3d temp = (std::abs(w.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  u = normalize(w.cross(temp));
  v = normalize(w.cross(u));
}

std::vector<Eigen::Vector3d> samplePlane(const PlaneParams& params,
                                         const std::vector<Eigen::Vector3d>& support_points,
                                         int num_points) {
  std::vector<Eigen::Vector3d> model;
  if (support_points.size() < 10 || num_points <= 0) {
    return model;
  }

  Eigen::Vector3d u, v, w;
  orthonormalBasisFromAxis(params.normal, u, v, w);
  const Eigen::Vector3d p0 = -params.d * w;

  std::vector<double> uu;
  std::vector<double> vv;
  uu.reserve(support_points.size());
  vv.reserve(support_points.size());
  for (const auto& point : support_points) {
    const Eigen::Vector3d rel = point - p0;
    uu.push_back(rel.dot(u));
    vv.push_back(rel.dot(v));
  }

  const auto [u_min_it, u_max_it] = std::minmax_element(uu.begin(), uu.end());
  const auto [v_min_it, v_max_it] = std::minmax_element(vv.begin(), vv.end());

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> du(*u_min_it, *u_max_it);
  std::uniform_real_distribution<double> dv(*v_min_it, *v_max_it);

  model.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    model.push_back(p0 + du(gen) * u + dv(gen) * v);
  }
  return model;
}

std::vector<Eigen::Vector3d> sampleCylinder(const CylinderParams& params, int num_points) {
  std::vector<Eigen::Vector3d> model;
  if (num_points <= 0 || params.r <= 1e-6 || params.h <= 1e-6) {
    return model;
  }

  Eigen::Vector3d u, v, w;
  orthonormalBasisFromAxis(params.axis, u, v, w);

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dtheta(0.0, 2.0 * M_PI);
  std::uniform_real_distribution<double> dt(-0.5 * params.h, 0.5 * params.h);

  model.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    const double theta = dtheta(gen);
    const double t = dt(gen);
    model.push_back(params.center + t * w + params.r * (std::cos(theta) * u + std::sin(theta) * v));
  }
  return model;
}

std::vector<Eigen::Vector3d> sampleBox(const BoxParams& params, int num_points) {
  std::vector<Eigen::Vector3d> model;
  if (num_points <= 0) {
    return model;
  }

  Eigen::Quaterniond q(params.quat_wxyz[0], params.quat_wxyz[1], params.quat_wxyz[2], params.quat_wxyz[3]);
  q.normalize();
  const Eigen::Matrix3d rotation = q.toRotationMatrix();
  const Eigen::Vector3d half = 0.5 * params.dims;

  const std::array<double, 6> areas = {
      params.dims.y() * params.dims.z(), params.dims.y() * params.dims.z(),
      params.dims.x() * params.dims.z(), params.dims.x() * params.dims.z(),
      params.dims.x() * params.dims.y(), params.dims.x() * params.dims.y()};
  const double total_area = std::accumulate(areas.begin(), areas.end(), 0.0);

  std::array<int, 6> counts{};
  int assigned = 0;
  for (int i = 0; i < 6; ++i) {
    counts[i] = static_cast<int>(std::floor((areas[i] / std::max(1e-9, total_area)) * num_points));
    assigned += counts[i];
  }
  for (int i = 0; assigned < num_points; ++i, ++assigned) {
    counts[i % 6]++;
  }

  std::mt19937 gen(0);
  auto sample_face = [&](int axis, double value, double lim_a, double lim_b, int k) {
    std::uniform_real_distribution<double> da(-lim_a, lim_a);
    std::uniform_real_distribution<double> db(-lim_b, lim_b);
    for (int i = 0; i < k; ++i) {
      Eigen::Vector3d local(0.0, 0.0, 0.0);
      if (axis == 0) {
        local.x() = value;
        local.y() = da(gen);
        local.z() = db(gen);
      } else if (axis == 1) {
        local.y() = value;
        local.x() = da(gen);
        local.z() = db(gen);
      } else {
        local.z() = value;
        local.x() = da(gen);
        local.y() = db(gen);
      }
      model.push_back(params.center + rotation * local);
    }
  };

  model.reserve(num_points);
  sample_face(0, +half.x(), half.y(), half.z(), counts[0]);
  sample_face(0, -half.x(), half.y(), half.z(), counts[1]);
  sample_face(1, +half.y(), half.x(), half.z(), counts[2]);
  sample_face(1, -half.y(), half.x(), half.z(), counts[3]);
  sample_face(2, +half.z(), half.x(), half.y(), counts[4]);
  sample_face(2, -half.z(), half.x(), half.y(), counts[5]);
  return model;
}

std::string classifyShape(const Cloud::Ptr& cloud,
                          const std::vector<Eigen::Vector3d>& points,
                          const SubspaceFitConfig& cfg) {
  if (!cloud || cloud->points.empty()) {
    return "box";
  }

  PlaneParams plane;
  double plane_ratio = 0.0;
  if (fitPlane(cloud, cfg, plane, plane_ratio)) {
    if (plane_ratio >= cfg.geom_plane_ratio) {
      return "plane";
    }
    if (plane_ratio >= cfg.geom_plane_min_ratio) {
      std::vector<double> dist;
      dist.reserve(points.size());
      for (const auto& point : points) {
        dist.push_back(std::abs(plane.normal.dot(point) + plane.d));
      }
      std::sort(dist.begin(), dist.end());
      const auto at = [&](double p) {
        const double idx = p * static_cast<double>(dist.size() - 1);
        const int i0 = static_cast<int>(std::floor(idx));
        const int i1 = std::min(i0 + 1, static_cast<int>(dist.size() - 1));
        const double t = idx - static_cast<double>(i0);
        return dist[i0] * (1.0 - t) + dist[i1] * t;
      };
      const double thickness = at(0.95) - at(0.05);
      if (thickness <= cfg.geom_plane_thickness_max) {
        return "plane";
      }
    }
  }

  CylinderParams cylinder;
  double cyl_inlier = 0.0;
  double cyl_score = 0.0;
  if (fitCylinder(points, cfg, cylinder, cyl_inlier, cyl_score)) {
    return "cylinder";
  }

  return "box";
}

Cloud::Ptr mergeModelCloud(const std::vector<SubspaceFitResult>& fits) {
  Cloud::Ptr merged(new Cloud);
  size_t total = 0;
  for (const auto& fit : fits) {
    total += fit.model_cloud.size();
  }
  merged->points.reserve(total);
  for (const auto& fit : fits) {
    for (const auto& point : fit.model_cloud) {
      merged->points.emplace_back(point.x(), point.y(), point.z());
    }
  }
  merged->width = merged->points.size();
  merged->height = 1;
  merged->is_dense = true;
  return merged;
}

double hardClampedMeanNearestDistance(const Cloud::Ptr& source,
                                      const Cloud::Ptr& target,
                                      double clamp_distance_m) {
  if (!source || !target || source->points.empty() || target->points.empty()) {
    return clamp_distance_m;
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  tree.setInputCloud(target);

  std::vector<int> nn_idx(1);
  std::vector<float> nn_dist_sq(1);
  double sum = 0.0;

  for (const auto& point : source->points) {
    if (tree.nearestKSearch(point, 1, nn_idx, nn_dist_sq) <= 0) {
      sum += clamp_distance_m;
      continue;
    }
    const double dist = std::sqrt(std::max(0.0f, nn_dist_sq[0]));
    sum += std::min(dist, clamp_distance_m);
  }

  return sum / static_cast<double>(source->points.size());
}

} // namespace

std::vector<SubspaceFitResult> SubspaceRansacFitter::FitAll(
    const std::vector<std::vector<Eigen::Vector3d>>& subspaces,
    const SubspaceFitConfig& config) {
  std::vector<SubspaceFitResult> outputs;
  outputs.reserve(subspaces.size());

  for (size_t i = 0; i < subspaces.size(); ++i) {
    SubspaceFitResult result;
    result.subspace_id = static_cast<int>(i);

    if (!config.enable || subspaces[i].empty()) {
      outputs.push_back(std::move(result));
      continue;
    }

    Cloud::Ptr cloud = toCloud(subspaces[i]);
    cloud = preprocessCloud(cloud, config);
    const auto processed_points = toVec(cloud);
    if (processed_points.empty()) {
      outputs.push_back(std::move(result));
      continue;
    }

    const std::string shape = config.auto_classify ? classifyShape(cloud, processed_points, config) : "plane";

    if (shape == "plane") {
      result.shape_type = "plane";
      result.shape_idx = 0;
      double inlier = 0.0;
      if (fitPlane(cloud, config, result.plane, inlier)) {
        result.success = true;
        result.inlier_ratio = inlier;
        result.score = inlier;
        result.model_cloud = samplePlane(result.plane, processed_points, config.model_points);
      }
    } else if (shape == "cylinder") {
      result.shape_type = "cylinder";
      result.shape_idx = 1;
      double inlier = 0.0;
      double score = 0.0;
      if (fitCylinder(processed_points, config, result.cylinder, inlier, score)) {
        result.success = true;
        result.inlier_ratio = inlier;
        result.score = score;
        result.model_cloud = sampleCylinder(result.cylinder, config.model_points);
      }
    } else {
      result.shape_type = "box";
      result.shape_idx = 2;
      double score = 0.0;
      if (fitBox(processed_points, config, result.box, score)) {
        result.success = true;
        result.inlier_ratio = 1.0;
        result.score = score;
        result.model_cloud = sampleBox(result.box, config.model_points);
      }
    }

    outputs.push_back(std::move(result));
  }

  return outputs;
}

OverallFitGateResult SubspaceRansacFitter::EvaluateOverallHardThreshold(
    const std::vector<SubspaceFitResult>& fits,
    const std::vector<Eigen::Vector3d>& gt_points,
    const OverallFitGateConfig& config) {
  OverallFitGateResult output;

  if (!config.enable) {
    return output;
  }

  Cloud::Ptr pred_cloud = mergeModelCloud(fits);
  Cloud::Ptr gt_cloud = toCloud(gt_points);
  output.pred_points = static_cast<int>(pred_cloud->points.size());
  output.gt_points = static_cast<int>(gt_cloud->points.size());

  if (output.pred_points < config.min_pred_points ||
      output.gt_points < config.min_gt_points) {
    return output;
  }

  output.evaluated = true;
  output.tmnd_pred_to_gt = hardClampedMeanNearestDistance(pred_cloud, gt_cloud, config.clamp_distance_m);
  output.tmnd_gt_to_pred = hardClampedMeanNearestDistance(gt_cloud, pred_cloud, config.clamp_distance_m);
  output.tmnd_bidir = 0.5 * (output.tmnd_pred_to_gt + output.tmnd_gt_to_pred);
  output.accepted = (output.tmnd_bidir <= config.accept_threshold_m);

  return output;
}

} // namespace fast_planner
