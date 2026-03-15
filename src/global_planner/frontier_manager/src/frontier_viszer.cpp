/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2024-07-08 15:24:56
 * @LastEditTime: 2024-08-05 20:55:22
 * @Description:
 * @
 * @Copyright (c) 2024 by ning-zelin, All Rights Reserved.
 */
#include <frontier_manager/frontier_manager.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/String.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sstream>
typedef visualization_msgs::Marker Marker;
typedef visualization_msgs::MarkerArray MarkerArray;

void FrontierManager::publishSupplementarySsdCache() {
  if (!supplementary_ssd_cache_.valid)
    return;

  static ros::Publisher summary_pub =
      nh_.advertise<std_msgs::String>("/frontier/ssd_cache_summary", 1, true);
  static ros::Publisher subspace_cloud_pub =
      nh_.advertise<sensor_msgs::PointCloud2>("/frontier/ssd_subspace_cloud", 1, true);
  static ros::Publisher skeleton_vertices_pub =
      nh_.advertise<sensor_msgs::PointCloud2>("/frontier/ssd_skeleton_vertices", 1, true);
  static ros::Publisher skeleton_edges_pub =
      nh_.advertise<visualization_msgs::Marker>("/frontier/ssd_skeleton_edges", 1, true);
  static ros::Publisher branches_pub =
      nh_.advertise<visualization_msgs::MarkerArray>("/frontier/ssd_branches", 1, true);
    static ros::Publisher fit_model_cloud_pub =
      nh_.advertise<sensor_msgs::PointCloud2>("/frontier/ssd_subspace_fit_model_cloud", 1, true);

  std_msgs::String summary_msg;
  std::ostringstream summary;
  summary << "stamp=" << supplementary_ssd_cache_.stamp.toSec()
          << ", center=[" << supplementary_ssd_cache_.center.x() << ","
          << supplementary_ssd_cache_.center.y() << ","
          << supplementary_ssd_cache_.center.z() << "]"
          << ", input=" << supplementary_ssd_cache_.input_cloud_size
          << ", skeleton_v=" << supplementary_ssd_cache_.skeleton_vertices.size()
          << ", skeleton_e=" << supplementary_ssd_cache_.skeleton_edges.size()
          << ", branches=" << supplementary_ssd_cache_.branches.size()
          << ", subspaces=" << supplementary_ssd_cache_.subspace_clouds.size()
          << ", fitted_subspaces=" << supplementary_ssd_cache_.subspace_fit_results.size();
  summary_msg.data = summary.str();
  summary_pub.publish(summary_msg);

  pcl::PointCloud<pcl::PointXYZ> skeleton_vertices_cloud;
  skeleton_vertices_cloud.points.reserve(supplementary_ssd_cache_.skeleton_vertices.size());
  for (const auto &vertex : supplementary_ssd_cache_.skeleton_vertices) {
    skeleton_vertices_cloud.points.emplace_back(vertex.x(), vertex.y(), vertex.z());
  }
  skeleton_vertices_cloud.width = skeleton_vertices_cloud.points.size();
  skeleton_vertices_cloud.height = 1;
  skeleton_vertices_cloud.is_dense = true;
  sensor_msgs::PointCloud2 skeleton_vertices_msg;
  pcl::toROSMsg(skeleton_vertices_cloud, skeleton_vertices_msg);
  skeleton_vertices_msg.header.frame_id = "world";
  skeleton_vertices_msg.header.stamp = ros::Time::now();
  skeleton_vertices_pub.publish(skeleton_vertices_msg);

  visualization_msgs::Marker edges_marker;
  edges_marker.header.frame_id = "world";
  edges_marker.header.stamp = ros::Time::now();
  edges_marker.ns = "ssd_skeleton_edges";
  edges_marker.id = 0;
  edges_marker.type = visualization_msgs::Marker::LINE_LIST;
  edges_marker.action = visualization_msgs::Marker::ADD;
  edges_marker.pose.orientation.w = 1.0;
  edges_marker.scale.x = 0.08;
  edges_marker.color.r = 0.1;
  edges_marker.color.g = 0.8;
  edges_marker.color.b = 0.1;
  edges_marker.color.a = 1.0;
  for (const auto &edge : supplementary_ssd_cache_.skeleton_edges) {
    if (edge.x() < 0 || edge.y() < 0 ||
        edge.x() >= (int)supplementary_ssd_cache_.skeleton_vertices.size() ||
        edge.y() >= (int)supplementary_ssd_cache_.skeleton_vertices.size())
      continue;
    geometry_msgs::Point p1, p2;
    const auto &v1 = supplementary_ssd_cache_.skeleton_vertices[edge.x()];
    const auto &v2 = supplementary_ssd_cache_.skeleton_vertices[edge.y()];
    p1.x = v1.x(); p1.y = v1.y(); p1.z = v1.z();
    p2.x = v2.x(); p2.y = v2.y(); p2.z = v2.z();
    edges_marker.points.push_back(p1);
    edges_marker.points.push_back(p2);
  }
  skeleton_edges_pub.publish(edges_marker);

  visualization_msgs::MarkerArray branch_markers;
  visualization_msgs::Marker clear_marker;
  clear_marker.action = visualization_msgs::Marker::DELETEALL;
  branch_markers.markers.push_back(clear_marker);
  int branch_id = 0;
  for (const auto &branch : supplementary_ssd_cache_.branches) {
    if (branch.size() < 2)
      continue;
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.ns = "ssd_branches";
    marker.id = branch_id++;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.color.a = 1.0;
    marker.color.r = 0.3f + 0.7f * (float((marker.id * 37) % 100) / 100.0f);
    marker.color.g = 0.3f + 0.7f * (float((marker.id * 17) % 100) / 100.0f);
    marker.color.b = 0.3f + 0.7f * (float((marker.id * 53) % 100) / 100.0f);
    for (size_t i = 0; i + 1 < branch.size(); ++i) {
      int idx1 = branch[i];
      int idx2 = branch[i + 1];
      if (idx1 < 0 || idx2 < 0 ||
          idx1 >= (int)supplementary_ssd_cache_.skeleton_vertices.size() ||
          idx2 >= (int)supplementary_ssd_cache_.skeleton_vertices.size())
        continue;
      geometry_msgs::Point p1, p2;
      const auto &v1 = supplementary_ssd_cache_.skeleton_vertices[idx1];
      const auto &v2 = supplementary_ssd_cache_.skeleton_vertices[idx2];
      p1.x = v1.x(); p1.y = v1.y(); p1.z = v1.z();
      p2.x = v2.x(); p2.y = v2.y(); p2.z = v2.z();
      marker.points.push_back(p1);
      marker.points.push_back(p2);
    }
    branch_markers.markers.push_back(marker);
  }
  branches_pub.publish(branch_markers);

  pcl::PointCloud<pcl::PointXYZRGB> subspace_cloud;
  int subspace_id = 0;
  for (const auto &subspace : supplementary_ssd_cache_.subspace_clouds) {
    uint8_t red = static_cast<uint8_t>((subspace_id * 67) % 255);
    uint8_t green = static_cast<uint8_t>((subspace_id * 137) % 255);
    uint8_t blue = static_cast<uint8_t>((subspace_id * 193) % 255);
    for (const auto &point : subspace) {
      pcl::PointXYZRGB p;
      p.x = point.x();
      p.y = point.y();
      p.z = point.z();
      p.r = red;
      p.g = green;
      p.b = blue;
      subspace_cloud.points.push_back(p);
    }
    subspace_id++;
  }
  subspace_cloud.width = subspace_cloud.points.size();
  subspace_cloud.height = 1;
  subspace_cloud.is_dense = true;
  sensor_msgs::PointCloud2 subspace_msg;
  pcl::toROSMsg(subspace_cloud, subspace_msg);
  subspace_msg.header.frame_id = "world";
  subspace_msg.header.stamp = ros::Time::now();
  subspace_cloud_pub.publish(subspace_msg);

  pcl::PointCloud<pcl::PointXYZRGB> fit_model_cloud;
  for (const auto &fit : supplementary_ssd_cache_.subspace_fit_results) {
    uint8_t red = 255, green = 255, blue = 255;
    if (fit.shape_type == "plane") {
      red = 255;
      green = 80;
      blue = 80;
    } else if (fit.shape_type == "cylinder") {
      red = 80;
      green = 255;
      blue = 80;
    } else if (fit.shape_type == "box") {
      red = 80;
      green = 80;
      blue = 255;
    }

    for (const auto &point : fit.model_cloud) {
      pcl::PointXYZRGB p;
      p.x = point.x();
      p.y = point.y();
      p.z = point.z();
      p.r = red;
      p.g = green;
      p.b = blue;
      fit_model_cloud.points.push_back(p);
    }
  }
  fit_model_cloud.width = fit_model_cloud.points.size();
  fit_model_cloud.height = 1;
  fit_model_cloud.is_dense = true;
  sensor_msgs::PointCloud2 fit_model_msg;
  pcl::toROSMsg(fit_model_cloud, fit_model_msg);
  fit_model_msg.header.frame_id = "world";
  fit_model_msg.header.stamp = ros::Time::now();
  fit_model_cloud_pub.publish(fit_model_msg);
}

enum VizColor { RED = 0, ORANGE = 1, BLACK = 2, YELLOW = 3, BLUE = 4, GREEN = 5, EMERALD = 6, WHITE = 7, MAGNA = 8, PURPLE = 9 };

void inline static SetColor(const VizColor &color, const float &alpha, Marker &scan_marker) {
  std_msgs::ColorRGBA c;
  c.a = alpha;
  if (color == VizColor::RED) {
    c.r = 1.0f, c.g = c.b = 0.f;
  } else if (color == VizColor::ORANGE) {
    c.r = 1.0f, c.g = 0.45f, c.b = 0.1f;
  } else if (color == VizColor::BLACK) {
    c.r = c.g = c.b = 0.1f;
  } else if (color == VizColor::YELLOW) {
    c.r = c.g = 0.9f, c.b = 0.1;
  } else if (color == VizColor::BLUE) {
    c.b = 1.0f, c.r = 0.1f, c.g = 0.1f;
  } else if (color == VizColor::GREEN) {
    c.g = 0.9f, c.r = c.b = 0.f;
  } else if (color == VizColor::EMERALD) {
    c.g = c.b = 0.9f, c.r = 0.f;
  } else if (color == VizColor::WHITE) {
    c.r = c.g = c.b = 0.9f;
  } else if (color == VizColor::MAGNA) {
    c.r = c.b = 0.9f, c.g = 0.f;
  } else if (color == VizColor::PURPLE) {
    c.r = c.b = 0.5f, c.g = 0.f;
  }
  scan_marker.color = c;
}

void inline SetMarker(const VizColor &color, const std::string &ns, const float &scale, const float &alpha, Marker &scan_marker,
                      const float &scale_ratio) {
  scan_marker.header.frame_id = "world";
  scan_marker.header.stamp = ros::Time::now();
  scan_marker.ns = ns;
  scan_marker.action = Marker::ADD;
  scan_marker.scale.x = scan_marker.scale.y = scan_marker.scale.z = scale * scale_ratio;
  scan_marker.pose.orientation.x = 0.0;
  scan_marker.pose.orientation.y = 0.0;
  scan_marker.pose.orientation.z = 0.0;
  scan_marker.pose.orientation.w = 1.0;
  scan_marker.pose.position.x = 0.0;
  scan_marker.pose.position.y = 0.0;
  scan_marker.pose.position.z = 0.0;
  SetColor(color, alpha, scan_marker);
}

void FrontierManager::visfrtcluster() {
  if (!frtp_.view_cluster_)
    return;
  static ros::Publisher sf_cluster_pub = nh_.advertise<visualization_msgs::MarkerArray>("sf_cluster_marker", 5);

  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  marker.action = visualization_msgs::Marker::DELETEALL;
  marker_array.markers.push_back(marker);

  for (auto &sf_cluster : cluster_list_) {

    visualization_msgs::Marker aabb_marker, viewpoint_number;
    visualization_msgs::Marker best_viewpoint, vp_frt_connecter;
    if (!sf_cluster->is_reachable_ || sf_cluster->is_dormant_) {
      SetMarker(VizColor::BLACK, "aabb", 1.0, 0.5, aabb_marker, 1.0);
    } else if (!sf_cluster->is_new_cluster_) {
      SetMarker(VizColor::GREEN, "aabb", 1.0, 0.5, aabb_marker, 1.0);
    } else {
      SetMarker(VizColor::RED, "aabb", 1.0, 0.5, aabb_marker, 1.0);
    }
    SetMarker(VizColor::WHITE, "viewpoint_number", 0.5, 1.0, viewpoint_number, 1.0);
    SetMarker(VizColor::RED, "best_viewpoint", 0.5, 1.0, best_viewpoint, 1.0);
    SetMarker(VizColor::WHITE, "vp_frt_connecter", 0.05, 0.7, vp_frt_connecter, 1.0);

    aabb_marker.id = sf_cluster->id_;
    viewpoint_number.id = sf_cluster->id_;
    best_viewpoint.id = sf_cluster->id_;
    vp_frt_connecter.id = sf_cluster->id_;

    aabb_marker.type = visualization_msgs::Marker::CUBE;
    viewpoint_number.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    best_viewpoint.type = visualization_msgs::Marker::ARROW;
    vp_frt_connecter.type = visualization_msgs::Marker::LINE_STRIP;

    aabb_marker.pose.position.x = (sf_cluster->box_min_.x() + sf_cluster->box_max_.x()) / 2.0;
    aabb_marker.pose.position.y = (sf_cluster->box_min_.y() + sf_cluster->box_max_.y()) / 2.0;
    aabb_marker.pose.position.z = (sf_cluster->box_min_.z() + sf_cluster->box_max_.z()) / 2.0;
    aabb_marker.scale.x = sf_cluster->box_min_.x() - sf_cluster->box_max_.x();
    aabb_marker.scale.y = sf_cluster->box_min_.y() - sf_cluster->box_max_.y();
    aabb_marker.scale.z = sf_cluster->box_min_.z() - sf_cluster->box_max_.z();
    viewpoint_number.pose = aabb_marker.pose;
    int size = 0;
    for (auto &vp_cluster : sf_cluster->vp_clusters_) {
      size += vp_cluster.vps_.size();
    }
    viewpoint_number.text = std::to_string(size);

    if (sf_cluster->is_reachable_ && !sf_cluster->is_dormant_ && !sf_cluster->is_new_cluster_) {
      best_viewpoint.scale.x = 0.2;
      best_viewpoint.scale.y = 0.5;
      best_viewpoint.scale.z = 0.5;
      Eigen::Vector3f best_vp = sf_cluster->best_vp_;
      float yaw = sf_cluster->best_vp_yaw_;
      Eigen::Vector3f diff(cos(yaw), sin(yaw), 0);
      geometry_msgs::Point pt;
      pt.x = best_vp.x();
      pt.y = best_vp.y();
      pt.z = best_vp.z();
      best_viewpoint.points.push_back(pt);
      vp_frt_connecter.points.push_back(pt);

      pt.x = best_vp.x() + diff.x();
      pt.y = best_vp.y() + diff.y();
      pt.z = best_vp.z() + diff.z();
      best_viewpoint.points.push_back(pt);
      marker_array.markers.push_back(best_viewpoint);
      Eigen::Vector3f center = (sf_cluster->box_min_ + sf_cluster->box_max_) / 2.0;
      pt.x = center.x();
      pt.y = center.y();
      pt.z = center.z();
      vp_frt_connecter.points.push_back(pt);
      marker_array.markers.push_back(vp_frt_connecter);
    }

    marker_array.markers.push_back(viewpoint_number);
    marker_array.markers.push_back(aabb_marker);
  }

  sf_cluster_pub.publish(marker_array);
}

void FrontierManager::visfrtnorm(const std::vector<Eigen::Vector3f> &centers, const std::vector<Eigen::Vector3f> &normals) {
  static ros::Publisher norm_pub = nh_.advertise<visualization_msgs::MarkerArray>("norm_directions", 1);

  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  marker.action = visualization_msgs::Marker::DELETEALL;
  marker_array.markers.push_back(marker);
  for (size_t i = 0; i < centers.size(); ++i) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world"; // 你的坐标系名称
    marker.header.stamp = ros::Time::now();
    marker.ns = "normal_directions";
    marker.id = i; // unique id for each marker
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    // 设置箭头方向
    geometry_msgs::Point end_point;
    geometry_msgs::Point start_point;
    start_point.x = centers[i].x();
    start_point.y = centers[i].y();
    start_point.z = centers[i].z();
    end_point.x = centers[i].x() + normals[i].x();
    end_point.y = centers[i].y() + normals[i].y();
    end_point.z = centers[i].z() + normals[i].z();
    marker.points.push_back(start_point);
    marker.points.push_back(end_point);

    // 设置颜色和尺寸
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.scale.x = 0.05; // 箭头宽度
    marker.scale.y = 0.08; // 箭头长度
    marker.scale.z = 0.1;

    marker_array.markers.push_back(marker);
  }

  norm_pub.publish(marker_array);
}

void FrontierManager::viz_point(PointVector &pts2viz, string topic_name) {
  static unordered_map<string, ros::Publisher> pub_map;
  ros::Publisher occ_pub;
  if (pub_map.count(topic_name))
    occ_pub = pub_map[topic_name];
  else {
    occ_pub = nh_.advertise<sensor_msgs::PointCloud2>(topic_name, 5);
    pub_map[topic_name] = occ_pub;
  }
  pcl::PointCloud<pcl::PointXYZ> occ_cloud;
  occ_cloud.width = pts2viz.size();
  occ_cloud.height = 1;
  occ_cloud.points = pts2viz;
  sensor_msgs::PointCloud2 occ_msg;
  pcl::toROSMsg(occ_cloud, occ_msg);
  occ_msg.header.stamp = ros::Time::now();
  occ_msg.header.frame_id = "world";
  occ_pub.publish(occ_msg);
}

void FrontierManager::viz_point(vector<Eigen::Vector3f> &pts2viz, string topic_name) {

  PointVector pts;
  pts.reserve(pts2viz.size());
  for (auto &pt : pts2viz) {
    pts.emplace_back(pt.x(), pt.y(), pt.z());
  }
  viz_point(pts, topic_name);
}

void FrontierManager::viz_pocc() {
  if (!frtp_.view_frt_)
    return;
  // cout << "bucket_count: " << frtd_.label_map_.bucket_count() << endl;
  // cout << "load factor: " << frtd_.label_map_.load_factor() << endl;
  static ros::Publisher occ_pub = nh_.advertise<sensor_msgs::PointCloud2>("occ", 5);
  static ros::Publisher pocc_pub = nh_.advertise<sensor_msgs::PointCloud2>("pocc", 5);
  static ros::Publisher frt_pub = nh_.advertise<sensor_msgs::PointCloud2>("frt", 5);
  PointVector occ_pts, pocc_pts, frt_pts;
  for (auto &[bytes, label] : frtd_.label_map_) {
    // Eigen::Vector3i idx = pt_label.first;
    // Eigen::Vector3f pt =
    // (idx.cast<float>() + 0.5 * Eigen::Vector3f::Ones()) * frtp_.cell_size_ + frtp_.map_min_;
    PointType pt;
    bytes2pos(bytes, pt);
    if (label == SPARSE) {
      pocc_pts.emplace_back(pt);
    } else if (label == DENSE) {
      occ_pts.emplace_back(pt);
    } else {
      frt_pts.emplace_back(pt);
    }
  }
  pcl::PointCloud<pcl::PointXYZ> occ_cloud;
  pcl::PointCloud<pcl::PointXYZ> pocc_cloud;
  pcl::PointCloud<pcl::PointXYZ> frt_cloud;
  occ_cloud.width = occ_pts.size();
  occ_cloud.height = 1;
  occ_cloud.points = occ_pts;
  pocc_cloud.width = pocc_pts.size();
  pocc_cloud.height = 1;
  pocc_cloud.points = pocc_pts;
  frt_cloud.width = frt_pts.size();
  frt_cloud.height = 1;
  frt_cloud.points = frt_pts;

  sensor_msgs::PointCloud2 occ_msg, pocc_msg, frt_msg;

  pcl::toROSMsg(occ_cloud, occ_msg);
  pcl::toROSMsg(pocc_cloud, pocc_msg);
  pcl::toROSMsg(frt_cloud, frt_msg);
  occ_msg.header.stamp = ros::Time::now();
  pocc_msg.header.stamp = ros::Time::now();
  occ_msg.header.frame_id = "world";
  pocc_msg.header.frame_id = "world";
  frt_msg.header.frame_id = "world";

  occ_pub.publish(occ_msg);
  pocc_pub.publish(pocc_msg);
  frt_pub.publish(frt_msg);
}
