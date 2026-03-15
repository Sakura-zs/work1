#include <traj_utils/planning_visualization.h>

using std::cout;
using std::endl;
namespace fast_planner {
PlanningVisualization::PlanningVisualization(ros::NodeHandle& nh) {
  node = nh;

  traj_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/trajectory", 100);
  pubs_.push_back(traj_pub_);

  topo_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/topo_path", 100);
  pubs_.push_back(topo_pub_);

  predict_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/prediction", 100);
  pubs_.push_back(predict_pub_);

  visib_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/"
                                                          "visib_constraint",
                                                          100);
  pubs_.push_back(visib_pub_);

  frontier_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/frontier", 10000);
  pubs_.push_back(frontier_pub_);

  yaw_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/yaw", 100);
  pubs_.push_back(yaw_pub_);

  viewpoint_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/viewpoints", 1000);
  pubs_.push_back(viewpoint_pub_);

  pred_pub_ = node.advertise<sensor_msgs::PointCloud2>("/global_planning/pred_cloud", 10);
  localReg_pub_ = node.advertise<sensor_msgs::PointCloud2>("/global_planning/local_region", 10);
  global_pub_ = node.advertise<visualization_msgs::MarkerArray>("/global_planning/global_tour", 10);
  vpg_pub_ = node.advertise<visualization_msgs::MarkerArray>("/global_planning/vp_global", 1);
  internal_pub_ =  node.advertise<sensor_msgs::PointCloud2>("/sdf_map/unknown", 10);
  global_dir_pub_ = node.advertise<visualization_msgs::Marker>("/global_planning/global_dir", 10);
  global_c_pub_ = node.advertise<sensor_msgs::PointCloud2>("/global_planning/cluster", 1);
  global_n_pub_ = node.advertise<visualization_msgs::MarkerArray>("/global_planning/normals", 1);

  local_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/local_tour", 10);
  localob_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/localob_tour", 10);
  localVP_pub_ = node.advertise<visualization_msgs::MarkerArray>("/local_planning/vp_local", 10);

  pcloud_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_vis/input_cloud", 10);
  mesh_pub_ = node.advertise<visualization_msgs::Marker>("/rosa_vis/input_mesh", 1);
  normal_pub_ = node.advertise<visualization_msgs::MarkerArray>("/rosa_vis/input_normal", 10);
  rosa_orientation_pub_ = node.advertise<visualization_msgs::MarkerArray>("/rosa_vis/rosa_orientation", 10);
  drosa_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_vis/drosa_pts", 10);
  le_pts_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_vis/le_pts", 10);
  le_lines_pub_ = node.advertise<visualization_msgs::Marker>("/rosa_vis/le_lines", 10);
  rr_pts_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_vis/rr_pts", 10);
  rr_lines_pub_ = node.advertise<visualization_msgs::Marker>("/rosa_vis/rr_lines", 10);
  decomp_pub_ = node.advertise<visualization_msgs::MarkerArray>("/rosa_vis/branches", 10);
  branch_start_end_pub_ = node.advertise<visualization_msgs::MarkerArray>("/rosa_vis/branches_start_end", 10);
  branch_dir_pub_ = node.advertise<visualization_msgs::MarkerArray>("/rosa_vis/branches_dir", 10);
  cut_plane_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_vis/cut_plane", 10);
  cut_pt_pub_ = node.advertise<visualization_msgs::Marker>("/rosa_vis/cut_pt", 10);
  sub_space_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_vis/sub_space", 1);
  sub_endpts_pub_ = node.advertise<visualization_msgs::MarkerArray>("/rosa_vis/sub_endpts", 1);
  vertex_ID_pub_ = node.advertise<visualization_msgs::MarkerArray>("/rosa_vis/vertex_ID", 1);

  checkPoint_pub_ = node.advertise<visualization_msgs::Marker>("/rosa_debug/checkPoint", 1);
  checkNeigh_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_debug/checkNeigh", 1);
  checkCPdir_pub_ = node.advertise<visualization_msgs::Marker>("/rosa_debug/CPDirection", 1);
  checkRP_pub_ = node.advertise<visualization_msgs::Marker>("/rosa_debug/checkRP", 1);
  checkCPpts_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_debug/CPPoints", 1);
  checkCPptsCluster_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_debug/CPPointsCluster", 1);
  checkBranch_pub_ = node.advertise<visualization_msgs::MarkerArray>("/rosa_debug/checkBranches", 10);
  checkAdj_pub_ = node.advertise<visualization_msgs::Marker>("/rosa_debug/adj", 1);

  optArea_pub_ = node.advertise<sensor_msgs::PointCloud2>("/rosa_opt/opt_area", 1);

  init_vps_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/init_vps", 1);
  sub_vps_hull_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/vps_hull", 1);
  before_opt_vp_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/before_vps", 1);
  after_opt_vp_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/after_vps", 1);
  hcopp_viewpoints_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/seg_viewpoints", 1);
  hcopp_occ_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/occupied", 1);
  hcopp_internal_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/internal", 1);
  hcopp_fov_pub_ = node.advertise<visualization_msgs::Marker>("/hcopp/fov_set", 1);
  hcopp_uncovered_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/uncovered_area", 1);
  hcopp_validvp_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/valid_vp", 1);
  hcopp_correctnormal_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/correctNormals", 10);
  hcopp_sub_finalvps_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/sub_finalvps", 1);
  hcopp_vps_drone_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/finalvps_drones", 1);
  hcopp_globalseq_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/global_seq", 1);
  hcopp_globalboundary_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/global_boundary", 1);
  hcopp_local_path_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/local_paths", 1);
  hcopp_full_path_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/HCOPP_Path", 1);
  fullatsp_full_path_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/FullATSP_Path", 1);
  fullgdcpca_full_path_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/FullGDCPCA_Path", 1);
  pca_vec_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/PCA_Vec", 1);
  cylinder_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/fit_cylinder_", 1);
  posi_traj_pub_ = node.advertise<quadrotor_msgs::PolynomialTraj>("/fc_planner/position_traj", 1);
  pitch_traj_pub_ = node.advertise<quadrotor_msgs::PolynomialTraj>("/fc_planner/pitch_traj", 1);
  yaw_traj_pub_ = node.advertise<quadrotor_msgs::PolynomialTraj>("/fc_planner/yaw_traj", 1);
  jointSphere_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/JointSphere", 1);
  hcoppYaw_pub_ = node.advertise<visualization_msgs::MarkerArray>("/hcopp/yaw_traj_", 1);
  pathVisible_pub_ = node.advertise<sensor_msgs::PointCloud2>("/hcopp/vis_path_cloud_", 1);

  currentPose_pub_ = node.advertise<visualization_msgs::MarkerArray>("/SGVG/cur_vps_", 1);
  currentVoxels_pub_ = node.advertise<sensor_msgs::PointCloud2>("/SGVG/cur_vox_", 1);

  drawFoV_pub_ = node.advertise<visualization_msgs::Marker>("/fc_planner/cmd_fov", 10);
  drone_pub_ = node.advertise<visualization_msgs::Marker>("/fc_planner/drone", 10);
  traveltraj_pub_ = node.advertise<nav_msgs::Path>("/fc_planner/travel_traj", 1, true);
  visible_pub_ = node.advertise<sensor_msgs::PointCloud2>("/fc_planner/vis_points", 1);
  nh.param("hcopp/droneMesh", droneMesh, std::string("null"));

  last_topo_path1_num_ = 0;
  last_topo_path2_num_ = 0;
  last_bspline_phase1_num_ = 0;
  last_bspline_phase2_num_ = 0;
  last_frontier_num_ = 0;
}

void PlanningVisualization::fillBasicInfo(visualization_msgs::Marker& mk, const Eigen::Vector3d& scale,
                                          const Eigen::Vector4d& color, const string& ns, const int& id,
                                          const int& shape) {
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = id;
  mk.ns = ns;
  mk.type = shape;

  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);

  mk.scale.x = scale[0];
  mk.scale.y = scale[1];
  mk.scale.z = scale[2];
}

void PlanningVisualization::fillGeometryInfo(visualization_msgs::Marker& mk,
                                             const vector<Eigen::Vector3d>& list) {
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++) {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
}

void PlanningVisualization::fillGeometryInfo(visualization_msgs::Marker& mk,
                                             const vector<Eigen::Vector3d>& list1,
                                             const vector<Eigen::Vector3d>& list2) {
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list1.size()); ++i) {
    pt.x = list1[i](0);
    pt.y = list1[i](1);
    pt.z = list1[i](2);
    mk.points.push_back(pt);

    pt.x = list2[i](0);
    pt.y = list2[i](1);
    pt.z = list2[i](2);
    mk.points.push_back(pt);
  }
}

void PlanningVisualization::drawBox(const Eigen::Vector3d& center, const Eigen::Vector3d& scale,
                                    const Eigen::Vector4d& color, const string& ns, const int& id,
                                    const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, scale, color, ns, id, visualization_msgs::Marker::CUBE);
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  mk.pose.position.x = center[0];
  mk.pose.position.y = center[1];
  mk.pose.position.z = center[2];
  mk.action = visualization_msgs::Marker::ADD;

  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawSpheres(const vector<Eigen::Vector3d>& list, const double& scale,
                                        const Eigen::Vector4d& color, const string& ns, const int& id,
                                        const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id,
                visualization_msgs::Marker::SPHERE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  // pub new marker
  fillGeometryInfo(mk, list);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawCubes(const vector<Eigen::Vector3d>& list, const double& scale,
                                      const Eigen::Vector4d& color, const string& ns, const int& id,
                                      const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id,
                visualization_msgs::Marker::CUBE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  // pub new marker
  fillGeometryInfo(mk, list);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawLines(const vector<Eigen::Vector3d>& list1,
                                      const vector<Eigen::Vector3d>& list2, const double& scale,
                                      const Eigen::Vector4d& color, const string& ns, const int& id,
                                      const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id,
                visualization_msgs::Marker::LINE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  if (list1.size() == 0) return;

  // pub new marker
  fillGeometryInfo(mk, list1, list2);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawLines(const vector<Eigen::Vector3d>& list, const double& scale,
                                      const Eigen::Vector4d& color, const string& ns, const int& id,
                                      const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id,
                visualization_msgs::Marker::LINE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  if (list.size() == 0) return;

  // split the single list into two
  vector<Eigen::Vector3d> list1, list2;
  for (int i = 0; i < list.size() - 1; ++i) {
    list1.push_back(list[i]);
    list2.push_back(list[i + 1]);
  }

  // pub new marker
  fillGeometryInfo(mk, list1, list2);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displaySphereList(const vector<Eigen::Vector3d>& list, double resolution,
                                              const Eigen::Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::SPHERE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);

  mk.scale.x = resolution;
  mk.scale.y = resolution;
  mk.scale.z = resolution;

  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++) {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displayCubeList(const vector<Eigen::Vector3d>& list, double resolution,
                                            const Eigen::Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);

  mk.scale.x = resolution;
  mk.scale.y = resolution;
  mk.scale.z = resolution;

  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++) {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);

  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displayLineList(const vector<Eigen::Vector3d>& list1,
                                            const vector<Eigen::Vector3d>& list2, double line_width,
                                            const Eigen::Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = line_width;

  geometry_msgs::Point pt;
  for (int i = 0; i < int(list1.size()); ++i) {
    pt.x = list1[i](0);
    pt.y = list1[i](1);
    pt.z = list1[i](2);
    mk.points.push_back(pt);

    pt.x = list2[i](0);
    pt.y = list2[i](1);
    pt.z = list2[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);

  ros::Duration(0.0005).sleep();
}


void PlanningVisualization::drawGoal(Eigen::Vector3d goal, double resolution,
                                     const Eigen::Vector4d& color, int id) {
  vector<Eigen::Vector3d> goal_vec = { goal };
  displaySphereList(goal_vec, resolution, color, GOAL + id % 100);
}

void PlanningVisualization::drawGeometricPath(const vector<Eigen::Vector3d>& path, double resolution,
                                              const Eigen::Vector4d& color, int id) {
  displaySphereList(path, resolution, color, PATH + id % 100);
}


void PlanningVisualization::drawVisibConstraint(const Eigen::MatrixXd& ctrl_pts,
                                                const vector<Eigen::Vector3d>& block_pts) {
  int visible_num = ctrl_pts.rows() - block_pts.size();

  /* draw block points, their projection rays and visible pairs */
  vector<Eigen::Vector3d> pts1, pts2, pts3, pts4;
  int n = ctrl_pts.rows() - visible_num;

  for (int i = 0; i < n; ++i) {
    Eigen::Vector3d qb = block_pts[i];

    if (fabs(qb[2] + 10086) > 1e-3) {
      // compute the projection
      Eigen::Vector3d qi = ctrl_pts.row(i);
      Eigen::Vector3d qj = ctrl_pts.row(i + visible_num);
      Eigen::Vector3d dir = (qj - qi).normalized();
      Eigen::Vector3d qp = qi + dir * ((qb - qi).dot(dir));

      pts1.push_back(qb);
      pts2.push_back(qp);
      pts3.push_back(qi);
      pts4.push_back(qj);
    }
  }

  displayCubeList(pts1, 0.1, Eigen::Vector4d(1, 1, 0, 1), 0, 3);
  displaySphereList(pts4, 0.2, Eigen::Vector4d(0, 1, 0, 1), 1, 3);
  displayLineList(pts1, pts2, 0.015, Eigen::Vector4d(0, 1, 1, 1), 2, 3);
  displayLineList(pts3, pts4, 0.015, Eigen::Vector4d(0, 1, 0, 1), 3, 3);
}

void PlanningVisualization::drawFrontier(const vector<vector<Eigen::Vector3d>>& frontiers) {
  for (int i = 0; i < frontiers.size(); ++i) {
    // displayCubeList(frontiers[i], 0.1, getColor(double(i) / frontiers.size(),
    // 0.4), i, 4);
    drawCubes(frontiers[i], 0.1, getColor(double(i) / frontiers.size(), 0.8), "frontier", i, 4);
  }

  vector<Eigen::Vector3d> frontier;
  for (int i = frontiers.size(); i < last_frontier_num_; ++i) {
    // displayCubeList(frontier, 0.1, getColor(1), i, 4);
    drawCubes(frontier, 0.1, getColor(1), "frontier", i, 4);
  }
  last_frontier_num_ = frontiers.size();
}

Eigen::Vector4d PlanningVisualization::getColor(const double& h, double alpha) {
  double h1 = h;
  if (h1 < 0.0 || h1 > 1.0) {
    std::cout << "h out of range" << std::endl;
    h1 = 0.0;
  }

  double lambda;
  Eigen::Vector4d color1, color2;
  if (h1 >= -1e-4 && h1 < 1.0 / 6) {
    lambda = (h1 - 0.0) * 6;
    color1 = Eigen::Vector4d(1, 0, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 1, 1);
  } else if (h1 >= 1.0 / 6 && h1 < 2.0 / 6) {
    lambda = (h1 - 1.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 0, 1, 1);
  } else if (h1 >= 2.0 / 6 && h1 < 3.0 / 6) {
    lambda = (h1 - 2.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 1, 1);
  } else if (h1 >= 3.0 / 6 && h1 < 4.0 / 6) {
    lambda = (h1 - 3.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 0, 1);
  } else if (h1 >= 4.0 / 6 && h1 < 5.0 / 6) {
    lambda = (h1 - 4.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 1, 0, 1);
  } else if (h1 >= 5.0 / 6 && h1 <= 1.0 + 1e-4) {
    lambda = (h1 - 5.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 0, 1);
  }

  Eigen::Vector4d fcolor = (1 - lambda) * color1 + lambda * color2;
  fcolor(3) = alpha;

  return fcolor;
}
// PlanningVisualization::
}  // namespace fast_planner