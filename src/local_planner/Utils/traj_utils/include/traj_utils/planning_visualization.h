#ifndef _PLANNING_VISUALIZATION_H_
#define _PLANNING_VISUALIZATION_H_

#include <Eigen/Eigen>
#include <algorithm>
#include <iostream>
#include <map>
#include <ros/ros.h>
#include <string>
#include <utility>
#include <vector>

#include <nav_msgs/Path.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <quadrotor_msgs/PolynomialTraj.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using std::vector;
using std::string;
using std::map;
namespace fast_planner {
class PlanningVisualization {
private:
  enum TRAJECTORY_PLANNING_ID {
    GOAL = 1,
    PATH = 200,
    BSPLINE = 300,
    BSPLINE_CTRL_PT = 400,
    POLY_TRAJ = 500
  };


  /* data */
  /* visib_pub is seperated from previous ones for different info */
  ros::NodeHandle node;
  ros::Publisher traj_pub_;       // 0
  ros::Publisher topo_pub_;       // 1
  ros::Publisher predict_pub_;    // 2
  ros::Publisher visib_pub_;      // 3, visibility constraints
  ros::Publisher frontier_pub_;   // 4, frontier searching
  ros::Publisher yaw_pub_;        // 5, yaw trajectory
  ros::Publisher viewpoint_pub_;  // 6, viewpoint planning
  vector<ros::Publisher> pubs_;   //

  ros::Publisher pred_pub_;
  ros::Publisher localReg_pub_;
  ros::Publisher global_pub_;
  ros::Publisher vpg_pub_;
  ros::Publisher internal_pub_;
  ros::Publisher global_dir_pub_;
  ros::Publisher global_c_pub_;
  ros::Publisher global_n_pub_;

  ros::Publisher local_pub_;
  ros::Publisher localob_pub_;
  ros::Publisher localVP_pub_;

  ros::Publisher pcloud_pub_;
  ros::Publisher mesh_pub_;
  ros::Publisher normal_pub_;
  ros::Publisher rosa_orientation_pub_;
  ros::Publisher drosa_pub_;
  ros::Publisher le_pts_pub_;
  ros::Publisher le_lines_pub_;
  ros::Publisher rr_pts_pub_;
  ros::Publisher rr_lines_pub_;
  ros::Publisher decomp_pub_;
  ros::Publisher branch_start_end_pub_;
  ros::Publisher branch_dir_pub_;
  ros::Publisher cut_plane_pub_;
  ros::Publisher cut_pt_pub_;
  ros::Publisher sub_space_pub_;
  ros::Publisher sub_endpts_pub_;
  ros::Publisher vertex_ID_pub_;

  ros::Publisher checkPoint_pub_;
  ros::Publisher checkNeigh_pub_;
  ros::Publisher checkCPdir_pub_;
  ros::Publisher checkRP_pub_;
  ros::Publisher checkCPpts_pub_;
  ros::Publisher checkCPptsCluster_pub_;
  ros::Publisher checkBranch_pub_;
  ros::Publisher checkAdj_pub_;

  ros::Publisher optArea_pub_;

  ros::Publisher init_vps_pub_;
  ros::Publisher sub_vps_hull_pub_;
  ros::Publisher before_opt_vp_pub_;
  ros::Publisher after_opt_vp_pub_;
  ros::Publisher hcopp_viewpoints_pub_;
  ros::Publisher hcopp_occ_pub_;
  ros::Publisher hcopp_internal_pub_;
  ros::Publisher hcopp_fov_pub_;
  ros::Publisher hcopp_uncovered_pub_;
  ros::Publisher hcopp_validvp_pub_;
  ros::Publisher hcopp_correctnormal_pub_;
  ros::Publisher hcopp_sub_finalvps_pub_;
  ros::Publisher hcopp_vps_drone_pub_;
  ros::Publisher hcopp_globalseq_pub_;
  ros::Publisher hcopp_globalboundary_pub_;
  ros::Publisher hcopp_local_path_pub_;
  ros::Publisher hcopp_full_path_pub_;
  ros::Publisher fullatsp_full_path_pub_;
  ros::Publisher fullgdcpca_full_path_pub_;
  ros::Publisher pca_vec_pub_;
  ros::Publisher cylinder_pub_;
  ros::Publisher posi_traj_pub_;
  ros::Publisher pitch_traj_pub_;
  ros::Publisher yaw_traj_pub_;
  ros::Publisher jointSphere_pub_;
  ros::Publisher hcoppYaw_pub_;
  ros::Publisher pathVisible_pub_;

  ros::Publisher currentPose_pub_;
  ros::Publisher currentVoxels_pub_;

  ros::Publisher drawFoV_pub_;
  ros::Publisher drone_pub_;
  ros::Publisher traveltraj_pub_;
  ros::Publisher visible_pub_;

  nav_msgs::Path path_msg;
  std::string droneMesh;

  int last_topo_path1_num_;
  int last_topo_path2_num_;
  int last_bspline_phase1_num_;
  int last_bspline_phase2_num_;
  int last_frontier_num_;

public:
  PlanningVisualization(/* args */) {
  }
  ~PlanningVisualization() {
  }
  PlanningVisualization(ros::NodeHandle& nh);

  // new interface
  void fillBasicInfo(visualization_msgs::Marker& mk, const Eigen::Vector3d& scale,
                     const Eigen::Vector4d& color, const string& ns, const int& id, const int& shape);
  void fillGeometryInfo(visualization_msgs::Marker& mk, const vector<Eigen::Vector3d>& list);
  void fillGeometryInfo(visualization_msgs::Marker& mk, const vector<Eigen::Vector3d>& list1,
                        const vector<Eigen::Vector3d>& list2);

  void drawSpheres(const vector<Eigen::Vector3d>& list, const double& scale,
                   const Eigen::Vector4d& color, const string& ns, const int& id, const int& pub_id);
  void drawCubes(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color,
                 const string& ns, const int& id, const int& pub_id);
  void drawLines(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
                 const double& scale, const Eigen::Vector4d& color, const string& ns, const int& id,
                 const int& pub_id);
  void drawLines(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color,
                 const string& ns, const int& id, const int& pub_id);
  void drawBox(const Eigen::Vector3d& center, const Eigen::Vector3d& scale, const Eigen::Vector4d& color,
               const string& ns, const int& id, const int& pub_id);

  // Deprecated
  // draw basic shapes
  void displaySphereList(const vector<Eigen::Vector3d>& list, double resolution,
                         const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayCubeList(const vector<Eigen::Vector3d>& list, double resolution,
                       const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayLineList(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
                       double line_width, const Eigen::Vector4d& color, int id, int pub_id = 0);
  // draw a piece-wise straight line path
  void drawGeometricPath(const vector<Eigen::Vector3d>& path, double resolution,
                         const Eigen::Vector4d& color, int id = 0);

  void drawGoal(Eigen::Vector3d goal, double resolution, const Eigen::Vector4d& color, int id = 0);

  Eigen::Vector4d getColor(const double& h, double alpha = 1.0);

  typedef std::shared_ptr<PlanningVisualization> Ptr;

  // SECTION developing
  void drawVisibConstraint(const Eigen::MatrixXd& ctrl_pts, const vector<Eigen::Vector3d>& block_pts);
  void drawFrontier(const vector<vector<Eigen::Vector3d>>& frontiers);

  void publishSurface(const pcl::PointCloud<pcl::PointXYZ>& input_cloud);
  void publishVisCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud);
  void publishMesh(std::string& mesh);
  void publishSurfaceNormal(const pcl::PointCloud<pcl::PointXYZ>& input_cloud,
                            const pcl::PointCloud<pcl::Normal>& normals);
  void publishROSAOrientation(const pcl::PointCloud<pcl::PointXYZ>& input_cloud,
                              const pcl::PointCloud<pcl::Normal>& normals);
  void publish_dROSA(const pcl::PointCloud<pcl::PointXYZ>& local_region);
  void publish_lineextract_vis(Eigen::MatrixXd& skelver, Eigen::MatrixXi& skeladj);
  void publish_recenter_vis(Eigen::MatrixXd& skelver, Eigen::MatrixXi& skeladj,
                            Eigen::MatrixXd& realVertices);
  void publish_decomposition(Eigen::MatrixXd& nodes, vector<vector<int>>& branches,
                             vector<Eigen::Vector3d>& dirs,
                             vector<Eigen::Vector3d>& centroids);
  void publishCutPlane(const pcl::PointCloud<pcl::PointXYZ>& input_cloud,
                       Eigen::Vector3d& p, Eigen::Vector3d& v);
  void publishSubSpace(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& space);
  void publishSubEndpts(std::map<int, vector<Eigen::Vector3d>>& endpts);
  void publishSegViewpoints(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& seg_vps);
  void publishOccupied(pcl::PointCloud<pcl::PointXYZ>& occupied);
  void publishInternal(pcl::PointCloud<pcl::PointXYZ>& internal);
  void publishFOV(const vector<vector<Eigen::Vector3d>>& list1,
                  const vector<vector<Eigen::Vector3d>>& list2);
  void publishUncovered(pcl::PointCloud<pcl::PointXYZ>& uncovered);
  void publishRevisedNormal(const pcl::PointCloud<pcl::PointXYZ>& input_cloud,
                            const pcl::PointCloud<pcl::Normal>& normals);
  void publishFinalFOV(map<int, vector<vector<Eigen::Vector3d>>>& list1,
                       map<int, vector<vector<Eigen::Vector3d>>>& list2,
                       map<int, vector<double>>& yaws);
  void publishGlobalSeq(Eigen::Vector3d& start_, vector<Eigen::Vector3d>& sub_rep,
                        vector<int>& global_seq);
  void publishGlobalBoundary(Eigen::Vector3d& start_,
                             map<int, vector<int>>& boundary_id_,
                             map<int, vector<Eigen::VectorXd>>& sub_vps,
                             vector<int>& global_seq);
  void publishLocalPath(map<int, vector<Eigen::VectorXd>>& sub_paths_);
  void publishHCOPPPath(vector<Eigen::VectorXd>& fullpath_);
  void publishFullATSPPath(vector<Eigen::VectorXd>& fullpath_);
  void publishFullGDCPCAPath(vector<Eigen::VectorXd>& fullpath_);
  void publishPCAVec(vector<Eigen::Vector3d>& sub_center,
                     map<int, Eigen::Matrix3d>& sub_pcavec);
  void publishVPOpt(pcl::PointCloud<pcl::PointNormal>::Ptr& before_,
                    pcl::PointCloud<pcl::PointNormal>::Ptr& after_);
  void publishFitCylinder(map<int, vector<double>>& cylinder_param);
  void publishHCOPPTraj(quadrotor_msgs::PolynomialTraj& posi,
                        quadrotor_msgs::PolynomialTraj& pitch,
                        quadrotor_msgs::PolynomialTraj& yaw);
  void publishJointSphere(vector<Eigen::Vector3d>& joints, double& radius,
                          vector<vector<Eigen::Vector3d>>& InnerVps);
  void publishYawTraj(vector<Eigen::Vector3d>& waypt, vector<double>& yaw);
  void publishCurrentFoV(const vector<Eigen::Vector3d>& list1,
                         const vector<Eigen::Vector3d>& list2,
                         const double& yaw);
  void publishTravelTraj(vector<Eigen::Vector3d> path, double resolution,
                         Eigen::Vector4d color, int id);
  void publishVisiblePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr& currentCloud,
                            int id);
  void publishCheckNeigh(Eigen::Vector3d& checkPoint,
                         const pcl::PointCloud<pcl::PointXYZ>& checkNeigh,
                         Eigen::MatrixXi& edgeMat);
  void publishCheckCP(Eigen::Vector3d& CPPoint, Eigen::Vector3d& CPDir,
                      Eigen::Vector3d& checkRP,
                      const pcl::PointCloud<pcl::PointXYZ>& CPPts,
                      const pcl::PointCloud<pcl::PointXYZ>& CPPtsCluster);
  void publishUpdatesPose(pcl::PointCloud<pcl::PointXYZ>& visCloud,
                          vector<vector<Eigen::Vector3d>>& list1,
                          vector<vector<Eigen::Vector3d>>& list2,
                          vector<double>& yaws);
  void publishVpsCHull(
      std::map<int, std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>>&
          vpHull,
      vector<Eigen::Vector3d>& hamiPath);
  void publishInitVps(pcl::PointCloud<pcl::PointNormal>::Ptr& init_vps);
  void publishOptArea(vector<pcl::PointCloud<pcl::PointXYZ>>& optArea);
};
}  // namespace fast_planner
#endif