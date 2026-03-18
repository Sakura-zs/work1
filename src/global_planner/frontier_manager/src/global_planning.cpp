/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2024-07-12 10:30:16
 * @LastEditTime: 2024-07-12 21:35:39
 * @Description:
 * @
 * @Copyright (c) 2024 by ning-zelin, All Rights Reserved.
 */
#include <frontier_manager/frontier_manager.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <rosa/rosa_main.h>
#include <rosa/ssd_adapter.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

class UF {
public:
  UF(int size) {
    father.resize(size);
    rank.resize(size, 0);
    for (int i = 0; i < size; ++i) {
      father[i] = i;
    }
  }

  int find(int x) {
    if (x != father[x]) {
      father[x] = find(father[x]); // Path compression
    }
    return father[x];
  }

  void connect(int x, int y) {
    int xx = find(x), yy = find(y);
    if (xx != yy) {
      if (rank[xx] < rank[yy]) {
        father[xx] = yy;
      } else if (rank[xx] > rank[yy]) {
        father[yy] = xx;
      } else {
        father[yy] = xx;
        rank[xx]++;
      }
    }
  }

private:
  vector<int> father;
  vector<int> rank;
};

namespace {

inline long long edgeKey(int a, int b) {
  const int u = std::min(a, b);
  const int v = std::max(a, b);
  return (static_cast<long long>(u) << 32) | static_cast<unsigned int>(v);
}

static void buildAdjacency(
    int vertex_count,
    const std::vector<Eigen::Vector2i> &edges,
    std::vector<std::vector<int>> &adj,
    std::vector<int> &degree) {
  adj.assign(vertex_count, {});
  degree.assign(vertex_count, 0);
  for (const auto &edge : edges) {
    const int u = edge.x();
    const int v = edge.y();
    if (u < 0 || v < 0 || u >= vertex_count || v >= vertex_count || u == v) {
      continue;
    }
    adj[u].push_back(v);
    adj[v].push_back(u);
    degree[u]++;
    degree[v]++;
  }
}

static void sanitizeEdges(
    int vertex_count,
    std::vector<Eigen::Vector2i> &edges) {
  std::unordered_set<long long> seen;
  std::vector<Eigen::Vector2i> filtered;
  filtered.reserve(edges.size());
  for (const auto &edge : edges) {
    int u = edge.x();
    int v = edge.y();
    if (u < 0 || v < 0 || u >= vertex_count || v >= vertex_count || u == v) {
      continue;
    }
    const long long key = edgeKey(u, v);
    if (!seen.insert(key).second) {
      continue;
    }
    filtered.emplace_back(u, v);
  }
  edges.swap(filtered);
}

static void enforceGraphConnectivity(
    const std::vector<Eigen::Vector3d> &vertices,
    std::vector<Eigen::Vector2i> &edges) {
  const int n = static_cast<int>(vertices.size());
  if (n <= 1) {
    return;
  }

  sanitizeEdges(n, edges);
  UF uf(n);
  for (const auto &edge : edges) {
    uf.connect(edge.x(), edge.y());
  }

  std::unordered_set<long long> seen;
  for (const auto &edge : edges) {
    seen.insert(edgeKey(edge.x(), edge.y()));
  }

  while (true) {
    std::vector<int> root_of(n);
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < n; ++i) {
      root_of[i] = uf.find(i);
      components[root_of[i]].push_back(i);
    }
    if (components.size() <= 1) {
      break;
    }

    double best_dist_sq = std::numeric_limits<double>::max();
    int best_u = -1;
    int best_v = -1;

    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        if (root_of[i] == root_of[j]) {
          continue;
        }
        const double dist_sq = (vertices[i] - vertices[j]).squaredNorm();
        if (dist_sq < best_dist_sq) {
          best_dist_sq = dist_sq;
          best_u = i;
          best_v = j;
        }
      }
    }

    if (best_u < 0 || best_v < 0) {
      break;
    }

    const long long key = edgeKey(best_u, best_v);
    if (seen.insert(key).second) {
      edges.emplace_back(best_u, best_v);
    }
    uf.connect(best_u, best_v);
  }
}

static void enforceAcyclicGraph(
    const std::vector<Eigen::Vector3d> &vertices,
    std::vector<Eigen::Vector2i> &edges,
    bool keep_connected) {
  const int n = static_cast<int>(vertices.size());
  if (n <= 1) {
    edges.clear();
    return;
  }

  sanitizeEdges(n, edges);

  struct WeightedEdge {
    int u;
    int v;
    double dist_sq;
  };

  std::vector<WeightedEdge> weighted_edges;
  weighted_edges.reserve(edges.size());
  for (const auto &edge : edges) {
    const int u = edge.x();
    const int v = edge.y();
    weighted_edges.push_back({u, v, (vertices[u] - vertices[v]).squaredNorm()});
  }
  std::sort(weighted_edges.begin(), weighted_edges.end(),
            [](const WeightedEdge &a, const WeightedEdge &b) {
              return a.dist_sq < b.dist_sq;
            });

  UF uf(n);
  std::vector<Eigen::Vector2i> tree_edges;
  tree_edges.reserve(std::max(0, n - 1));

  for (const auto &edge : weighted_edges) {
    if (uf.find(edge.u) == uf.find(edge.v)) {
      continue;
    }
    uf.connect(edge.u, edge.v);
    tree_edges.emplace_back(edge.u, edge.v);
  }

  if (keep_connected) {
    while (true) {
      std::vector<int> root_of(n);
      int root0 = uf.find(0);
      bool all_connected = true;
      for (int i = 0; i < n; ++i) {
        root_of[i] = uf.find(i);
        if (root_of[i] != root0) {
          all_connected = false;
        }
      }
      if (all_connected) {
        break;
      }

      double best_dist_sq = std::numeric_limits<double>::max();
      int best_u = -1;
      int best_v = -1;
      for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
          if (root_of[i] == root_of[j]) {
            continue;
          }
          const double dist_sq = (vertices[i] - vertices[j]).squaredNorm();
          if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_u = i;
            best_v = j;
          }
        }
      }

      if (best_u < 0 || best_v < 0) {
        break;
      }
      uf.connect(best_u, best_v);
      tree_edges.emplace_back(best_u, best_v);
    }
  }

  edges.swap(tree_edges);
}

static void pruneSmallAngleSpurs(
    const std::vector<Eigen::Vector3d> &vertices,
    std::vector<Eigen::Vector2i> &edges,
    double min_angle_deg) {
  const int n = static_cast<int>(vertices.size());
  if (n <= 2 || edges.size() <= 1 || min_angle_deg <= 1e-3) {
    return;
  }

  const double min_angle_rad = min_angle_deg * M_PI / 180.0;
  bool changed = true;
  int guard = 0;

  while (changed && guard < 2000) {
    guard++;
    changed = false;

    std::vector<std::vector<int>> adj;
    std::vector<int> degree;
    buildAdjacency(n, edges, adj, degree);

    int remove_u = -1;
    int remove_v = -1;

    for (int v = 0; v < n; ++v) {
      if (adj[v].size() < 2) {
        continue;
      }

      double best_angle = std::numeric_limits<double>::max();
      int best_a = -1;
      int best_b = -1;

      for (size_t i = 0; i < adj[v].size(); ++i) {
        for (size_t j = i + 1; j < adj[v].size(); ++j) {
          const int a = adj[v][i];
          const int b = adj[v][j];
          const Eigen::Vector3d va = vertices[a] - vertices[v];
          const Eigen::Vector3d vb = vertices[b] - vertices[v];
          const double na = va.norm();
          const double nb = vb.norm();
          if (na < 1e-6 || nb < 1e-6) {
            continue;
          }
          double c = va.dot(vb) / (na * nb);
          c = std::max(-1.0, std::min(1.0, c));
          const double angle = std::acos(c);
          if (angle < best_angle) {
            best_angle = angle;
            best_a = a;
            best_b = b;
          }
        }
      }

      if (best_a < 0 || best_b < 0 || best_angle >= min_angle_rad) {
        continue;
      }

      int nbr_to_remove = -1;
      if (degree[best_a] == 1 && degree[best_b] > 1) {
        nbr_to_remove = best_a;
      } else if (degree[best_b] == 1 && degree[best_a] > 1) {
        nbr_to_remove = best_b;
      } else {
        const double da = (vertices[best_a] - vertices[v]).norm();
        const double db = (vertices[best_b] - vertices[v]).norm();
        nbr_to_remove = (da <= db) ? best_a : best_b;
      }

      remove_u = v;
      remove_v = nbr_to_remove;
      break;
    }

    if (remove_u < 0 || remove_v < 0) {
      break;
    }

    std::vector<Eigen::Vector2i> new_edges;
    new_edges.reserve(edges.size());
    bool removed = false;
    for (const auto &edge : edges) {
      const int u = edge.x();
      const int v = edge.y();
      if (!removed && ((u == remove_u && v == remove_v) || (u == remove_v && v == remove_u))) {
        removed = true;
        continue;
      }
      new_edges.push_back(edge);
    }

    if (removed) {
      edges.swap(new_edges);
      changed = true;
    }
  }

  sanitizeEdges(n, edges);
}

static void rebuildBranchesFromEdges(
    int vertex_count,
    const std::vector<Eigen::Vector2i> &edges,
    std::vector<std::vector<int>> &branches) {
  branches.clear();
  if (vertex_count <= 0 || edges.empty()) {
    return;
  }

  std::vector<std::vector<int>> adj;
  std::vector<int> degree;
  buildAdjacency(vertex_count, edges, adj, degree);

  std::unordered_set<long long> used;

  auto walk_path = [&](int start, int next) {
    std::vector<int> path;
    path.push_back(start);
    int prev = start;
    int cur = next;
    used.insert(edgeKey(start, next));

    while (true) {
      path.push_back(cur);
      if (adj[cur].size() != 2) {
        break;
      }
      int nxt = (adj[cur][0] == prev) ? adj[cur][1] : adj[cur][0];
      const long long k = edgeKey(cur, nxt);
      if (used.count(k)) {
        break;
      }
      used.insert(k);
      prev = cur;
      cur = nxt;
    }
    if (path.size() >= 2) {
      branches.push_back(std::move(path));
    }
  };

  for (int v = 0; v < vertex_count; ++v) {
    if (adj[v].empty() || adj[v].size() == 2) {
      continue;
    }
    for (int nb : adj[v]) {
      const long long k = edgeKey(v, nb);
      if (used.count(k)) {
        continue;
      }
      walk_path(v, nb);
    }
  }

  for (const auto &edge : edges) {
    const int u = edge.x();
    const int v = edge.y();
    const long long k = edgeKey(u, v);
    if (used.count(k)) {
      continue;
    }

    std::vector<int> cycle;
    cycle.push_back(u);
    int prev = u;
    int cur = v;
    used.insert(k);

    while (true) {
      cycle.push_back(cur);
      int nxt = -1;
      for (int nb : adj[cur]) {
        if (nb == prev) {
          continue;
        }
        if (!used.count(edgeKey(cur, nb))) {
          nxt = nb;
          break;
        }
      }
      if (nxt < 0) {
        break;
      }
      used.insert(edgeKey(cur, nxt));
      prev = cur;
      cur = nxt;
      if (cur == u) {
        cycle.push_back(cur);
        break;
      }
    }

    if (cycle.size() >= 2) {
      branches.push_back(std::move(cycle));
    }
  }
}

static void postProcessSkeletonGraph(
    fast_planner::SsdGraphData &graph,
    bool enforce_connected,
  bool enforce_acyclic,
    bool prune_small_angle,
    double min_angle_deg) {
  if (graph.vertices.empty()) {
    graph.edges.clear();
    graph.branches.clear();
    return;
  }

  sanitizeEdges(static_cast<int>(graph.vertices.size()), graph.edges);

  if (enforce_connected) {
    enforceGraphConnectivity(graph.vertices, graph.edges);
  }
  if (prune_small_angle) {
    pruneSmallAngleSpurs(graph.vertices, graph.edges, min_angle_deg);
  }
  if (enforce_acyclic) {
    enforceAcyclicGraph(graph.vertices, graph.edges, enforce_connected);
  } else if (enforce_connected) {
    enforceGraphConnectivity(graph.vertices, graph.edges);
  }

  rebuildBranchesFromEdges(static_cast<int>(graph.vertices.size()), graph.edges,
                          graph.branches);
}

struct SupplementaryVpCandidate {
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Vector3f target = Eigen::Vector3f::Zero();
  Eigen::Vector3f direction = Eigen::Vector3f::UnitX();
  int subspace_id = -1;
  float yaw = 0.0f;
  double clearance = 0.0;
};

struct VoxelKey {
  int x;
  int y;
  int z;

  bool operator==(const VoxelKey &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey &key) const {
    std::size_t seed = 0;
    seed ^= std::hash<int>()(key.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>()(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

static VoxelKey makeVoxelKey(const Eigen::Vector3f &point, double voxel_size) {
  const double safe_voxel = std::max(1e-3, voxel_size);
  const double inv = 1.0 / safe_voxel;
  return VoxelKey{static_cast<int>(std::floor(point.x() * inv)),
                  static_cast<int>(std::floor(point.y() * inv)),
                  static_cast<int>(std::floor(point.z() * inv))};
}

static Eigen::Matrix3d quatWxyzToRotation(const Eigen::Vector4d &quat_wxyz) {
  Eigen::Quaterniond quat(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]);
  if (quat.norm() < 1e-9) {
    return Eigen::Matrix3d::Identity();
  }
  quat.normalize();
  return quat.toRotationMatrix();
}

static Eigen::Vector3d normalFromFitAtPoint(const SubspaceFitResult &fit,
                                            const Eigen::Vector3d &point) {
  if (!fit.success) {
    return Eigen::Vector3d::Zero();
  }

  if (fit.shape_type == "plane") {
    const double norm = fit.plane.normal.norm();
    if (norm < 1e-9) {
      return Eigen::Vector3d::Zero();
    }
    return fit.plane.normal / norm;
  }

  if (fit.shape_type == "cylinder") {
    Eigen::Vector3d axis = fit.cylinder.axis;
    const double axis_norm = axis.norm();
    if (axis_norm < 1e-9) {
      return Eigen::Vector3d::Zero();
    }
    axis /= axis_norm;
    Eigen::Vector3d radial = point - fit.cylinder.center;
    radial -= radial.dot(axis) * axis;
    const double radial_norm = radial.norm();
    if (radial_norm < 1e-9) {
      return Eigen::Vector3d::Zero();
    }
    return radial / radial_norm;
  }

  if (fit.shape_type == "box") {
    const Eigen::Matrix3d rotation = quatWxyzToRotation(fit.box.quat_wxyz);
    const Eigen::Vector3d half = 0.5 * fit.box.dims.cwiseMax(1e-6);
    const Eigen::Vector3d local = rotation.transpose() * (point - fit.box.center);
    const Eigen::Array3d ratio = local.array().abs() / half.array();

    Eigen::Vector3d local_normal = Eigen::Vector3d::Zero();
    int axis_idx = 0;
    ratio.matrix().maxCoeff(&axis_idx);
    local_normal(axis_idx) = (local(axis_idx) >= 0.0) ? 1.0 : -1.0;

    Eigen::Vector3d world_normal = rotation * local_normal;
    const double world_norm = world_normal.norm();
    if (world_norm < 1e-9) {
      return Eigen::Vector3d::Zero();
    }
    return world_normal / world_norm;
  }

  return Eigen::Vector3d::Zero();
}

} // namespace



void FrontierManager::generateSupplementaryViewpoints(
    Eigen::Vector3f &center, vector<TopoNode::Ptr> &viewpoints) {

  // 防止短时间内重复计算同一位置
  if ((ros::Time::now() - last_ssd_time_).toSec() < 2.0) {
    return;
  }
  
  pcl::PointCloud<pcl::PointXYZ> accum_cloud_snapshot;
  {
    std::lock_guard<std::mutex> lock(accum_cloud_mtx_);
    if (!has_accum_cloud_ || accum_cloud_.points.empty()) {
      ROS_WARN_THROTTLE(1.0, "[FRONTIER] ROSA skipped: waiting for accumulated cloud topic.");
      return;
    }
    accum_cloud_snapshot = accum_cloud_;
  }

  const auto &raw_points = accum_cloud_snapshot.points;
  if (raw_points.empty()) {
    return;
  }
  
  last_ssd_time_ = ros::Time::now();

  int min_points;
  int max_points;
  int subspace_topk;
  bool use_local_crop;
  double local_radius;
  nh_.param("frontier/rosa_use_local_crop", use_local_crop, false);
  nh_.param("frontier/rosa_local_radius", local_radius,50.0);
  nh_.param("frontier/rosa_min_points", min_points, 300);
  nh_.param("frontier/rosa_max_points", max_points, 300000);
  nh_.param("frontier/rosa_subspace_topk", subspace_topk, 6);

  pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  
  if (use_local_crop) {
    // 仅使用 center 周围局部点云
    local_cloud->points.reserve(raw_points.size());
    const float radius_sq = static_cast<float>(local_radius * local_radius);
    for (const auto &pt : raw_points) {
      Eigen::Vector3f p(pt.x, pt.y, pt.z);
      if ((p - center).squaredNorm() <= radius_sq) {
        local_cloud->points.emplace_back(pt.x, pt.y, pt.z);
      }
    }
    ROS_INFO("[FRONTIER] ROSA using local crop: %zu points (radius=%.1fm)", 
             local_cloud->points.size(), local_radius);
  } else {
    // 直接使用全部累积点云
    local_cloud->points = raw_points;
    ROS_INFO("[FRONTIER] ROSA using full accumulated cloud: %zu points", 
             local_cloud->points.size());
  }

  if ((int)local_cloud->points.size() < min_points) {
    ROS_WARN("[FRONTIER] ROSA skipped: only %zu points (min=%d)", 
             local_cloud->points.size(), min_points);
    return;
  }

  if ((int)local_cloud->points.size() > max_points) {
    double leaf_min;
    double leaf_max;
    int leaf_search_steps;
    nh_.param("frontier/rosa_density_leaf_min", leaf_min, 0.03);
    nh_.param("frontier/rosa_density_leaf_max", leaf_max, 0.80);
    nh_.param("frontier/rosa_density_leaf_search_steps", leaf_search_steps, 12);

    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>(*local_cloud));
    pcl::PointCloud<pcl::PointXYZ>::Ptr best_cloud(new pcl::PointCloud<pcl::PointXYZ>(*local_cloud));
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;

    float left = static_cast<float>(std::max(1e-4, leaf_min));
    float right = static_cast<float>(std::max(leaf_max, leaf_min));
    leaf_search_steps = std::max(1, leaf_search_steps);

    for (int i = 0; i < leaf_search_steps; ++i) {
      float mid = 0.5f * (left + right);
      pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
      voxel_filter.setInputCloud(source_cloud);
      voxel_filter.setLeafSize(mid, mid, mid);
      voxel_filter.filter(*filtered);

      if (filtered->points.empty()) {
        left = mid;
        continue;
      }

      if ((int)filtered->points.size() <= max_points) {
        best_cloud = filtered;
        right = mid;
      } else {
        left = mid;
      }
    }

    local_cloud = best_cloud;

    if ((int)local_cloud->points.size() > max_points) {
      std::mt19937 random_engine(
          static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
      std::shuffle(local_cloud->points.begin(), local_cloud->points.end(), random_engine);
      local_cloud->points.resize(max_points);
      ROS_WARN("[FRONTIER] ROSA voxel result still > max_points, fallback random to %d", max_points);
    } else {
      ROS_INFO("[FRONTIER] ROSA density-aware downsample to %zu points", local_cloud->points.size());
    }
  }

  local_cloud->width = local_cloud->points.size();
  local_cloud->height = 1;
  local_cloud->is_dense = true;

  const std::string pcd_path =
      "/tmp/epic_rosa_" + std::to_string(::getpid()) + "_" +
      std::to_string(ros::Time::now().toNSec()) + ".pcd";
  if (pcl::io::savePCDFileBinary(pcd_path, *local_cloud) != 0) {
    ROS_WARN("[FRONTIER] ROSA skipped: failed to save temp pcd %s", pcd_path.c_str());
    return;
  }

  auto set_default_if_unset = [this](const std::string &key, const auto &value) {
    if (!nh_.hasParam(key)) {
      nh_.setParam(key, value);
    }
  };

  nh_.setParam("rosa_main/pcd", pcd_path);
  set_default_if_unset("rosa_main/radius", 0.1);
  set_default_if_unset("rosa_main/th_mah", 0.01);
  set_default_if_unset("rosa_main/delta", 0.01);
  set_default_if_unset("rosa_main/num_drosa", 5);
  set_default_if_unset("rosa_main/num_dcrosa", 2);
  set_default_if_unset("rosa_main/k_KNN", 6);
  set_default_if_unset("rosa_main/sample_r", 0.05);
  set_default_if_unset("rosa_main/alpha", 0.3);
  set_default_if_unset("rosa_main/pt_downsample_size", 0.02);
  set_default_if_unset("rosa_main/estimation_number", 10);
  set_default_if_unset("rosa_main/upper_bound_angle_inner_decomp", 30.0);
  set_default_if_unset("rosa_main/upper_bound_length_inner_decomp", 1.0);
  set_default_if_unset("rosa_main/Prune", false);
  set_default_if_unset("rosa_main/lower_bound_length", 0.2);
  set_default_if_unset("rosa_main/lower_bound_prune_angle", 75.0);
  set_default_if_unset("rosa_main/upper_bound_original_points_num", 50000);
  set_default_if_unset("rosa_main/Ground", false);
  set_default_if_unset("rosa_main/estimation_num", 50000);

  auto t_start = ros::Time::now();
  bool rosa_enable_vis;
  nh_.param("frontier/rosa_enable_vis", rosa_enable_vis, false);

  ROSA_main rosa;
  rosa.init(nh_);
  rosa.visFlag = rosa_enable_vis;
  rosa.main();
  auto result = SsdResultAdapter::Convert(rosa.P);

  bool enforce_connected_graph;
  bool enforce_acyclic_graph;
  bool prune_small_angle;
  double min_skeleton_angle_deg;
  nh_.param("frontier/rosa_enforce_connected_graph", enforce_connected_graph, true);
  nh_.param("frontier/rosa_enforce_acyclic_graph", enforce_acyclic_graph, true);
  nh_.param("frontier/rosa_prune_small_angle", prune_small_angle, true);
  nh_.param("frontier/rosa_min_skeleton_angle_deg", min_skeleton_angle_deg, 45.0);
  postProcessSkeletonGraph(result.graph, enforce_connected_graph,
                           enforce_acyclic_graph,
                           prune_small_angle, min_skeleton_angle_deg);

  SubspaceFitConfig fit_cfg;
  nh_.param("frontier/rosa_fit_subspace_enable", fit_cfg.enable, true);
  nh_.param("frontier/rosa_fit_subspace_auto_classify", fit_cfg.auto_classify, true);
  nh_.param("frontier/rosa_fit_subspace_model_points", fit_cfg.model_points, 2000);
  nh_.param("frontier/rosa_fit_subspace_voxel", fit_cfg.voxel_size, 0.0);
  nh_.param("frontier/rosa_fit_subspace_sor_k", fit_cfg.sor_k, 0);
  nh_.param("frontier/rosa_fit_subspace_sor_std", fit_cfg.sor_std, 2.0);
  nh_.param("frontier/rosa_fit_subspace_plane_dist", fit_cfg.plane_dist, 0.01);
  nh_.param("frontier/rosa_fit_subspace_plane_iters", fit_cfg.plane_iters, 3000);
  nh_.param("frontier/rosa_fit_subspace_geom_plane_ratio", fit_cfg.geom_plane_ratio, 0.6);
  nh_.param("frontier/rosa_fit_subspace_geom_plane_min_ratio", fit_cfg.geom_plane_min_ratio, 0.3);
  nh_.param("frontier/rosa_fit_subspace_geom_plane_thickness_max", fit_cfg.geom_plane_thickness_max, 0.1);
  nh_.param("frontier/rosa_fit_subspace_geom_cyl_rstd_max", fit_cfg.geom_cyl_rstd_max, 0.12);
  nh_.param("frontier/rosa_fit_subspace_geom_cyl_mad_k", fit_cfg.geom_cyl_mad_k, 3.0);
  nh_.param("frontier/rosa_fit_subspace_geom_cyl_inlier_ratio_min", fit_cfg.geom_cyl_inlier_ratio_min, 0.4);
  nh_.param("frontier/rosa_fit_subspace_geom_cyl_min_height_ratio", fit_cfg.geom_cyl_min_height_ratio, 1.0);
  nh_.param("frontier/rosa_fit_subspace_fit_radius", fit_cfg.fit_radius, true);
  nh_.param("frontier/rosa_fit_subspace_radius_mad_k", fit_cfg.radius_mad_k, 3.0);
  nh_.param("frontier/rosa_fit_subspace_radius_min", fit_cfg.radius_min, -1.0);
  nh_.param("frontier/rosa_fit_subspace_radius_max", fit_cfg.radius_max, -1.0);
  nh_.param("frontier/rosa_fit_subspace_min_radius_inlier_ratio", fit_cfg.min_radius_inlier_ratio, 0.2);
  nh_.param("frontier/rosa_fit_subspace_min_points_plane", fit_cfg.min_points_plane, 30);
  nh_.param("frontier/rosa_fit_subspace_min_points_cylinder", fit_cfg.min_points_cylinder, 50);
  nh_.param("frontier/rosa_fit_subspace_min_points_box", fit_cfg.min_points_box, 20);

  OverallFitGateConfig overall_gate_cfg;
  nh_.param("frontier/rosa_fit_overall_gate_enable", overall_gate_cfg.enable, true);
  nh_.param("frontier/rosa_fit_overall_gate_clamp_distance_m",
            overall_gate_cfg.clamp_distance_m, 4.0);
  nh_.param("frontier/rosa_fit_overall_gate_accept_threshold_m",
            overall_gate_cfg.accept_threshold_m, 3.0);
  nh_.param("frontier/rosa_fit_overall_gate_min_pred_points",
            overall_gate_cfg.min_pred_points, 200);
  nh_.param("frontier/rosa_fit_overall_gate_min_gt_points",
            overall_gate_cfg.min_gt_points, 200);

  bool supp_vp_enable;
  bool supp_vp_bidirectional;
  bool supp_vp_best_of_two_sides;
  bool supp_vp_enable_z_constraint;
  bool supp_vp_avoid_existing_viewpoints;
  bool supp_vp_enable_los_check;
  bool supp_vp_enable_target_cover_filter;
  bool supp_vp_enable_voxel_dedup;
  bool supp_vp_round_robin_subspace;
  bool supp_vp_enable_min_separation;
  bool supp_vp_enable_visibility_eval;
  bool supp_vp_enable_coverage_stop;
  bool supp_vp_enable_yaw_diversity;
  bool supp_vp_publish_normals;
  bool supp_vp_publish_sampled_only;
  bool supp_vp_report_filter_stats;
  int supp_vp_sample_stride;
  int supp_vp_min_subspace_points;
  int supp_vp_max_per_subspace;
  int supp_vp_max_total;
  int supp_vp_filter_upto_stage;
  int supp_vp_visibility_sample_stride;
  int supp_vp_visibility_min_gain;
  int supp_vp_coverage_min_keep;
  int supp_vp_yaw_diversity_bins;
  int supp_vp_normal_vis_sample_stride;
  int supp_vp_normal_vis_max_count;
  double supp_vp_dist;
  double supp_vp_min_occ_clearance;
  double supp_vp_safe_radius;
  double supp_vp_safe_step;
  double supp_vp_los_step;
  double supp_vp_los_min_clearance;
  double supp_vp_los_target_margin;
  double supp_vp_los_check_ratio;
  double supp_vp_dedup_voxel;
  double supp_vp_target_cover_voxel;
  double supp_vp_min_separation;
  double supp_vp_visibility_radius;
  double supp_vp_visibility_half_fov_deg;
  double supp_vp_target_coverage_ratio;
  double supp_vp_normal_vis_scale;
  double supp_vp_ground_z;
  double supp_vp_safe_height;
  const int supp_default_max_per_subspace = std::max(80, subspace_topk * 20);
  nh_.param("frontier/supp_vp_enable", supp_vp_enable, true);
  nh_.param("frontier/supp_vp_bidirectional", supp_vp_bidirectional, true);
  nh_.param("frontier/supp_vp_best_of_two_sides", supp_vp_best_of_two_sides, true);
  nh_.param("frontier/supp_vp_enable_z_constraint", supp_vp_enable_z_constraint, false);
  nh_.param("frontier/supp_vp_avoid_existing_viewpoints", supp_vp_avoid_existing_viewpoints, false);
  nh_.param("frontier/supp_vp_enable_los_check", supp_vp_enable_los_check, true);
  nh_.param("frontier/supp_vp_enable_target_cover_filter", supp_vp_enable_target_cover_filter, false);
  nh_.param("frontier/supp_vp_enable_voxel_dedup", supp_vp_enable_voxel_dedup, true);
  nh_.param("frontier/supp_vp_round_robin_subspace", supp_vp_round_robin_subspace, true);
  nh_.param("frontier/supp_vp_enable_min_separation", supp_vp_enable_min_separation, true);
  nh_.param("frontier/supp_vp_enable_visibility_eval", supp_vp_enable_visibility_eval, true);
  nh_.param("frontier/supp_vp_enable_coverage_stop", supp_vp_enable_coverage_stop, true);
  nh_.param("frontier/supp_vp_enable_yaw_diversity", supp_vp_enable_yaw_diversity, true);
  nh_.param("frontier/supp_vp_publish_normals", supp_vp_publish_normals, true);
  nh_.param("frontier/supp_vp_publish_sampled_only", supp_vp_publish_sampled_only, false);
  nh_.param("frontier/supp_vp_report_filter_stats", supp_vp_report_filter_stats, true);
  nh_.param("frontier/supp_vp_sample_stride", supp_vp_sample_stride, 1);
  nh_.param("frontier/supp_vp_min_subspace_points", supp_vp_min_subspace_points, 10);
  nh_.param("frontier/supp_vp_max_per_subspace", supp_vp_max_per_subspace, supp_default_max_per_subspace);
  nh_.param("frontier/supp_vp_max_total", supp_vp_max_total, 800);
  nh_.param("frontier/supp_vp_filter_upto_stage", supp_vp_filter_upto_stage, 7);
  nh_.param("frontier/supp_vp_visibility_sample_stride", supp_vp_visibility_sample_stride, 3);
  nh_.param("frontier/supp_vp_visibility_min_gain", supp_vp_visibility_min_gain, 1);
  nh_.param("frontier/supp_vp_coverage_min_keep", supp_vp_coverage_min_keep, 60);
  nh_.param("frontier/supp_vp_yaw_diversity_bins", supp_vp_yaw_diversity_bins, 12);
  nh_.param("frontier/supp_vp_normal_vis_sample_stride", supp_vp_normal_vis_sample_stride, 2);
  nh_.param("frontier/supp_vp_normal_vis_max_count", supp_vp_normal_vis_max_count, 3000);
  nh_.param("frontier/supp_vp_dist", supp_vp_dist, 1.5);
  nh_.param("frontier/supp_vp_min_occ_clearance", supp_vp_min_occ_clearance, 0.35);
  nh_.param("frontier/supp_vp_safe_radius", supp_vp_safe_radius, 0.25);
  nh_.param("frontier/supp_vp_safe_step", supp_vp_safe_step, 0.15);
  nh_.param("frontier/supp_vp_los_step", supp_vp_los_step, 0.15);
  nh_.param("frontier/supp_vp_los_min_clearance", supp_vp_los_min_clearance, 0.10);
  nh_.param("frontier/supp_vp_los_target_margin", supp_vp_los_target_margin, 0.25);
  nh_.param("frontier/supp_vp_los_check_ratio", supp_vp_los_check_ratio, 0.80);
  nh_.param("frontier/supp_vp_dedup_voxel", supp_vp_dedup_voxel, 0.15);
  nh_.param("frontier/supp_vp_target_cover_voxel", supp_vp_target_cover_voxel, 0.15);
  nh_.param("frontier/supp_vp_min_separation", supp_vp_min_separation, 0.10);
  nh_.param("frontier/supp_vp_visibility_radius", supp_vp_visibility_radius, 3.0);
  nh_.param("frontier/supp_vp_visibility_half_fov_deg", supp_vp_visibility_half_fov_deg, 65.0);
  nh_.param("frontier/supp_vp_target_coverage_ratio", supp_vp_target_coverage_ratio, 0.85);
  nh_.param("frontier/supp_vp_normal_vis_scale", supp_vp_normal_vis_scale, 0.6);
  nh_.param("frontier/supp_vp_ground_z", supp_vp_ground_z, 0.0);
  nh_.param("frontier/supp_vp_safe_height", supp_vp_safe_height, 1.0);

  std::vector<SubspaceFitResult> subspace_fit_results;
  if (fit_cfg.enable) {
    subspace_fit_results = SubspaceRansacFitter::FitAll(result.cloud.subspace_clouds, fit_cfg);

    std::vector<Eigen::Vector3d> gate_gt_points;
    size_t gt_total_points = 0;
    for (const auto &subspace : result.cloud.subspace_clouds) {
      gt_total_points += subspace.size();
    }
    gate_gt_points.reserve(gt_total_points);
    for (const auto &subspace : result.cloud.subspace_clouds) {
      gate_gt_points.insert(gate_gt_points.end(), subspace.begin(), subspace.end());
    }

    const auto overall_gate_result = SubspaceRansacFitter::EvaluateOverallHardThreshold(
        subspace_fit_results, gate_gt_points, overall_gate_cfg);

    if (overall_gate_cfg.enable && !supp_vp_publish_sampled_only) {
      if (!overall_gate_result.evaluated) {
        ROS_WARN("[FRONTIER] Overall fit gate skipped: pred=%d, gt=%d, min_pred=%d, min_gt=%d",
                 overall_gate_result.pred_points, overall_gate_result.gt_points,
                 overall_gate_cfg.min_pred_points, overall_gate_cfg.min_gt_points);
      } else if (!overall_gate_result.accepted) {
        ROS_WARN("[FRONTIER] Overall fit gate rejected: tmnd_bidir=%.4fm, pred_to_gt=%.4fm, gt_to_pred=%.4fm, threshold=%.4fm. Drop fitted models.",
                 overall_gate_result.tmnd_bidir, overall_gate_result.tmnd_pred_to_gt,
                 overall_gate_result.tmnd_gt_to_pred, overall_gate_cfg.accept_threshold_m);
        subspace_fit_results.clear();
      } else {
        ROS_INFO("[FRONTIER] Overall fit gate accepted: tmnd_bidir=%.4fm, pred_to_gt=%.4fm, gt_to_pred=%.4fm, threshold=%.4fm",
                 overall_gate_result.tmnd_bidir, overall_gate_result.tmnd_pred_to_gt,
                 overall_gate_result.tmnd_gt_to_pred, overall_gate_cfg.accept_threshold_m);
      }
    } else if (overall_gate_cfg.enable && supp_vp_publish_sampled_only && !overall_gate_result.accepted) {
      ROS_WARN("[FRONTIER] Overall fit gate rejected but ignored in sampled-only mode: tmnd_bidir=%.4fm, threshold=%.4fm",
               overall_gate_result.tmnd_bidir, overall_gate_cfg.accept_threshold_m);
    }
  }

  std::vector<SupplementaryVpCandidate> appended_candidates;
  std::vector<Eigen::Vector3f> normal_vis_origins;
  std::vector<Eigen::Vector3f> normal_vis_dirs;
  size_t raw_vp_count = 0;
  size_t safe_vp_count = 0;
  size_t dedup_vp_count = 0;
  size_t normal_raw_count = 0;
  std::vector<size_t> stage_removed_final(9, 0);
  bool stage_removed_valid = false;

  if (supp_vp_enable && fit_cfg.enable && !subspace_fit_results.empty()) {
    auto cubeSafetyCheck = [&](const Eigen::Vector3f &vp) {
      const double radius = std::max(0.0, supp_vp_safe_radius);
      if (radius < 1e-4) {
        return true;
      }

      const double step = std::max(0.02, supp_vp_safe_step);
      for (double dx = -radius; dx <= radius + 1e-6; dx += step) {
        for (double dy = -radius; dy <= radius + 1e-6; dy += step) {
          for (double dz = -radius; dz <= radius + 1e-6; dz += step) {
            Eigen::Vector3f probe =
                vp + Eigen::Vector3f(static_cast<float>(dx),
                                     static_cast<float>(dy),
                                     static_cast<float>(dz));
            if (!isInBox(probe)) {
              return false;
            }
            if (lidar_map_interface_->getDisToOcc(probe) < supp_vp_min_occ_clearance) {
              return false;
            }
          }
        }
      }
      return true;
    };

    auto losSafetyCheck = [&](const Eigen::Vector3f &vp,
                              const Eigen::Vector3f &target_pt) {
      if (!supp_vp_enable_los_check) {
        return true;
      }

      Eigen::Vector3f ray = target_pt - vp;
      const double ray_length = ray.norm();
      if (ray_length < 1e-4) {
        return false;
      }
      ray /= static_cast<float>(ray_length);

      const double step = std::max(0.02, supp_vp_los_step);
      const double target_margin = std::max(0.0, supp_vp_los_target_margin);
      const double los_ratio = std::max(0.05, std::min(1.0, supp_vp_los_check_ratio));
      const double max_dist = std::min(std::max(0.0, ray_length - target_margin),
                                       ray_length * los_ratio);

      for (double dist = 0.0; dist <= max_dist + 1e-6; dist += step) {
        Eigen::Vector3f probe = vp + static_cast<float>(dist) * ray;
        if (!isInBox(probe)) {
          return false;
        }
        if (lidar_map_interface_->getDisToOcc(probe) < supp_vp_los_min_clearance) {
          return false;
        }
      }
      return true;
    };

    const int sample_stride = std::max(1, supp_vp_sample_stride);
    const int normal_vis_stride = std::max(1, supp_vp_normal_vis_sample_stride);
    const int normal_vis_max_count = std::max(0, supp_vp_normal_vis_max_count);

    if (supp_vp_publish_sampled_only) {
      for (const auto &fit : subspace_fit_results) {
        if (!fit.success || fit.subspace_id < 0) {
          continue;
        }

        const auto &fit_model_points = fit.model_cloud;
        for (int point_idx = 0;
             point_idx < static_cast<int>(fit_model_points.size());
             point_idx += sample_stride) {
          const Eigen::Vector3d source_pt_d = fit_model_points[point_idx];
          const Eigen::Vector3d normal_d = normalFromFitAtPoint(fit, source_pt_d);
          if (normal_d.norm() < 1e-6) {
            continue;
          }

          if (supp_vp_publish_normals) {
            ++normal_raw_count;
            if ((normal_vis_max_count <= 0 ||
                 static_cast<int>(normal_vis_origins.size()) < normal_vis_max_count) &&
                ((normal_raw_count - 1) % static_cast<size_t>(normal_vis_stride) == 0)) {
              normal_vis_origins.push_back(source_pt_d.cast<float>());
              normal_vis_dirs.push_back(normal_d.cast<float>().normalized());
            }
          }

          const int sign_count = (supp_vp_bidirectional || supp_vp_best_of_two_sides) ? 2 : 1;
          for (int sign_idx = 0; sign_idx < sign_count; ++sign_idx) {
            const double sign = (sign_idx == 0) ? 1.0 : -1.0;
            Eigen::Vector3f source_pt = source_pt_d.cast<float>();
            Eigen::Vector3f normal = (sign * normal_d).cast<float>();
            if (normal.norm() < 1e-6f) {
              continue;
            }
            normal.normalize();

            Eigen::Vector3f vp = source_pt + static_cast<float>(supp_vp_dist) * normal;
            if (supp_vp_enable_z_constraint) {
              const float min_safe_z = static_cast<float>(supp_vp_ground_z + supp_vp_safe_height);
              if (vp.z() < min_safe_z) {
                vp.z() = min_safe_z;
              }
            }

            const Eigen::Vector3f look_vec = source_pt - vp;
            if (std::hypot(look_vec.x(), look_vec.y()) < 1e-4) {
              continue;
            }

            SupplementaryVpCandidate candidate;
            candidate.position = vp;
            candidate.target = source_pt;
            candidate.direction = look_vec.normalized();
            candidate.subspace_id = fit.subspace_id;
            candidate.yaw = std::atan2(look_vec.y(), look_vec.x());
            candidate.clearance = lidar_map_interface_->getDisToOcc(vp);
            appended_candidates.push_back(candidate);

            raw_vp_count++;
            safe_vp_count++;
          }
        }
      }

      dedup_vp_count = appended_candidates.size();
    } else {
      std::unordered_map<VoxelKey, SupplementaryVpCandidate, VoxelKeyHash>
          best_candidate_in_voxel;
      std::vector<SupplementaryVpCandidate> all_candidates_no_voxel_dedup;

      std::unordered_map<int, std::vector<Eigen::Vector3f>> visibility_targets_by_subspace;
      std::unordered_set<VoxelKey, VoxelKeyHash> all_target_voxels;
      const int visibility_sample_stride = std::max(1, supp_vp_visibility_sample_stride);
      for (const auto &fit : subspace_fit_results) {
        if (!fit.success) {
          continue;
        }
        if (fit.subspace_id < 0) {
          continue;
        }

        // Use fitted model cloud as the sampling source for supplementary viewpoints.
        const auto &fit_model_points = fit.model_cloud;
        auto &targets = visibility_targets_by_subspace[fit.subspace_id];
        for (int point_idx = 0;
             point_idx < static_cast<int>(fit_model_points.size());
             point_idx += visibility_sample_stride) {
          const Eigen::Vector3f sample_pt = fit_model_points[point_idx].cast<float>();
          targets.push_back(sample_pt);
          all_target_voxels.insert(makeVoxelKey(sample_pt, supp_vp_target_cover_voxel));
        }
        if (targets.empty() && !fit_model_points.empty()) {
          const Eigen::Vector3f sample_pt = fit_model_points.front().cast<float>();
          targets.push_back(sample_pt);
          all_target_voxels.insert(makeVoxelKey(sample_pt, supp_vp_target_cover_voxel));
        }
      }

      const int min_subspace_points = std::max(1, supp_vp_min_subspace_points);
      const int max_per_subspace = std::max(1, supp_vp_max_per_subspace);
      const int filter_upto_stage = std::max(0, std::min(8, supp_vp_filter_upto_stage));
        const bool disable_stage3_for_upto3 = (filter_upto_stage == 3);
        const bool apply_stage1_hard_safety = (filter_upto_stage >= 1);
        const bool apply_stage2_target_cover = (filter_upto_stage >= 2);
        const bool apply_stage3_subspace_cap =
          (filter_upto_stage >= 3) && !disable_stage3_for_upto3;
        const bool apply_stage4_voxel_dedup = (filter_upto_stage >= 4);
        const bool apply_stage5_order_total = (filter_upto_stage >= 5);
        const bool apply_stage6_min_separation = (filter_upto_stage >= 6);
        const bool apply_stage7_visibility_gain = (filter_upto_stage >= 7);
        const bool apply_stage8_coverage_stop = (filter_upto_stage >= 8);

      std::vector<size_t> stage_removed(9, 0);
      size_t stage4_input_count = 0;

      if (supp_vp_report_filter_stats) {
        ROS_INFO("[FRONTIER][VP_FILTER] stage_upto=%d, order: 1-hard_safety 2-target_cover 3-subspace_cap 4-voxel_dedup 5-order_max_total 6-min_separation 7-visibility_gain 8-coverage_stop",
                 filter_upto_stage);
        ROS_INFO("[FRONTIER][VP_FILTER] stage5_ordering: rr=%d, yaw_diversity=%d, yaw_bins=%d",
                 supp_vp_round_robin_subspace,
                 supp_vp_enable_yaw_diversity,
                 std::max(3, supp_vp_yaw_diversity_bins));
        if (disable_stage3_for_upto3) {
          ROS_WARN("[FRONTIER][VP_FILTER] stage_upto=3: stage3(subspace_cap) is explicitly bypassed for debugging.");
        }
      }

      for (const auto &fit : subspace_fit_results) {
      if (!fit.success) {
        continue;
      }
      if (fit.subspace_id < 0) {
        continue;
      }

      const auto &fit_model_points = fit.model_cloud;
      if (static_cast<int>(fit_model_points.size()) < min_subspace_points) {
        continue;
      }

      std::vector<SupplementaryVpCandidate> subspace_candidates;
      for (int point_idx = 0;
           point_idx < static_cast<int>(fit_model_points.size());
           point_idx += sample_stride) {
        const Eigen::Vector3d source_pt_d = fit_model_points[point_idx];
        const Eigen::Vector3d normal_d = normalFromFitAtPoint(fit, source_pt_d);
        if (normal_d.norm() < 1e-6) {
          continue;
        }

        if (supp_vp_publish_normals) {
          ++normal_raw_count;
          if ((normal_vis_max_count <= 0 ||
               static_cast<int>(normal_vis_origins.size()) < normal_vis_max_count) &&
              ((normal_raw_count - 1) % static_cast<size_t>(normal_vis_stride) == 0)) {
            normal_vis_origins.push_back(source_pt_d.cast<float>());
            normal_vis_dirs.push_back(normal_d.cast<float>().normalized());
          }
        }

        std::vector<SupplementaryVpCandidate> directional_candidates;
        const int sign_count = (supp_vp_bidirectional || supp_vp_best_of_two_sides) ? 2 : 1;
        for (int sign_idx = 0; sign_idx < sign_count; ++sign_idx) {
          const double sign = (sign_idx == 0) ? 1.0 : -1.0;
          Eigen::Vector3f source_pt = source_pt_d.cast<float>();
          Eigen::Vector3f normal = (sign * normal_d).cast<float>();
          if (normal.norm() < 1e-6f) {
            continue;
          }
          normal.normalize();

          Eigen::Vector3f vp = source_pt + static_cast<float>(supp_vp_dist) * normal;
          if (supp_vp_enable_z_constraint) {
            const float min_safe_z = static_cast<float>(supp_vp_ground_z + supp_vp_safe_height);
            if (vp.z() < min_safe_z) {
              vp.z() = min_safe_z;
            }
          }

          raw_vp_count++;
          const bool in_box = isInBox(vp);
          const double clearance = in_box ? lidar_map_interface_->getDisToOcc(vp) : -1.0;
          if (apply_stage1_hard_safety) {
            if (!in_box) {
              stage_removed[1]++;
              continue;
            }
            if (clearance < supp_vp_min_occ_clearance) {
              stage_removed[1]++;
              continue;
            }
            if (!cubeSafetyCheck(vp)) {
              stage_removed[1]++;
              continue;
            }
            if (!losSafetyCheck(vp, source_pt)) {
              stage_removed[1]++;
              continue;
            }
          }

          const Eigen::Vector3f look_vec = source_pt - vp;
          if (std::hypot(look_vec.x(), look_vec.y()) < 1e-4) {
            continue;
          }

          SupplementaryVpCandidate candidate;
          candidate.position = vp;
          candidate.target = source_pt;
          candidate.direction = look_vec.normalized();
          candidate.subspace_id = fit.subspace_id;
          candidate.yaw = std::atan2(look_vec.y(), look_vec.x());
          candidate.clearance = clearance;
          directional_candidates.push_back(candidate);
          safe_vp_count++;
        }

        if (directional_candidates.empty()) {
          continue;
        }

        if (supp_vp_bidirectional) {
          subspace_candidates.insert(subspace_candidates.end(),
                                     directional_candidates.begin(),
                                     directional_candidates.end());
        } else {
          const auto best_it = std::max_element(
              directional_candidates.begin(), directional_candidates.end(),
              [](const SupplementaryVpCandidate &a,
                 const SupplementaryVpCandidate &b) {
                return a.clearance < b.clearance;
              });
          subspace_candidates.push_back(*best_it);
        }
      }

      if (apply_stage2_target_cover && supp_vp_enable_target_cover_filter) {
        const size_t before_target_cover_count = subspace_candidates.size();
        // Coverage-first selection in each subspace: keep one best candidate for each target voxel.
        std::unordered_map<VoxelKey, SupplementaryVpCandidate, VoxelKeyHash>
            best_candidate_in_target_voxel;
        for (const auto &candidate : subspace_candidates) {
          const VoxelKey target_key =
              makeVoxelKey(candidate.target, supp_vp_target_cover_voxel);
          auto it = best_candidate_in_target_voxel.find(target_key);
          if (it == best_candidate_in_target_voxel.end() ||
              candidate.clearance > it->second.clearance) {
            best_candidate_in_target_voxel[target_key] = candidate;
          }
        }
        std::vector<SupplementaryVpCandidate> covered_candidates;
        covered_candidates.reserve(best_candidate_in_target_voxel.size());
        for (const auto &entry : best_candidate_in_target_voxel) {
          covered_candidates.push_back(entry.second);
        }
        subspace_candidates.swap(covered_candidates);
        if (before_target_cover_count > subspace_candidates.size()) {
          stage_removed[2] += before_target_cover_count - subspace_candidates.size();
        }
      }

      std::sort(subspace_candidates.begin(), subspace_candidates.end(),
                [](const SupplementaryVpCandidate &a,
                   const SupplementaryVpCandidate &b) {
                  return a.clearance > b.clearance;
                });
    //   if (apply_stage3_subspace_cap &&
    //       static_cast<int>(subspace_candidates.size()) > max_per_subspace) {
    //     stage_removed[3] +=
    //         static_cast<size_t>(subspace_candidates.size() - max_per_subspace);
    //     subspace_candidates.resize(max_per_subspace);
    //   }

      for (const auto &candidate : subspace_candidates) {
        if (apply_stage4_voxel_dedup && supp_vp_enable_voxel_dedup) {
          stage4_input_count++;
          const VoxelKey key = makeVoxelKey(candidate.position, supp_vp_dedup_voxel);
          auto it = best_candidate_in_voxel.find(key);
          if (it == best_candidate_in_voxel.end() ||
              candidate.clearance > it->second.clearance) {
            best_candidate_in_voxel[key] = candidate;
          }
        } else {
          all_candidates_no_voxel_dedup.push_back(candidate);
        }
      }
    }

    std::vector<SupplementaryVpCandidate> ranked_candidates;
    if (apply_stage4_voxel_dedup && supp_vp_enable_voxel_dedup) {
      dedup_vp_count = best_candidate_in_voxel.size();
      ranked_candidates.reserve(best_candidate_in_voxel.size());
      for (const auto &entry : best_candidate_in_voxel) {
        ranked_candidates.push_back(entry.second);
      }
      if (stage4_input_count > dedup_vp_count) {
        stage_removed[4] += stage4_input_count - dedup_vp_count;
      }
    } else {
      dedup_vp_count = all_candidates_no_voxel_dedup.size();
      ranked_candidates.swap(all_candidates_no_voxel_dedup);
    }

    const size_t ranked_before_stage5 = ranked_candidates.size();
    if (apply_stage5_order_total && supp_vp_round_robin_subspace) {
      std::unordered_map<int, std::vector<SupplementaryVpCandidate>> candidates_by_subspace;
      std::vector<int> subspace_order;
      for (const auto &candidate : ranked_candidates) {
        if (candidates_by_subspace.find(candidate.subspace_id) ==
            candidates_by_subspace.end()) {
          subspace_order.push_back(candidate.subspace_id);
        }
        candidates_by_subspace[candidate.subspace_id].push_back(candidate);
      }

      const auto clearance_desc = [](const SupplementaryVpCandidate &a,
                                     const SupplementaryVpCandidate &b) {
        return a.clearance > b.clearance;
      };

      std::sort(subspace_order.begin(), subspace_order.end());
      for (auto &entry : candidates_by_subspace) {
        auto &candidates = entry.second;
        if (supp_vp_enable_yaw_diversity &&
            apply_stage7_visibility_gain &&
            supp_vp_enable_visibility_eval) {
          const int yaw_bins = std::max(3, supp_vp_yaw_diversity_bins);
          std::vector<std::vector<SupplementaryVpCandidate>> binned_candidates(
              static_cast<size_t>(yaw_bins));

          for (const auto &candidate : candidates) {
            double yaw = candidate.yaw;
            while (yaw < 0.0) {
              yaw += 2.0 * M_PI;
            }
            while (yaw >= 2.0 * M_PI) {
              yaw -= 2.0 * M_PI;
            }
            int yaw_idx = static_cast<int>(std::floor(yaw / (2.0 * M_PI) * yaw_bins));
            yaw_idx = std::max(0, std::min(yaw_bins - 1, yaw_idx));
            binned_candidates[static_cast<size_t>(yaw_idx)].push_back(candidate);
          }

          for (auto &bin : binned_candidates) {
            std::sort(bin.begin(), bin.end(), clearance_desc);
          }

          std::vector<size_t> bin_offsets(static_cast<size_t>(yaw_bins), 0);
          std::vector<SupplementaryVpCandidate> diversified_candidates;
          diversified_candidates.reserve(candidates.size());
          bool has_bin_remaining = true;
          while (has_bin_remaining) {
            has_bin_remaining = false;
            for (int yaw_idx = 0; yaw_idx < yaw_bins; ++yaw_idx) {
              auto &bin = binned_candidates[static_cast<size_t>(yaw_idx)];
              size_t &offset = bin_offsets[static_cast<size_t>(yaw_idx)];
              if (offset >= bin.size()) {
                continue;
              }
              diversified_candidates.push_back(bin[offset]);
              ++offset;
              has_bin_remaining = true;
            }
          }

          candidates.swap(diversified_candidates);
        } else {
          std::sort(candidates.begin(), candidates.end(), clearance_desc);
        }
      }

      std::unordered_map<int, size_t> subspace_offsets;
      std::vector<SupplementaryVpCandidate> interleaved_candidates;
      interleaved_candidates.reserve(ranked_candidates.size());
      bool has_remaining = true;
      while (has_remaining) {
        has_remaining = false;
        for (const int subspace_id : subspace_order) {
          auto &candidates = candidates_by_subspace[subspace_id];
          size_t &offset = subspace_offsets[subspace_id];
          if (offset >= candidates.size()) {
            continue;
          }
          interleaved_candidates.push_back(candidates[offset]);
          offset++;
          has_remaining = true;

          // 总数上限
        //   if (supp_vp_max_total > 0 &&
        //       static_cast<int>(interleaved_candidates.size()) >= supp_vp_max_total) {
        //     has_remaining = false;
        //     break;
        //   }
        }
      }
      ranked_candidates.swap(interleaved_candidates);
    } else if (apply_stage5_order_total) {
      std::sort(ranked_candidates.begin(), ranked_candidates.end(),
                [](const SupplementaryVpCandidate &a,
                   const SupplementaryVpCandidate &b) {
                  return a.clearance > b.clearance;
                });
      if (supp_vp_max_total > 0 &&
          static_cast<int>(ranked_candidates.size()) > supp_vp_max_total) {
        ranked_candidates.resize(supp_vp_max_total);
      }
    } else {
      std::sort(ranked_candidates.begin(), ranked_candidates.end(),
                [](const SupplementaryVpCandidate &a,
                   const SupplementaryVpCandidate &b) {
                  return a.clearance > b.clearance;
                });
    }

    if (ranked_before_stage5 > ranked_candidates.size()) {
      stage_removed[5] += ranked_before_stage5 - ranked_candidates.size();
    }

    std::vector<Eigen::Vector3f> accepted_positions;
    if (supp_vp_avoid_existing_viewpoints) {
      accepted_positions.reserve(viewpoints.size() + ranked_candidates.size());
      for (const auto &existing_vp : viewpoints) {
        if (!existing_vp) {
          continue;
        }
        accepted_positions.push_back(existing_vp->center_);
      }
    } else {
      accepted_positions.reserve(ranked_candidates.size());
    }

    std::unordered_set<VoxelKey, VoxelKeyHash> covered_target_voxels;
    const double visibility_radius = std::max(0.5, supp_vp_visibility_radius);
    const double visibility_half_fov_deg =
        std::max(5.0, std::min(89.0, supp_vp_visibility_half_fov_deg));
    const double visibility_cos_threshold =
        std::cos(visibility_half_fov_deg * M_PI / 180.0);
    const int visibility_min_gain = std::max(1, supp_vp_visibility_min_gain);
    const int coverage_min_keep = std::max(0, supp_vp_coverage_min_keep);
    const double target_coverage_ratio =
        std::max(0.0, std::min(1.0, supp_vp_target_coverage_ratio));

    auto evaluateVisibilityGain = [&](const SupplementaryVpCandidate &candidate,
                                      std::vector<VoxelKey> &newly_covered) {
      newly_covered.clear();
      std::unordered_set<VoxelKey, VoxelKeyHash> local_new_keys;

      auto addTargetIfUncovered = [&](const Eigen::Vector3f &target_pt) {
        const VoxelKey key = makeVoxelKey(target_pt, supp_vp_target_cover_voxel);
        if (covered_target_voxels.find(key) == covered_target_voxels.end()) {
          local_new_keys.insert(key);
        }
      };

      if (!supp_vp_enable_visibility_eval) {
        addTargetIfUncovered(candidate.target);
      } else {
        auto vis_it = visibility_targets_by_subspace.find(candidate.subspace_id);
        if (vis_it == visibility_targets_by_subspace.end() || vis_it->second.empty()) {
          addTargetIfUncovered(candidate.target);
        } else {
          for (const auto &target_pt : vis_it->second) {
            Eigen::Vector3f ray = target_pt - candidate.position;
            const double dist = ray.norm();
            if (dist < 1e-4 || dist > visibility_radius) {
              continue;
            }
            ray /= static_cast<float>(dist);
            if (ray.dot(candidate.direction) < visibility_cos_threshold) {
              continue;
            }
            if (supp_vp_enable_los_check &&
                !losSafetyCheck(candidate.position, target_pt)) {
              continue;
            }
            addTargetIfUncovered(target_pt);
          }
        }
      }

      newly_covered.reserve(local_new_keys.size());
      for (const auto &key : local_new_keys) {
        newly_covered.push_back(key);
      }
      return static_cast<int>(newly_covered.size());
    };

    for (size_t candidate_idx = 0; candidate_idx < ranked_candidates.size();
         ++candidate_idx) {
      const auto &candidate = ranked_candidates[candidate_idx];
      bool too_close = false;
      if (apply_stage6_min_separation && supp_vp_enable_min_separation &&
          supp_vp_min_separation > 1e-4) {
        for (const auto &accepted_pos : accepted_positions) {
          if ((accepted_pos - candidate.position).norm() < supp_vp_min_separation) {
            too_close = true;
            break;
          }
        }
      }
      if (too_close) {
        stage_removed[6]++;
        continue;
      }

      std::vector<VoxelKey> newly_covered;
      const int visibility_gain = evaluateVisibilityGain(candidate, newly_covered);
      if (apply_stage7_visibility_gain && supp_vp_enable_visibility_eval &&
          visibility_gain < visibility_min_gain &&
          static_cast<int>(appended_candidates.size()) >= coverage_min_keep) {
        stage_removed[7]++;
        continue;
      }

      TopoNode::Ptr vp_node = make_shared<TopoNode>();
      vp_node->is_viewpoint_ = true;
      vp_node->center_ = candidate.position;
      vp_node->yaw_ = candidate.yaw;
      viewpoints.push_back(vp_node);
      appended_candidates.push_back(candidate);
      accepted_positions.push_back(candidate.position);

      for (const auto &key : newly_covered) {
        covered_target_voxels.insert(key);
      }

      if (apply_stage8_coverage_stop && supp_vp_enable_coverage_stop &&
          !all_target_voxels.empty() &&
          static_cast<int>(appended_candidates.size()) >= coverage_min_keep) {
        const double covered_ratio =
            static_cast<double>(covered_target_voxels.size()) /
            static_cast<double>(all_target_voxels.size());
        if (covered_ratio >= target_coverage_ratio) {
          if (candidate_idx + 1 < ranked_candidates.size()) {
            stage_removed[8] += ranked_candidates.size() - candidate_idx - 1;
          }
          break;
        }
      }
    }

      const double final_covered_ratio =
        all_target_voxels.empty()
          ? 0.0
          : static_cast<double>(covered_target_voxels.size()) /
            static_cast<double>(all_target_voxels.size());
      ROS_INFO("[FRONTIER] Supplementary VP coverage: covered_vox=%zu/%zu (%.1f%%), vis_eval=%d, cov_stop=%d, min_gain=%d",
           covered_target_voxels.size(), all_target_voxels.size(),
           final_covered_ratio * 100.0, supp_vp_enable_visibility_eval,
           supp_vp_enable_coverage_stop, visibility_min_gain);

      if (supp_vp_report_filter_stats) {
        ROS_INFO("[FRONTIER][VP_FILTER] removed_by_stage(order): hard_safety=%zu, target_cover=%zu, subspace_cap=%zu, voxel_dedup=%zu, order_max_total=%zu, min_separation=%zu, visibility_gain=%zu, coverage_stop=%zu",
                 stage_removed[1], stage_removed[2], stage_removed[3],
                 stage_removed[4], stage_removed[5], stage_removed[6],
                 stage_removed[7], stage_removed[8]);

        std::vector<std::pair<std::string, size_t>> sorted_removed = {
            {"hard_safety", stage_removed[1]},
            {"target_cover", stage_removed[2]},
            {"subspace_cap", stage_removed[3]},
            {"voxel_dedup", stage_removed[4]},
            {"order_max_total", stage_removed[5]},
            {"min_separation", stage_removed[6]},
            {"visibility_gain", stage_removed[7]},
            {"coverage_stop", stage_removed[8]}};
        std::sort(sorted_removed.begin(), sorted_removed.end(),
                  [](const std::pair<std::string, size_t> &a,
                     const std::pair<std::string, size_t> &b) {
                    return a.second > b.second;
                  });

        std::ostringstream sorted_report;
        for (size_t i = 0; i < sorted_removed.size(); ++i) {
          if (i > 0) {
            sorted_report << ", ";
          }
          sorted_report << sorted_removed[i].first << "=" << sorted_removed[i].second;
        }
        ROS_INFO("[FRONTIER][VP_FILTER] removed_by_stage(sorted): %s",
                 sorted_report.str().c_str());
      }

      stage_removed_final = stage_removed;
      stage_removed_valid = true;
    }
  }

  static ros::Publisher supplementary_vp_pose_pub =
      nh_.advertise<geometry_msgs::PoseArray>("/frontier/supplementary_viewpoints", 1, true);
  static ros::Publisher supplementary_vp_marker_pub =
      nh_.advertise<visualization_msgs::MarkerArray>("/frontier/supplementary_viewpoints_markers", 1, true);
    static ros::Publisher supplementary_normal_marker_pub =
      nh_.advertise<visualization_msgs::MarkerArray>("/frontier/supplementary_normals_markers", 1, true);

  geometry_msgs::PoseArray vp_pose_array;
  vp_pose_array.header.frame_id = "world";
  vp_pose_array.header.stamp = ros::Time::now();

  visualization_msgs::MarkerArray vp_marker_array;
  visualization_msgs::Marker clear_marker;
  clear_marker.header = vp_pose_array.header;
  clear_marker.action = visualization_msgs::Marker::DELETEALL;
  vp_marker_array.markers.push_back(clear_marker);

  visualization_msgs::Marker vp_points_marker;
  vp_points_marker.header = vp_pose_array.header;
  vp_points_marker.ns = "supplementary_vp_points";
  vp_points_marker.id = 0;
  vp_points_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  vp_points_marker.action = visualization_msgs::Marker::ADD;
  vp_points_marker.pose.orientation.w = 1.0;
  vp_points_marker.scale.x = 0.25;
  vp_points_marker.scale.y = 0.25;
  vp_points_marker.scale.z = 0.25;
  vp_points_marker.color.r = 0.95;
  vp_points_marker.color.g = 0.65;
  vp_points_marker.color.b = 0.15;
  vp_points_marker.color.a = 0.95;

  visualization_msgs::Marker vp_dirs_marker;
  vp_dirs_marker.header = vp_pose_array.header;
  vp_dirs_marker.ns = "supplementary_vp_dirs";
  vp_dirs_marker.id = 1;
  vp_dirs_marker.type = visualization_msgs::Marker::LINE_LIST;
  vp_dirs_marker.action = visualization_msgs::Marker::ADD;
  vp_dirs_marker.pose.orientation.w = 1.0;
  vp_dirs_marker.scale.x = 0.04;
  vp_dirs_marker.color.r = 0.15;
  vp_dirs_marker.color.g = 0.85;
  vp_dirs_marker.color.b = 0.95;
  vp_dirs_marker.color.a = 0.95;

  for (const auto &candidate : appended_candidates) {
    geometry_msgs::Pose pose;
    pose.position.x = candidate.position.x();
    pose.position.y = candidate.position.y();
    pose.position.z = candidate.position.z();
    Eigen::Quaternionf q(Eigen::AngleAxisf(candidate.yaw, Eigen::Vector3f::UnitZ()));
    pose.orientation.w = q.w();
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    vp_pose_array.poses.push_back(pose);

    geometry_msgs::Point p_vp;
    p_vp.x = candidate.position.x();
    p_vp.y = candidate.position.y();
    p_vp.z = candidate.position.z();
    vp_points_marker.points.push_back(p_vp);

    geometry_msgs::Point p_target;
    p_target.x = candidate.target.x();
    p_target.y = candidate.target.y();
    p_target.z = candidate.target.z();
    vp_dirs_marker.points.push_back(p_vp);
    vp_dirs_marker.points.push_back(p_target);
  }

  vp_marker_array.markers.push_back(vp_points_marker);
  vp_marker_array.markers.push_back(vp_dirs_marker);
  supplementary_vp_pose_pub.publish(vp_pose_array);
  supplementary_vp_marker_pub.publish(vp_marker_array);

  visualization_msgs::MarkerArray normal_marker_array;
  visualization_msgs::Marker normal_clear_marker;
  normal_clear_marker.header = vp_pose_array.header;
  normal_clear_marker.action = visualization_msgs::Marker::DELETEALL;
  normal_marker_array.markers.push_back(normal_clear_marker);

  visualization_msgs::Marker normal_marker;
  normal_marker.header = vp_pose_array.header;
  normal_marker.ns = "supplementary_normals";
  normal_marker.id = 0;
  normal_marker.type = visualization_msgs::Marker::LINE_LIST;
  normal_marker.action = visualization_msgs::Marker::ADD;
  normal_marker.pose.orientation.w = 1.0;
  normal_marker.scale.x = 0.025;
  normal_marker.color.r = 0.20;
  normal_marker.color.g = 0.95;
  normal_marker.color.b = 0.30;
  normal_marker.color.a = 0.95;

  if (supp_vp_publish_normals) {
    const float normal_vis_scale = static_cast<float>(std::max(0.05, supp_vp_normal_vis_scale));
    const size_t line_count = std::min(normal_vis_origins.size(), normal_vis_dirs.size());
    normal_marker.points.reserve(line_count * 2);
    for (size_t i = 0; i < line_count; ++i) {
      const Eigen::Vector3f &origin = normal_vis_origins[i];
      const Eigen::Vector3f &dir = normal_vis_dirs[i];
      geometry_msgs::Point p0;
      p0.x = origin.x();
      p0.y = origin.y();
      p0.z = origin.z();

      geometry_msgs::Point p1;
      const Eigen::Vector3f tip = origin + normal_vis_scale * dir;
      p1.x = tip.x();
      p1.y = tip.y();
      p1.z = tip.z();

      normal_marker.points.push_back(p0);
      normal_marker.points.push_back(p1);
    }
  }
  normal_marker_array.markers.push_back(normal_marker);
  supplementary_normal_marker_pub.publish(normal_marker_array);

  if (supp_vp_enable) {
    ROS_INFO("[FRONTIER] Supplementary viewpoint generation: raw=%zu, safe=%zu, dedup=%zu, appended=%zu, normals=%zu/%zu, sampled_only=%d, filter_upto=%d, bidir=%d, best2side=%d, rr=%d, target_cover=%d",
             raw_vp_count, safe_vp_count, dedup_vp_count, appended_candidates.size(),
             normal_vis_origins.size(), normal_raw_count,
       supp_vp_publish_sampled_only, supp_vp_filter_upto_stage,
             supp_vp_bidirectional, supp_vp_best_of_two_sides,
             supp_vp_round_robin_subspace, supp_vp_enable_target_cover_filter);
  }

  supplementary_ssd_cache_.valid = true;
  supplementary_ssd_cache_.stamp = ros::Time::now();
  supplementary_ssd_cache_.center = center;
  supplementary_ssd_cache_.input_cloud_size = static_cast<int>(local_cloud->points.size());
  supplementary_ssd_cache_.input_cloud.clear();
  supplementary_ssd_cache_.input_cloud.reserve(local_cloud->points.size());
  for (const auto &point : local_cloud->points) {
    supplementary_ssd_cache_.input_cloud.emplace_back(point.x, point.y, point.z);
  }
  supplementary_ssd_cache_.skeleton_vertices = result.graph.vertices;
  supplementary_ssd_cache_.skeleton_edges = result.graph.edges;
  supplementary_ssd_cache_.branches = result.graph.branches;
  supplementary_ssd_cache_.segment_clouds = result.cloud.segment_clouds;
  supplementary_ssd_cache_.subspace_clouds = result.cloud.subspace_clouds;
  supplementary_ssd_cache_.subspace_fit_results = std::move(subspace_fit_results);
  publishSupplementarySsdCache();

  std::remove(pcd_path.c_str());
  ROS_INFO("[FRONTIER] ROSA cache updated. input=%d, skeleton_v=%zu, edges=%zu, branches=%zu, subspace=%zu, cost=%.2f ms",
           (int)local_cloud->points.size(), result.graph.vertices.size(), result.graph.edges.size(),
           result.graph.branches.size(), result.cloud.subspace_clouds.size(),
           (ros::Time::now() - t_start).toSec() * 1000.0);

  if (supp_vp_enable && supp_vp_report_filter_stats) {
    ROS_INFO("[FRONTIER][VP_FILTER] final_stage_removed: hard_safety=%zu, target_cover=%zu, subspace_cap=%zu, voxel_dedup=%zu, order_max_total=%zu, min_separation=%zu, visibility_gain=%zu, coverage_stop=%zu",
             stage_removed_final[1], stage_removed_final[2], stage_removed_final[3],
             stage_removed_final[4], stage_removed_final[5], stage_removed_final[6],
             stage_removed_final[7], stage_removed_final[8]);
    if (!stage_removed_valid) {
      ROS_INFO("[FRONTIER][VP_FILTER] final_stage_removed is zero-filled because filtering stages were not executed (sampled_only=%d).",
               supp_vp_publish_sampled_only);
    }
  }
}


void FrontierManager::generateTSPViewpoints(Eigen::Vector3f&center,  vector<TopoNode::Ptr> &viewpoints) {

  unordered_set<ClusterInfo::Ptr> revp_clusters_set; // (re)-generate viewpoints clusters
  vector<float> distance_odom2cluster;
  vector<ClusterInfo::Ptr> old_clusters_within_consideration;
  for (auto &cluster : cluster_list_) {
    if (cluster->is_dormant_ || !cluster->is_reachable_)
      continue;
    if (revp_clusters_set.count(cluster))
      continue;
    old_clusters_within_consideration.push_back(cluster);
    // float distance =
    // (center- cluster->center_).norm() + fabs(graph_->odom_node_->center_.z() - cluster->center_.z()) * 0.5;
    float distance = graph_->estimateRoughDistance(cluster->center_, cluster->odom_id_);
    distance_odom2cluster.push_back(distance);
  }

  vector<int> idx;
  for (int i = 0; i < distance_odom2cluster.size(); i++) {
    idx.push_back(i);
  }

  sort(idx.begin(), idx.end(), [&](int a, int b) { return distance_odom2cluster[a] < distance_odom2cluster[b]; });

  int consider_range = min(vpp_.local_tsp_size_, (int)idx.size());
  // cout << "old_clusters_within_consideration num: " << consider_range << endl;
  for (int i = 0; i < consider_range; i++) {
    old_clusters_within_consideration[idx[i]]->is_new_cluster_ = false;
    revp_clusters_set.insert(old_clusters_within_consideration[idx[i]]);
  }
  // 附近的+新生成的
  vector<ClusterInfo::Ptr> revp_clusters_vec; // revp: regenerate viewpoint
  revp_clusters_vec.insert(revp_clusters_vec.end(), revp_clusters_set.begin(), revp_clusters_set.end());
  ros::Time t1 = ros::Time::now();
  omp_set_num_threads(4);
  // clang-format off
  #pragma omp parallel for
  // clang-format on
  for (auto &cluster : revp_clusters_vec) {
    initClusterViewpoints(cluster);
  }
  ros::Time t2 = ros::Time::now();
  // cout << "init cluster viewpoint cost: " << (t2 - t1).toSec() * 1000 << "ms" << endl;

  PointVector vp_centers;
  for (auto &cls : revp_clusters_vec) {
    for (auto &vpc : cls->vp_clusters_) {
      vp_centers.emplace_back(vpc.center_.x(), vpc.center_.y(), vpc.center_.z());
    }
  }
  viz_point(vp_centers, "viewpoint_centers");

  removeUnreachableViewpoints(revp_clusters_vec);
  vector<ClusterInfo::Ptr> clusters_can_be_searched_;
  for (auto &cluster : revp_clusters_vec) {
    if (cluster->is_reachable_)
      clusters_can_be_searched_.push_back(cluster);
  }

  ros::Time t3 = ros::Time::now();
  // cout << "remove unreachable cluster cost: " << (t3 - t2).toSec() * 1000 << "ms" << endl;
  // cout << "revp cluster size: " << revp_clusters_vec.size() << endl;
  // cout << "reab cluster size: " << clusters_can_be_searched_.size() << endl;
  // updateHalfSpaces(clusters_can_be_searched_);
  vector<ClusterInfo::Ptr> tsp_clusters;
  mutex mtx;
  unordered_set<int> cluster2remove;
  omp_set_num_threads(6);
  // clang-format off
  #pragma omp parallel for
  // clang-format on
  for (int i = 0; i < clusters_can_be_searched_.size(); i++) {
    auto cluster = clusters_can_be_searched_[i];
    selectBestViewpoint(cluster);
    if (!cluster->is_reachable_)
      continue;
    mtx.lock();
    tsp_clusters.push_back(cluster);
    if (cluster->is_dormant_) {
      cluster2remove.insert(i);
    }
    mtx.unlock();
  }
  // 飞到但看不到，说明odom漂了，这篇工作不处理，直接跳过
  cluster_list_.remove_if([&](ClusterInfo::Ptr cluster) {
    bool remove = cluster2remove.count(cluster->id_);
    if (remove) {
      for (auto &cell : cluster->cells_) {
        Eigen::Vector3i idx;
        pos2idx(cell, idx);
        ByteArrayRaw bytes;
        idx2bytes(idx, bytes);
        frtd_.label_map_[bytes] = DENSE;
      }
    }
    return remove;
  });
  ros::Time t4 = ros::Time::now();
  // cout << "select best viewpoint cost: " << (t4 - t3).toSec() * 1000 << "ms" << endl;
  // 重新topK
  vector<float> distance2odom2;
  vector<int> idx2;
  for (int i = 0; i < tsp_clusters.size(); i++) {
    float distance = tsp_clusters[i]->distance_;
    distance2odom2.push_back(distance);
    idx2.push_back(i);
  }
  sort(idx2.begin(), idx2.end(), [&](int a, int b) { return distance2odom2[a] < distance2odom2[b]; });
  float mean_distance = accumulate(distance2odom2.begin(), distance2odom2.end(), 0.0) / distance2odom2.size();
  viewpoints.clear();
  for (int i = 0; i < (int)idx2.size(); i++) {
    // 剔除异常值
    // if (i > (int)(idx2.size() / 2.0) && distance2odom2[idx2[i]] > mean_distance * 5.0)
    //   break;
    TopoNode::Ptr vp_node = make_shared<TopoNode>();
    vp_node->is_viewpoint_ = true;
    vp_node->center_ = tsp_clusters[idx2[i]]->best_vp_;
    vp_node->yaw_ = tsp_clusters[idx2[i]]->best_vp_yaw_;
    viewpoints.push_back(vp_node);
  }

  ROS_INFO("vp cluster cost: %fms  ,remove unreachable cost: %fms, select vp cost: %fms", (t2 - t1).toSec() * 1000, (t3 - t2).toSec() * 1000,
           (t4 - t3).toSec() * 1000);
}