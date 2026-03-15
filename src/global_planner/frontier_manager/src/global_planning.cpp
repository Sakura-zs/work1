/***
 * @Author: ning-zelin && zl.ning@qq.com
 * @Date: 2024-07-12 10:30:16
 * @LastEditTime: 2024-07-12 21:35:39
 * @Description:
 * @
 * @Copyright (c) 2024 by ning-zelin, All Rights Reserved.
 */
#include <frontier_manager/frontier_manager.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <rosa/rosa_main.h>
#include <rosa/ssd_adapter.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <random>
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

} // namespace



void FrontierManager::generateSupplementaryViewpoints(
    Eigen::Vector3f &center, vector<TopoNode::Ptr> &viewpoints) {
  (void)viewpoints;
  
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

  std::vector<SubspaceFitResult> subspace_fit_results;
  if (fit_cfg.enable) {
    subspace_fit_results = SubspaceRansacFitter::FitAll(result.cloud.subspace_clouds, fit_cfg);
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