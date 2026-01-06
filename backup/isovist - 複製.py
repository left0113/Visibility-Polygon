import compas.datastructures as cd
import compas.geometry as cg
import compas_cgal as cgal
from compas_viewer import Viewer
import pathlib
#import skeletor as sk
import trimesh as tm
#import pymesh
import numpy as np
from shapely.geometry import Polygon, LineString, Point as ShPoint
from shapely.ops import unary_union

# === Extract boundary edges&lines ===
def find_boundary_edges(mesh):
    boundary_edges = []
    for u, v in mesh.edges():
        faces = mesh.edge_faces((u, v))
        clean_faces = [f for f in faces if f is not None]
        if len(clean_faces) <= 1:
            boundary_edges.append((u, v))
    return boundary_edges

def find_boundary_lines(mesh):
    boundary_lines = []
    for u, v in find_boundary_edges(mesh):
        a = mesh.vertex_coordinates(u)
        b = mesh.vertex_coordinates(v)
        boundary_lines.append(cg.Line(cg.Point(*a), cg.Point(*b)))
    return boundary_lines

def find_boundary_vertices(mesh):
    boundary_vertices = set()
    for u, v in mesh.edges():
        faces = mesh.edge_faces((u, v))
        clean_faces = [f for f in faces if f is not None]
        if len(clean_faces) <= 1:
            boundary_vertices.add(u)
            boundary_vertices.add(v)    
    return list(boundary_vertices)

def compute_isovist_trimesh(tri_mesh, origin, n_rays=720, max_dist=10.0):
    """
    tri_mesh : trimesh.Trimesh
    origin   : (x,y,z) iterable - 發射點
    n_rays   : int - 射線數量（解析度）
    max_dist : float - 若無相交，射線延伸至此距離
    回傳： compas.geometry.Polyline （沿角度順序）
    """
    origin = np.asarray(origin, dtype=float)
    # 產生角度與方向（水平平面）
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    origins = np.tile(origin, (n_rays, 1))

    # 使用 trimesh.ray 查找所有交點
    locations, index_ray, index_tri = tri_mesh.ray.intersects_location(origins, dirs)
    # 為每條射線挑最近的交點
    pts = np.empty((n_rays, 3), dtype=float)
    pts[:] = np.nan
    if len(locations) > 0:
        for i in range(n_rays):
            mask = (index_ray == i)
            if np.any(mask):
                hits = locations[mask]
                d = np.linalg.norm(hits - origin, axis=1)
                pts[i] = hits[np.argmin(d)]
            else:
                pts[i] = origin + dirs[i] * max_dist
    else:
        # 沒有任何相交，全部延申
        for i in range(n_rays):
            pts[i] = origin + dirs[i] * max_dist

    # optional: 移除重複/NaN 或收斂短距離點（可視需求調整）
    pts_list = [cg.Point(*p) for p in pts if not np.any(np.isnan(p))]

    # 建 Polyline（若要閉合可 append 第一點）
    poly = cg.Polyline(pts_list)
    return poly

def compute_isovist_compas(mesh, origin, n_rays=360, max_dist=1000.0, z_index=None):
    """
    使用 COMPAS mesh 與 Shapely 計算可見多邊形（2D 射線投射）
    
    mesh : compas.datastructures.Mesh
    origin : (x, y, z) 或 (x, y) - 發射點座標
    n_rays : int - 射線數量
    max_dist : float - 射線最大距離
    z_index : float or None - 若指定，只用該 Z 高度的邊；若 None，取最接近的邊
    回傳 : compas.geometry.Polyline
    """
    origin = np.asarray(origin, dtype=float)
    if len(origin) == 2:
        origin = np.append(origin, 0)
    
    # 從 mesh 提取 2D 邊界（投影到 XY 平面）
    boundary_edges = find_boundary_edges(mesh)
    boundary_segments = []
    
    for u, v in boundary_edges:
        a = np.asarray(mesh.vertex_coordinates(u), dtype=float)
        b = np.asarray(mesh.vertex_coordinates(v), dtype=float)
        # 只用 XY 座標
        boundary_segments.append(((a[0], a[1]), (b[0], b[1])))
    
    # 用 Shapely 建立邊界多邊形或 LineString 集合
    lines = [LineString([seg[0], seg[1]]) for seg in boundary_segments]
    boundary = unary_union(lines)
    
    print(f"[isovist] origin (XY): ({origin[0]}, {origin[1]})")
    print(f"[isovist] boundary type: {boundary.geom_type}, n_rays: {n_rays}")
    
    # 產生射線
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    pts = []
    
    for angle in angles:
        # 射線方向
        dx = np.cos(angle)
        dy = np.sin(angle)
        # 遠點（沿射線方向延伸 max_dist）
        far_pt = (origin[0] + dx * max_dist, origin[1] + dy * max_dist)
        ray = LineString([origin[:2], far_pt])
        
        # 求交點
        inter = ray.intersection(boundary)
        
        if inter.is_empty:
            # 無相交，用射線終點
            pts.append(cg.Point(far_pt[0], far_pt[1], origin[2]))
        elif inter.geom_type == "Point":
            # 單一交點
            pts.append(cg.Point(inter.x, inter.y, origin[2]))
        elif inter.geom_type == "LineString":
            # 線段，取距離最近的端點或中點
            closest = min(
                [inter.coords[0], inter.coords[-1]],
                key=lambda p: (p[0] - origin[0])**2 + (p[1] - origin[1])**2
            )
            pts.append(cg.Point(closest[0], closest[1], origin[2]))
        elif inter.geom_type == "MultiPoint":
            # 多個點，取最近的
            closest = min(
                [(pt.x, pt.y) for pt in inter.geoms],
                key=lambda p: (p[0] - origin[0])**2 + (p[1] - origin[1])**2
            )
            pts.append(cg.Point(closest[0], closest[1], origin[2]))
    
    print(f"[isovist] final points: {len(pts)}")
    
    if len(pts) < 3:
        print("[isovist] WARNING: Less than 3 points")
        return None
    
    poly = cg.Polyline(pts)
    return poly


# === Load Meshes ===
filepath_road = pathlib.Path(__file__).parent / "Roads.stl"
filepath_block = pathlib.Path(__file__).parent / "Block.stl"

RoadMesh = cd.Mesh.from_stl(filepath_road)
BlockMesh = cd.Mesh.from_stl(filepath_block)

# === Find Boundary Lines ===
BlockLines = find_boundary_lines(BlockMesh)
RoadLines = find_boundary_lines(RoadMesh)

# Convert COMPAS mesh to trimesh 
road_trimesh = tm.Trimesh([BlockMesh.vertex_coordinates(k) for k in BlockMesh.vertices()], [BlockMesh.face_vertices(f) for f in BlockMesh.faces()], process=False)

Vis = compute_isovist_trimesh(road_trimesh, origin=(0,0,0), n_rays=36, max_dist=10.0)
Vis2 = compute_isovist_compas(RoadMesh, (0, 0, 0), n_rays=360, max_dist=100.0)

"""Skeleton = sk.skeletonize.by_wavefront(road_trimesh)
# === Convert skeleton to visualizable geometry ===
# skeletor 回傳 skeleton object，通常包含 vertices 和 edges
# Extract skeleton vertices and edges
skel_vertices = Skeleton.vertices
skel_edges = Skeleton.edges
# Convert skeleton edges to Line objects
skeleton_lines = []
for u, v in skel_edges:
    a = skel_vertices[u]
    b = skel_vertices[v]
    line = cg.Line(cg.Point(*a), cg.Point(*b))
    skeleton_lines.append(line)"""


viewer = Viewer()
#viewer.scene.add(RoadMesh)

# Add boundary lines
for line in RoadLines:
    viewer.scene.add(line, linecolor=(1.0, 0.0, 0.0))
for line in BlockLines:
    viewer.scene.add(line, linecolor=(0.0, 1.0, 0.0))

# Add skeleton lines

viewer.scene.add(Vis, linecolor=(0.0, 0.0, 1.0), linewidth=2)
viewer.scene.add(Vis2, linecolor=(0.0, 1.0, 1.0), linewidth=2)

viewer.show()