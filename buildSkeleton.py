import compas.geometry as cg
import compas_cgal.straight_skeleton_2 as skeleton
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

def compute_isovist_compas(mesh, origin, n_rays=360, max_dist=1000.0):
    """
    使用 COMPAS mesh 與 Shapely 計算可見多邊形(2D 射線投射)
    
    mesh : compas.datastructures.Mesh
    origin : (x, y, z) 或 (x, y) - 發射點座標
    n_rays : int - 射線數量
    max_dist : float - 射線最大距離
    回傳 : compas.geometry.Polyline
    """
    origin = (origin[0], origin[1])  # 僅使用 XY 座標
    
    # 從 mesh 提取 2D 邊界（投影到 XY 平面）
    boundary_edges = find_boundary_edges(mesh)
    lines=[]

    for u, v in boundary_edges:
        a = mesh.vertex_coordinates(u)
        b = mesh.vertex_coordinates(v)
        lines.append(LineString([(a[0], a[1]), (b[0], b[1])])) # 只用 XY 座標
    
    # 用 Shapely 建立邊界多邊形或 LineString 集合
    boundary = unary_union(lines)
    
    print(f"[isovist] origin : {origin}")
    print(f"[isovist] boundary type: {boundary.geom_type}, n_rays: {n_rays}")
    
    # 產生射線
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    pts = []
    
    for angle in angles:
        # 射線方向
        dx = np.cos(angle)
        dy = np.sin(angle)
        far_pt = (origin[0] + dx * max_dist, origin[1] + dy * max_dist)
        ray = LineString([origin, far_pt])
        
        # 求交點
        inter = ray.intersection(boundary)
        
        if inter.is_empty:
            # 無相交，用射線終點
            pts.append(cg.Point(far_pt[0], far_pt[1]))
        elif inter.geom_type == "Point":
            # 單一交點
            pts.append(cg.Point(inter.x, inter.y))
        elif inter.geom_type == "LineString":
            # 線段，取距離最近的端點或中點
            closest = min(
                [inter.coords[0], inter.coords[-1]],
                key=lambda p: (p[0] - origin[0])**2 + (p[1] - origin[1])**2
            )
            pts.append(cg.Point(closest[0], closest[1]))
        elif inter.geom_type == "MultiPoint":
            # 多個點，取最近的
            closest = min(
                [(pt.x, pt.y) for pt in inter.geoms],
                key=lambda p: (p[0] - origin[0])**2 + (p[1] - origin[1])**2
            )
            pts.append(cg.Point(closest[0], closest[1]))
    
    print(f"[isovist] final points: {len(pts)}")
    
    if len(pts) < 3:
        print("[isovist] WARNING: Less than 3 points")
        return None
    
    poly = cg.Polyline(pts)
    return poly

def mesh_boundaries_to_polygons(mesh):
    """
    將 mesh 的邊界邊轉成 Shapely Polygon 物件列表
    回傳 (outer_polygon, hole_polygons_list)
    """
    
    boundary_edges = find_boundary_edges(mesh)
    
    # 組成邊界環（loops）
    edge_set = set((min(u, v), max(u, v)) for u, v in boundary_edges)
    loops = []
    
    while edge_set:
        a, b = edge_set.pop()
        loop = [a, b]
        prev, curr = a, b
        
        while True:
            found = None
            for e in list(edge_set):
                if e[0] == curr:
                    found = e
                    nextv = e[1]
                    break
                elif e[1] == curr:
                    found = e
                    nextv = e[0]
                    break
            
            if not found:
                break
            
            edge_set.remove(found)
            loop.append(nextv)
            prev, curr = curr, nextv
            
            if curr == loop[0]:
                break
        
        loops.append(loop)
    
    # 把每個 loop 轉成 Polygon
    polygons = []
    for loop in loops:
        pts = [mesh.vertex_coordinates(k)[:2] for k in loop]  # 只用 XY
        if len(pts) >= 3:
            coords = [tuple(p) for p in pts]
            # 計算周長
            perim = 0.0
            for i in range(len(coords)):
                x1, y1 = coords[i]
                x2, y2 = coords[(i + 1) % len(coords)]
                perim += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            polygons.append((coords, perim))
    
    # 按周長排序，最長的作為外邊界 (回傳 coords lists)
    if not polygons:
        return None, []
    
    polygons.sort(key=lambda x: x[1], reverse=True)
    outer_coords = polygons[0][0]
    hole_coords = [p[0] for p in polygons[1:]]
    
    print(f"[mesh_to_polygon] outer perimeter: {polygons[0][1]:.2f}")
    print(f"[mesh_to_polygon] hole count: {len(hole_coords)}")
    for i, (p, length) in enumerate(polygons[1:]):
        print(f"  hole {i}: perimeter {length:.2f}")
    
    return outer_coords, hole_coords

def signed_area(coords):
    a = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a

def ensure_ccw(coords):
    if signed_area(coords) < 0:
        return list(reversed(coords))
    return coords

def costume_straight_skeleton(mesh):
    # === 製作圖形骨架 ===
    print("\n=== 製作圖形骨架 ===")
    outer_poly, hole_polys = mesh_boundaries_to_polygons(mesh)
    if outer_poly:
        print(f"空洞數量: {len(hole_polys)}")
    
        # 確保外環為 CCW（normal = +Z）
        outer_ccw = ensure_ccw(outer_poly)
        # 把 2D 座標轉成 3D (x, y, 0)
        outer_3d = [(x, y, 0) for x, y in outer_ccw]

        # 對洞做同樣處理，但需反向成 CW（normal = -Z）
        holes_3d = []
        for hole in hole_polys:
            hole_ccw = ensure_ccw(hole)
            # 反向成 CW（相反於外環方向）
            hole_cw = list(reversed(hole_ccw))
            hole_3d = [(x, y, 0) for x, y in hole_cw]
            holes_3d.append(hole_3d)
    
        # 現在傳入的是：外環 CCW 3D + 洞 CW 3D
        Skeleton = skeleton.interior_straight_skeleton_with_holes(outer_3d, holes_3d)
    else:
        print("沒有可用多邊形邊界來製作骨架")
        Skeleton = None

    # === 為骨架節點添加 inner_node / boundary_node 屬性 ===
    print("\n=== 為骨架節點添加屬性 ===")

    # 遍歷所有節點，根據連接的邊類型設置屬性
    for node in Skeleton.nodes():
        is_inner_node = False
        is_boundary_node = False

        # 檢查該節點連接的所有邊
        for u, v in Skeleton.edges():
            if node == u or node == v:
                # 檢查邊的屬性
                if Skeleton.edge_attribute((u, v), "inner_bisector"):
                    is_inner_node = True
                elif Skeleton.edge_attribute((u, v), "bisector"):
                    is_boundary_node = True
    
        # 設置節點屬性
        if is_inner_node:
            Skeleton.node_attribute(node, "type", "inner_node")
        elif is_boundary_node:
            Skeleton.node_attribute(node, "type", "boundary_node")
        else:
            # 既不在 inner_bisector 也不在 boundary 邊上的節點（不應該存在）
            Skeleton.node_attribute(node, "type", "unknown")

    # 統計節點類型
    inner_nodes_count = sum(1 for node in Skeleton.nodes() 
                            if Skeleton.node_attribute(node, "type") == "inner_node")
    boundary_nodes_count = sum(1 for node in Skeleton.nodes() 
                           if Skeleton.node_attribute(node, "type") == "boundary_node")

    print(f"inner_node 數量: {inner_nodes_count}")    
    print(f"boundary_node 數量: {boundary_nodes_count}")
    print("節點屬性設置完成\n")

    return Skeleton
