import compas.datastructures as cd
import compas.geometry as cg
import compas_cgal.straight_skeleton_2 as skeleton
import pathlib
import numpy as np
from shapely.geometry import Polygon, LineString, Point as ShPoint
from shapely.ops import unary_union
from compas_viewer import Viewer
from compas_viewer.scene import Tag

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
    使用 COMPAS mesh 與 Shapely 計算可見多邊形（2D 射線投射）
    
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

# === Load Meshes ===
filepath_road = pathlib.Path(__file__).parent / "Roads.stl"
filepath_block = pathlib.Path(__file__).parent / "Block.stl"

RoadMesh = cd.Mesh.from_stl(filepath_road)
BlockMesh = cd.Mesh.from_stl(filepath_block)

# === Find Boundary Lines ===
BlockLines = find_boundary_lines(BlockMesh)
RoadLines = find_boundary_lines(RoadMesh)

# === 製作圖形骨架 ===
outer_poly, hole_polys = mesh_boundaries_to_polygons(RoadMesh)
if outer_poly:
    print(f"Number of holes: {len(hole_polys)}")
    
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
    print("No valid boundary polygon found")
    Skeleton = None




node_count = sum(1 for _ in Skeleton.nodes())
print("skeleton nodes:", node_count)

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


from collections import deque

def compute_graph_distances_bfs(network, start_node):
    """
    使用 BFS 計算從 start_node 到所有其他節點的圖學距離
    
    參數:
        network: compas Network 或有 neighbors() 方法的圖結構
        start_node: 起始節點 index
    
    回傳:
        dict: {node_index: distance}
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    
    while queue:
        current = queue.popleft()
        current_dist = distances[current]
        
        # 取得鄰居節點
        for neighbor in network.neighbors(current):
            if neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    return distances

def is_line_intersecting_boundaries(start_pt, end_pt, boundary_lines, tolerance=1e-6):
    """
    檢查從 start_pt 到 end_pt 的線段是否與邊界相交
    
    參數:
        start_pt: (x, y, z) 起點座標
        end_pt: (x, y, z) 終點座標
        boundary_lines: list of compas.geometry.Line 邊界線段列表
        tolerance: 容差值，避免浮點數誤差
    
    回傳:
        bool: True 表示有相交（不可見），False 表示無相交（可見）
    """
    # 建立視線線段（只用 XY 平面）
    sight_line = LineString([
        (start_pt[0], start_pt[1]),
        (end_pt[0], end_pt[1])
    ])
    
    # 檢查是否與任何邊界線段相交
    for boundary_line in boundary_lines:
        # 轉換 compas Line 為 Shapely LineString
        boundary_geom = LineString([
            (boundary_line.start[0], boundary_line.start[1]),
            (boundary_line.end[0], boundary_line.end[1])
        ])
        
        # 檢查相交（排除端點重合的情況）
        if sight_line.intersects(boundary_geom):
            intersection = sight_line.intersection(boundary_geom)
            
            # 如果交點不是端點，則視為真正的相交
            if intersection.geom_type == "Point":
                # 檢查交點是否在線段內部（不是端點）
                dist_to_start = ((intersection.x - start_pt[0])**2 + 
                                (intersection.y - start_pt[1])**2)**0.5
                dist_to_end = ((intersection.x - end_pt[0])**2 + 
                              (intersection.y - end_pt[1])**2)**0.5
                total_dist = ((end_pt[0] - start_pt[0])**2 + 
                             (end_pt[1] - start_pt[1])**2)**0.5
                
                # 如果交點在線段內部（不是端點），則表示被遮擋
                if dist_to_start > tolerance and dist_to_end > tolerance and \
                   abs(dist_to_start + dist_to_end - total_dist) < tolerance:
                    return True
            elif intersection.geom_type == "LineString":
                # 線段重疊，視為相交
                return True
    
    return False

def find_visible_nodes_by_distance(skeleton, start_node, boundary_lines, view_point=None):
    """
    從給定的起點節點開始，逐層檢查不同圖學距離上的節點可見性
    
    參數:
        skeleton: compas Network (骨架圖)
        start_node: int 起始節點 index（用於圖距離計算）
        boundary_lines: list of compas.geometry.Line 邊界線段
        view_point: tuple (x, y, z) 視線起點座標，如果為 None 則使用 start_node 的座標
    
    回傳:
        dict: {
            'visible_nodes': {distance: [node_list]},  # 各距離上可見的節點
            'visible_boundary': [node_list],  # 可見的邊界節點
            'all_visible': [node_list],  # 所有可見節點
            'max_distance': int  # 最大可見距離
        }
    """
    # 第1步：計算所有節點的圖學距離
    print(f"\n=== 開始從節點 {start_node} 進行視線分析 ===")
    distances = compute_graph_distances_bfs(skeleton, start_node)
    print(f"圖中共有 {len(distances)} 個可達節點")
    
    # 視線起點座標（可以與圖距離起點不同）
    if view_point is not None:
        origin = view_point if len(view_point) == 3 else (view_point[0], view_point[1], 0)
        print(f"視線起點（測試點）: {origin}")
        print(f"圖距離起點（節點 {start_node}）: {skeleton.node_coordinates(start_node)}")
    else:
        origin = skeleton.node_coordinates(start_node)
        print(f"起點座標: {origin}")
    
    # 按距離分組節點
    nodes_by_distance = {}
    for node, dist in distances.items():
        if dist not in nodes_by_distance:
            nodes_by_distance[dist] = []
        nodes_by_distance[dist].append(node)
    
    print(f"距離層級: {sorted(nodes_by_distance.keys())}")
    
    # 第2步：使用節點屬性識別邊界節點和內部節點
    boundary_nodes = set()
    inner_nodes = set()
    
    for node in skeleton.nodes():
        node_type = skeleton.node_attribute(node, "type")
        if node_type == "boundary_node":
            boundary_nodes.add(node)
        elif node_type == "inner_node":
            inner_nodes.add(node)
    
    print(f"邊界節點數: {len(boundary_nodes)}, 內部節點數: {len(inner_nodes)}")
    
    # 第3步：逐層檢查可見性
    visible_by_distance = {}  # {distance: [visible_nodes]}
    visible_boundary = []  # 可見的邊界節點
    all_visited = {start_node}  # 已檢查的節點
    current_visible = {start_node}  # 當前層可見的節點（用於下一層的篩選）
    
    max_distance = max(nodes_by_distance.keys()) if nodes_by_distance else 0
    
    for distance in range(1, max_distance + 1):
        if distance not in nodes_by_distance:
            continue
            
        print(f"\n--- 檢查距離 {distance} 的節點 (共 {len(nodes_by_distance[distance])} 個) ---")
        
        visible_at_this_distance = []
        
        for node in nodes_by_distance[distance]:
            if node in all_visited:
                continue
                
            node_coords = skeleton.node_coordinates(node)
            
            # 檢查從起點到該節點的視線是否被遮擋
            is_visible = not is_line_intersecting_boundaries(
                origin, node_coords, boundary_lines
            )
            
            if is_visible:
                visible_at_this_distance.append(node)
                current_visible.add(node)
                
                # 如果是邊界節點，記錄下來
                if node in boundary_nodes:
                    visible_boundary.append(node)
                    print(f"  節點 {node} 可見 (邊界節點)")
                else:
                    print(f"  節點 {node} 可見 (內部節點)")
            else:
                print(f"  節點 {node} 不可見 (被遮擋)")
            
            all_visited.add(node)
        
        # 記錄這一層的可見節點
        if visible_at_this_distance:
            visible_by_distance[distance] = visible_at_this_distance
        
        # 檢查是否所有方向都已被遮擋（可選的終止條件）
        # 如果這一層沒有可見節點，可能後面的層級也不可見
        if not visible_at_this_distance:
            print(f"距離 {distance} 沒有可見節點，但繼續檢查更遠的層級...")
    
    # 第4步：統計結果
    all_visible_nodes = [node for nodes in visible_by_distance.values() for node in nodes]
    max_visible_distance = max(visible_by_distance.keys()) if visible_by_distance else 0
    
    print(f"\n=== 視線分析完成 ===")
    print(f"總共可見節點數: {len(all_visible_nodes)}")
    print(f"可見的邊界節點數: {len(visible_boundary)}")
    print(f"最大可見距離: {max_visible_distance}")
    
    return {
        'visible_nodes': visible_by_distance,
        'visible_boundary': visible_boundary,
        'all_visible': all_visible_nodes,
        'max_distance': max_visible_distance,
        'origin': origin
    }

def analyze_point_visibility(test_point, skeleton, boundary_polygon, boundary_lines):
    """
    對測試點進行完整的可見性分析
    
    參數:
        test_point: (x, y) 或 (x, y, z) 測試點座標
        skeleton: compas Network (骨架圖)
        boundary_polygon: Shapely Polygon 邊界多邊形
        boundary_lines: list of compas.geometry.Line 邊界線段
    
    回傳:
        dict: {
            'is_inside': bool,  # 是否在圖形內
            'test_point': tuple,  # 測試點座標
            'nearest_node': int or None,  # 最近的 inner_bisector 節點
            'nearest_distance': float or None,  # 到最近節點的距離
            'nearest_coords': tuple or None,  # 最近節點的座標
            'visibility_result': dict or None,  # 可見性分析結果
            'visible_boundary_nodes': list,  # 所有可見的 boundary_node
            'visible_boundary_coords': list,  # 可見 boundary_node 的座標
        }
    """
    # 轉換為 3D 座標
    test_point_3d = test_point if len(test_point) == 3 else (test_point[0], test_point[1], 0)
    test_point_2d = (test_point[0], test_point[1])
    
    print(f"\n{'='*70}")
    print(f"開始分析測試點: {test_point_2d}")
    print(f"{'='*70}")
    
    # 步驟1: 判斷點是否在圖形內
    shapely_point = ShPoint(test_point_2d)
    is_inside = boundary_polygon.contains(shapely_point)
    
    if not is_inside:
        print(f"✗ 測試點 {test_point_2d} 在圖形外，不進行可見性分析")
        return {
            'is_inside': False,
            'test_point': test_point_2d,
            'nearest_node': None,
            'nearest_distance': None,
            'nearest_coords': None,
            'visibility_result': None,
            'visible_boundary_nodes': [],
            'visible_boundary_coords': [],
        }
    
    print(f"✓ 測試點 {test_point_2d} 在圖形內")
    
    # 步驟2: 尋找最近的 inner_bisector 節點
    inner_bisector_nodes = []
    for node in skeleton.nodes():
        if skeleton.node_attribute(node, "type") == "inner_node":
            inner_bisector_nodes.append(node)
    
    print(f"找到 {len(inner_bisector_nodes)} 個 inner_bisector 節點")
    
    if not inner_bisector_nodes:
        print("警告：沒有找到 inner_bisector 節點")
        return {
            'is_inside': True,
            'test_point': test_point_2d,
            'nearest_node': None,
            'nearest_distance': None,
            'nearest_coords': None,
            'visibility_result': None,
            'visible_boundary_nodes': [],
            'visible_boundary_coords': [],
        }
    
    # 計算最近的 inner_bisector 節點
    min_distance = float('inf')
    nearest_node = None
    nearest_coords = None
    
    for node in inner_bisector_nodes:
        node_coords = skeleton.node_coordinates(node)
        dist = ((node_coords[0] - test_point_2d[0])**2 + 
                (node_coords[1] - test_point_2d[1])**2)**0.5
        
        if dist < min_distance:
            min_distance = dist
            nearest_node = node
            nearest_coords = node_coords
    
    print(f"最近的 inner_bisector 節點: {nearest_node}")
    print(f"節點座標: {nearest_coords}")
    print(f"距離: {min_distance:.6f}")
    
    # 步驟3: 執行可見性分析
    print(f"\n開始從測試點進行可見性分析...")
    visibility_result = find_visible_nodes_by_distance(
        skeleton=skeleton,
        start_node=nearest_node,  # 用最近節點計算圖距離
        boundary_lines=boundary_lines,
        view_point=test_point_3d  # 用測試點作為視線起點
    )
    
    # 步驟4: 提取所有可見的 boundary_node
    visible_boundary_nodes = visibility_result['visible_boundary']
    visible_boundary_coords = []
    
    print(f"\n{'='*70}")
    print(f"可見的 boundary_node 列表 (共 {len(visible_boundary_nodes)} 個):")
    print(f"{'='*70}")
    
    for i, node in enumerate(visible_boundary_nodes, 1):
        coords = skeleton.node_coordinates(node)
        visible_boundary_coords.append(coords)
        print(f"  {i}. 節點 {node}: {coords}")
    
    print(f"{'='*70}\n")
    
    return {
        'is_inside': True,
        'test_point': test_point_2d,
        'nearest_node': nearest_node,
        'nearest_distance': min_distance,
        'nearest_coords': nearest_coords,
        'visibility_result': visibility_result,
        'visible_boundary_nodes': visible_boundary_nodes,
        'visible_boundary_coords': visible_boundary_coords,
    }

def compute_visibility_polygon(test_point, visible_boundary_coords, boundary_lines, angle_offset=0.1, max_dist=1000.0):
    """
    計算可見多邊形，使用射線投射方法
    
    參數:
        test_point: (x, y) 或 (x, y, z) 測試點座標
        visible_boundary_coords: list 可見的 boundary_node 座標列表
        boundary_lines: list of compas.geometry.Line 邊界線段
        angle_offset: float 向可見點兩側投射的角度偏移（度）
        max_dist: float 射線最大距離
    
    回傳:
        dict: {
            'polygon_points': list,  # 多邊形頂點列表（已排序）
            'compas_polygon': compas.geometry.Polygon,  # COMPAS 多邊形對象
            'shapely_polygon': shapely.geometry.Polygon,  # Shapely 多邊形對象
            'area': float,  # 多邊形面積
            'perimeter': float,  # 多邊形周長
        }
    """
    test_point_2d = (test_point[0], test_point[1])
    
    print(f"\n{'='*70}")
    print(f"計算可見多邊形")
    print(f"測試點: {test_point_2d}")
    print(f"可見 boundary 點數: {len(visible_boundary_coords)}")
    print(f"{'='*70}")
    
    if len(visible_boundary_coords) < 3:
        print("警告：可見點數少於3個，無法形成多邊形")
        return None
    
    # 準備 Shapely 邊界線段
    shapely_boundaries = []
    for line in boundary_lines:
        shapely_boundaries.append(LineString([
            (line.start[0], line.start[1]),
            (line.end[0], line.end[1])
        ]))
    
    # 收集所有多邊形頂點（角度, 點）
    polygon_vertices = []
    
    # 處理每個可見的 boundary_node
    for i, boundary_coords in enumerate(visible_boundary_coords):
        boundary_2d = (boundary_coords[0], boundary_coords[1])
        
        # 計算到 boundary_node 的角度
        dx = boundary_2d[0] - test_point_2d[0]
        dy = boundary_2d[1] - test_point_2d[1]
        angle = np.arctan2(dy, dx)  # 弧度
        
        # 投射 +angle_offset 和 -angle_offset 的兩條射線
        for offset_deg in [-angle_offset, angle_offset]:
            offset_rad = np.radians(offset_deg)
            ray_angle = angle + offset_rad
            
            # 射線方向
            ray_dx = np.cos(ray_angle)
            ray_dy = np.sin(ray_angle)
            far_pt = (test_point_2d[0] + ray_dx * max_dist, 
                     test_point_2d[1] + ray_dy * max_dist)
            
            ray = LineString([test_point_2d, far_pt])
            
            # 找到最近的交點
            min_dist = float('inf')
            closest_intersection = None
            
            for boundary in shapely_boundaries:
                if ray.intersects(boundary):
                    intersection = ray.intersection(boundary)
                    
                    if intersection.geom_type == "Point":
                        pt = (intersection.x, intersection.y)
                        dist = ((pt[0] - test_point_2d[0])**2 + 
                               (pt[1] - test_point_2d[1])**2)**0.5
                        if dist < min_dist and dist > 1e-6:  # 避免起點本身
                            min_dist = dist
                            closest_intersection = pt
                    
                    elif intersection.geom_type == "MultiPoint":
                        for point in intersection.geoms:
                            pt = (point.x, point.y)
                            dist = ((pt[0] - test_point_2d[0])**2 + 
                                   (pt[1] - test_point_2d[1])**2)**0.5
                            if dist < min_dist and dist > 1e-6:
                                min_dist = dist
                                closest_intersection = pt
            
            # 如果找到交點，加入頂點列表
            if closest_intersection:
                # 計算交點的角度（用於排序）
                inter_dx = closest_intersection[0] - test_point_2d[0]
                inter_dy = closest_intersection[1] - test_point_2d[1]
                inter_angle = np.arctan2(inter_dy, inter_dx)
                polygon_vertices.append((inter_angle, closest_intersection))
        
        # 也將 boundary_node 本身加入（如果它確實可見）
        polygon_vertices.append((angle, boundary_2d))
    
    # 按角度排序頂點
    polygon_vertices.sort(key=lambda x: x[0])
    
    # 去除重複點（使用小的容差）
    tolerance = 1e-3
    unique_vertices = []
    for angle, pt in polygon_vertices:
        is_duplicate = False
        for existing_pt in unique_vertices:
            dist = ((pt[0] - existing_pt[0])**2 + (pt[1] - existing_pt[1])**2)**0.5
            if dist < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_vertices.append(pt)
    
    print(f"原始頂點數: {len(polygon_vertices)}")
    print(f"去重後頂點數: {len(unique_vertices)}")
    
    if len(unique_vertices) < 3:
        print("警告：去重後頂點數少於3個，無法形成多邊形")
        return None
    
    # 建立 COMPAS 多邊形
    compas_points = [cg.Point(pt[0], pt[1], 0) for pt in unique_vertices]
    compas_polygon = cg.Polygon(compas_points)
    
    # 建立 Shapely 多邊形
    shapely_polygon = Polygon(unique_vertices)
    
    # 計算面積和周長
    area = shapely_polygon.area
    perimeter = shapely_polygon.length
    
    print(f"多邊形面積: {area:.6f}")
    print(f"多邊形周長: {perimeter:.6f}")
    print(f"{'='*70}\n")
    
    return {
        'polygon_points': unique_vertices,
        'compas_polygon': compas_polygon,
        'shapely_polygon': shapely_polygon,
        'area': area,
        'perimeter': perimeter,
    }

def find_nearest_inner_bisector_node(point, skeleton, boundary_polygon):
    """
    對於任意點，判斷是否在圖形內，若在圖形內則尋找最近的 inner_bisector 節點
    
    參數:
        point: (x, y) 或 (x, y, z) 座標
        skeleton: compas Network (骨架圖)
        boundary_polygon: Shapely Polygon 邊界多邊形
    
    回傳:
        dict: {
            'is_inside': bool,
            'nearest_node': int or None,
            'distance': float or None,
            'node_coords': tuple or None
        }
    """
    # 轉換為 2D 座標
    test_point_2d = (point[0], point[1])
    shapely_point = ShPoint(test_point_2d)
    
    print(f"\n=== 測試點 {test_point_2d} ===")
    
    # 判斷點是否在多邊形內
    is_inside = boundary_polygon.contains(shapely_point)
    
    if not is_inside:
        print(f"點 {test_point_2d} 不在圖形內")
        return {
            'is_inside': False,
            'nearest_node': None,
            'distance': None,
            'node_coords': None,
            'point': test_point_2d
        }
    
    print(f"點 {test_point_2d} 在圖形內，尋找最近的 inner_bisector 節點...")
    
    # 收集所有 inner_node (inner_bisector 上的節點)
    inner_bisector_nodes = []
    for node in skeleton.nodes():
        # 使用節點屬性直接判斷
        if skeleton.node_attribute(node, "type") == "inner_node":
            inner_bisector_nodes.append(node)
    
    print(f"找到 {len(inner_bisector_nodes)} 個 inner_bisector 節點")
    
    if not inner_bisector_nodes:
        print("警告：沒有找到 inner_bisector 節點")
        return {
            'is_inside': True,
            'nearest_node': None,
            'distance': None,
            'node_coords': None,
            'point': test_point_2d
        }
    
    # 尋找最近的 inner_bisector 節點
    min_distance = float('inf')
    nearest_node = None
    nearest_coords = None
    
    for node in inner_bisector_nodes:
        node_coords = skeleton.node_coordinates(node)
        # 計算 2D 距離
        dist = ((node_coords[0] - test_point_2d[0])**2 + 
                (node_coords[1] - test_point_2d[1])**2)**0.5
        
        if dist < min_distance:
            min_distance = dist
            nearest_node = node
            nearest_coords = node_coords
    
    print(f"最近的 inner_bisector 節點: {nearest_node}")
    print(f"節點座標: {nearest_coords}")
    print(f"距離: {min_distance:.6f}")
    
    return {
        'is_inside': True,
        'nearest_node': nearest_node,
        'distance': min_distance,
        'node_coords': nearest_coords,
        'point': test_point_2d
    }

viewer = Viewer()
#viewer.scene.add(RoadMesh)

'''
# Add boundary lines
for line in RoadLines:
    viewer.scene.add(line, linecolor=(1.0, 0.0, 0.0))
for line in BlockLines:
    viewer.scene.add(line, linecolor=(0.0, 1.0, 0.0))
'''

for edge in Skeleton.edges():
    line = Skeleton.edge_line(edge)
    if Skeleton.edge_attribute(edge, "inner_bisector"):
        viewer.scene.add(line, linecolor=(1.0, 0.0, 0.0), linewidth=2)
    elif Skeleton.edge_attribute(edge, "bisector"):
        viewer.scene.add(line, linecolor=(0.0, 0.0, 1.0))
    else:
        viewer.scene.add(line, linecolor=(0.0, 0.0, 0.0))

# === 測試：尋找最近的 inner_bisector 節點並進行可見性分析 ===
# 創建邊界多邊形用於點內外判斷
if outer_poly:
    # 使用 Shapely Polygon
    boundary_polygon = Polygon(outer_poly, holes=hole_polys)
    
    # 定義測試點
    test_points = [
        (7.8, -4.8, 0),  # 測試點1
        (-0.2, 0.2, 0)   # 測試點2
    ]
    
    test_colors = [
        (1.0, 1.0, 0.0),  # 黃色
        (1.0, 0.5, 0.0)   # 橙色
    ]
    
    # 對每個測試點進行分析
    for idx, test_point in enumerate(test_points):
        test_color = test_colors[idx]
        
        # 使用新的統一分析函數
        result = analyze_point_visibility(
            test_point=test_point,
            skeleton=Skeleton,
            boundary_polygon=boundary_polygon,
            boundary_lines=RoadLines
        )
        
        # 視覺化結果
        if not result['is_inside']:
            # 點在圖形外
            viewer.scene.add(
                cg.Point(*test_point),
                pointcolor=(0.5, 0.5, 0.5),  # 灰色
                pointsize=20
            )
            continue
        
        # 點在圖形內但沒有找到 inner_bisector 節點
        if result['nearest_node'] is None:
            viewer.scene.add(
                cg.Point(*test_point),
                pointcolor=test_color,
                pointsize=20
            )
            continue
        
        # 完整的視覺化
        # 1. 測試點（大號標記）
        viewer.scene.add(
            cg.Point(*test_point),
            pointcolor=test_color,
            pointsize=22
        )
        
        # 2. 最近的 inner_bisector 節點（綠色）
        viewer.scene.add(
            cg.Point(*result['nearest_coords']),
            pointcolor=(0.0, 1.0, 0.0),
            pointsize=18
        )
        
        # 3. 連接測試點和最近節點
        connection_line = cg.Line(
            cg.Point(*test_point),
            cg.Point(*result['nearest_coords'])
        )
        viewer.scene.add(connection_line, linecolor=(0.0, 1.0, 0.0), linewidth=3)
        
        # 4. 視覺化所有可見的 boundary_node（用特殊顏色標記）
        for boundary_node in result['visible_boundary_nodes']:
            boundary_coords = Skeleton.node_coordinates(boundary_node)
            viewer.scene.add(
                cg.Point(*boundary_coords),
                pointcolor=(1.0, 0.0, 1.0),  # 洋紅色
                pointsize=12
            )
        
        # 5. 視覺化其他可見節點（按距離用不同亮度）
        visibility_result = result['visibility_result']
        if visibility_result:
            visibility_colors = [
                (*test_color, 1.0),   # 距離1
                (*test_color, 0.7),   # 距離2
                (*test_color, 0.5),   # 距離3
                (*test_color, 0.3),   # 距離4+
            ]
            
            for distance, nodes in visibility_result['visible_nodes'].items():
                color_idx = min(distance - 1, len(visibility_colors) - 1)
                vis_color = visibility_colors[color_idx][:3]
                
                for node in nodes:
                    # 跳過 boundary_node（已經用洋紅色標記）
                    if Skeleton.node_attribute(node, "type") == "boundary_node":
                        continue
                    
                    node_pt = Skeleton.node_coordinates(node)
                    viewer.scene.add(
                        cg.Point(*node_pt),
                        pointcolor=vis_color,
                        pointsize=8
                    )
                    
                    # 畫視線（半透明）
                    sight_line = cg.Line(test_point, node_pt)
                    viewer.scene.add(sight_line, linecolor=(*vis_color, 0.15), linewidth=0.5)
        
        # 6. 計算並視覺化可見多邊形
        if result['visible_boundary_coords'] and len(result['visible_boundary_coords']) >= 3:
            vis_polygon = compute_visibility_polygon(
                test_point=test_point,
                visible_boundary_coords=result['visible_boundary_coords'],
                boundary_lines=RoadLines,
                angle_offset=0.1,  # ±0.1度
                max_dist=1000.0
            )
            
            if vis_polygon:
                # 視覺化可見多邊形（半透明填充）
                compas_polygon = vis_polygon['compas_polygon']
                
                # 繪製多邊形邊界（實線）
                poly_points = vis_polygon['polygon_points']
                for i in range(len(poly_points)):
                    p1 = poly_points[i]
                    p2 = poly_points[(i + 1) % len(poly_points)]
                    edge_line = cg.Line(
                        cg.Point(p1[0], p1[1], 0.1),  # 稍微抬高避免z-fighting
                        cg.Point(p2[0], p2[1], 0.1)
                    )
                    viewer.scene.add(edge_line, linecolor=test_color, linewidth=3)
                
                # 繪製多邊形頂點
                for pt in poly_points:
                    viewer.scene.add(
                        cg.Point(pt[0], pt[1], 0.1),
                        pointcolor=(1.0, 1.0, 1.0),  # 白色
                        pointsize=6
                    )
                
                print(f"\n測試點 {idx + 1} 的可見多邊形：")
                print(f"  - 頂點數: {len(poly_points)}")
                print(f"  - 面積: {vis_polygon['area']:.6f}")
                print(f"  - 周長: {vis_polygon['perimeter']:.6f}")
    
    print("\n" + "="*60)
    print("視覺化圖例：")
    print("  - 黃色/橙色大點 = 測試點（在圖形內）")
    print("  - 灰色大點 = 測試點（在圖形外）")
    print("  - 綠色點 = 最近的 inner_bisector 節點")
    print("  - 洋紅色點 = 可見的 boundary_node")
    print("  - 小彩色點 = 從測試點可見的其他骨架節點")
    print("  - 綠色粗線 = 測試點到最近節點")
    print("  - 細彩色線 = 視線（半透明）")
    print("  - 黃色/橙色粗線 = 可見多邊形邊界")
    print("  - 白色小點 = 可見多邊形頂點")
    print("="*60 + "\n")

# === 展示如何使用可見多邊形進行布爾運算 ===
print("\n" + "="*60)
print("可見多邊形布爾運算示例")
print("="*60)

# 重新計算並存儲所有測試點的可見多邊形
visibility_polygons = []

for idx, test_point in enumerate(test_points):
    result = analyze_point_visibility(
        test_point=test_point,
        skeleton=Skeleton,
        boundary_polygon=boundary_polygon,
        boundary_lines=RoadLines
    )
    
    if result['is_inside'] and result['visible_boundary_coords'] and len(result['visible_boundary_coords']) >= 3:
        vis_polygon = compute_visibility_polygon(
            test_point=test_point,
            visible_boundary_coords=result['visible_boundary_coords'],
            boundary_lines=RoadLines,
            angle_offset=0.1,
            max_dist=1000.0
        )
        
        if vis_polygon:
            visibility_polygons.append({
                'test_point': test_point,
                'shapely_polygon': vis_polygon['shapely_polygon'],
                'compas_polygon': vis_polygon['compas_polygon'],
                'area': vis_polygon['area'],
                'perimeter': vis_polygon['perimeter'],
                'index': idx
            })
            print(f"\n測試點 {idx + 1} {test_point[:2]}:")
            print(f"  面積: {vis_polygon['area']:.6f}")
            print(f"  周長: {vis_polygon['perimeter']:.6f}")

# 如果有多個可見多邊形，可以進行布爾運算
if len(visibility_polygons) >= 2:
    print(f"\n{'='*60}")
    print("布爾運算示例：")
    print(f"{'='*60}")
    
    poly1 = visibility_polygons[0]['shapely_polygon']
    poly2 = visibility_polygons[1]['shapely_polygon']
    
    # 交集
    intersection = poly1.intersection(poly2)
    if not intersection.is_empty:
        print(f"\n交集面積: {intersection.area:.6f}")
    else:
        print(f"\n兩個可見多邊形沒有交集")
    
    # 聯集
    union = poly1.union(poly2)
    print(f"聯集面積: {union.area:.6f}")
    
    # 差集
    difference = poly1.difference(poly2)
    print(f"差集面積 (多邊形1 - 多邊形2): {difference.area:.6f}")
    
    print(f"\n可見多邊形已存儲在 'visibility_polygons' 列表中")
    print(f"每個元素包含：test_point, shapely_polygon, compas_polygon, area, perimeter")
    print(f"{'='*60}\n")

viewer.show()