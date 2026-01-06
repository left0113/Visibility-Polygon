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

# === 執行視線可見性分析 ===
# 設定起點節點 index（你可以修改這個值）
START_NODE = 310

# 執行分析
try:
    visibility_result = find_visible_nodes_by_distance(
        skeleton=Skeleton,
        start_node=START_NODE,
        boundary_lines=RoadLines  # 使用道路邊界作為遮擋物
    )
    
    # 加入起點節點（用特殊顏色標記）
    origin_pt = Skeleton.node_coordinates(START_NODE)
    viewer.scene.add(
        cg.Point(*origin_pt),
        pointcolor=(1.0, 0.0, 1.0),  # 洋紅色
        pointsize=20
    )
    print(f"Added origin node {START_NODE}: {origin_pt}")
    
    # 視覺化可見節點（按距離用不同顏色）
    distance_colors = [
        (0.0, 1.0, 0.0),   # 綠色 - 距離1
        (0.0, 1.0, 1.0),   # 青色 - 距離2
        (0.0, 0.5, 1.0),   # 淺藍 - 距離3
        (1.0, 1.0, 0.0),   # 黃色 - 距離4
        (1.0, 0.5, 0.0),   # 橙色 - 距離5
        (1.0, 0.0, 0.0),   # 紅色 - 距離6+
    ]
    
    for distance, nodes in visibility_result['visible_nodes'].items():
        color_idx = min(distance - 1, len(distance_colors) - 1)
        color = distance_colors[color_idx]
        
        for node in nodes:
            node_pt = Skeleton.node_coordinates(node)
            viewer.scene.add(
                cg.Point(*node_pt),
                pointcolor=color,
                pointsize=12
            )
            
            # 畫出從起點到可見節點的視線
            sight_line = cg.Line(origin_pt, node_pt)
            viewer.scene.add(sight_line, linecolor=(*color, 0.3), linewidth=1)
    
    print(f"\n視覺化完成：")
    print(f"  - 洋紅色點 = 起點 (node {START_NODE})")
    print(f"  - 綠色點 = 距離1可見節點")
    print(f"  - 青色點 = 距離2可見節點")
    print(f"  - 依此類推...")
    
except Exception as e:
    print(f"視線分析失敗: {e}")
    import traceback
    traceback.print_exc()

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
        (4.0, -4.0, 0),  # 測試點1
        (1.0, 0.0, 0)   # 測試點2
    ]
    
    test_colors = [
        (1.0, 1.0, 0.0),  # 黃色
        (1.0, 0.5, 0.0)   # 橙色
    ]
    
    # 對每個測試點進行分析
    for idx, test_point in enumerate(test_points):
        test_color = test_colors[idx]
        test_point_2d = (test_point[0], test_point[1])
        
        print(f"\n{'='*60}")
        print(f"測試點 {idx + 1}: {test_point_2d}")
        print(f"{'='*60}")
        
        # 步驟1: 尋找最近的 inner_bisector 節點
        result = find_nearest_inner_bisector_node(test_point, Skeleton, boundary_polygon)
        
        if not result['is_inside']:
            # 點在圖形外，不進行分析
            print(f"✗ 測試點 {test_point_2d} 在圖形外，不進行可見性分析")
            
            # 標記測試點（灰色）
            viewer.scene.add(
                cg.Point(*test_point),
                pointcolor=(0.5, 0.5, 0.5),  # 灰色
                pointsize=20
            )
            continue
        
        # 點在圖形內
        print(f"✓ 測試點 {test_point_2d} 在圖形內")
        
        if result['nearest_node'] is None:
            print("  警告：未找到 inner_bisector 節點")
            viewer.scene.add(
                cg.Point(*test_point),
                pointcolor=test_color,
                pointsize=20
            )
            continue
        
        print(f"  最近的 inner_bisector 節點: {result['nearest_node']}")
        print(f"  距離: {result['distance']:.6f}")
        
        # 步驟2: 從最近節點進行圖距離計算，用測試點進行可見性判斷
        try:
            visibility_result = find_visible_nodes_by_distance(
                skeleton=Skeleton,
                start_node=result['nearest_node'],  # 用最近節點計算圖距離
                boundary_lines=RoadLines,
                view_point=test_point  # 用測試點作為視線起點
            )
            
            # 視覺化測試點（大號標記）
            viewer.scene.add(
                cg.Point(*test_point),
                pointcolor=test_color,
                pointsize=22
            )
            
            # 視覺化最近的 inner_bisector 節點（綠色環形）
            viewer.scene.add(
                cg.Point(*result['node_coords']),
                pointcolor=(0.0, 1.0, 0.0),
                pointsize=18
            )
            
            # 連接測試點和最近節點（虛線效果）
            connection_line = cg.Line(
                cg.Point(*test_point),
                cg.Point(*result['node_coords'])
            )
            viewer.scene.add(connection_line, linecolor=(0.0, 1.0, 0.0), linewidth=3)
            
            # 視覺化可見節點（按距離用不同亮度）
            visibility_colors = [
                (*test_color, 1.0),   # 距離1 - 完全不透明
                (*test_color, 0.7),   # 距離2
                (*test_color, 0.5),   # 距離3
                (*test_color, 0.3),   # 距離4+
            ]
            
            for distance, nodes in visibility_result['visible_nodes'].items():
                color_idx = min(distance - 1, len(visibility_colors) - 1)
                vis_color = visibility_colors[color_idx][:3]  # 只用 RGB
                
                for node in nodes:
                    node_pt = Skeleton.node_coordinates(node)
                    viewer.scene.add(
                        cg.Point(*node_pt),
                        pointcolor=vis_color,
                        pointsize=8
                    )
                    
                    # 畫視線（半透明）
                    sight_line = cg.Line(test_point, node_pt)
                    viewer.scene.add(sight_line, linecolor=(*vis_color, 0.15), linewidth=0.5)
            
            print(f"\n  可見性分析完成：")
            print(f"    - 總可見節點數: {len(visibility_result['all_visible'])}")
            print(f"    - 可見邊界節點數: {len(visibility_result['visible_boundary'])}")
            print(f"    - 最大可見距離: {visibility_result['max_distance']}")
            
        except Exception as e:
            print(f"  可見性分析失敗: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("視覺化圖例：")
    print("  - 黃色/橙色大點 = 測試點（在圖形內）")
    print("  - 灰色大點 = 測試點（在圖形外）")
    print("  - 綠色點 = 最近的 inner_bisector 節點")
    print("  - 小彩色點 = 從測試點可見的骨架節點")
    print("  - 綠色粗線 = 測試點到最近節點")
    print("  - 細彩色線 = 視線（半透明）")
    print("="*60 + "\n")

viewer.show()