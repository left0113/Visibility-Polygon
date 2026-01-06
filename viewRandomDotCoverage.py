import compas.datastructures as cd
import compas.geometry as cg
import pathlib
import numpy as np
from shapely.geometry import Polygon, LineString, Point as ShPoint
from compas_viewer import Viewer
import isovist
import buildSkeleton as skel

# === Load Road Mesh ===
filepath_road = pathlib.Path(__file__).parent / "Roads.stl"
RoadMesh = cd.Mesh.from_stl(filepath_road)

# === Find Boundary Lines ===
RoadLines = isovist.find_boundary_lines(RoadMesh)

# === 製作圖形骨架 ===
Skeleton = skel.costume_straight_skeleton(RoadMesh)

# === 獲取邊界多邊形 ===
outer_poly, hole_polys = skel.mesh_boundaries_to_polygons(RoadMesh)

if not outer_poly:
    print("錯誤：無法提取邊界多邊形")
    exit()

# === 計算 road_mesh 的 XY 範圍 ===
print("\n=== 計算 Road Mesh XY 範圍 ===")
outer_poly_array = np.array(outer_poly)
x_min, x_max = outer_poly_array[:, 0].min(), outer_poly_array[:, 0].max()
y_min, y_max = outer_poly_array[:, 1].min(), outer_poly_array[:, 1].max()

print(f"XY 範圍：")
print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")

# === 建立邊界多邊形 ===
boundary_polygon = Polygon(outer_poly, holes=hole_polys)
print(f"邊界多邊形面積: {boundary_polygon.area:.6f}")

# === 生成隨機點 ===
# np.random.seed(42)  # 註解掉固定種子，每次運行都會產生不同的隨機點
num_points = 5000
random_points = []
print(f"\n=== 生成 {num_points} 個隨機點 ===\n")
for i in range(num_points):
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    random_points.append((x, y, 0))

print(f"已生成 {len(random_points)} 個隨機點")

# === 對每個點進行可見性分析並計算 area_ratio ===
print("\n=== 對隨機點進行可見性分析 ===")
point_results = []

for idx, test_point in enumerate(random_points):
    if (idx + 1) % 10 == 0:
        print(f"處理進度: {idx + 1}/{num_points}")
    
    # 分析該點的可見性
    result = isovist.analyze_point_visibility(
        point=test_point,
        skeleton=Skeleton,
        boundary_polygon=boundary_polygon,
        boundary_lines=RoadLines
    )
    
    # 如果點在圖形內且找到可見邊界節點
    if result['is_inside'] and result['visible_boundary_coords'] and len(result['visible_boundary_coords']) >= 3:
        vis_polygon = isovist.compute_visibility_polygon(
            points=test_point,
            visible_boundary_coords=result['visible_boundary_coords'],
            boundary_lines=RoadLines,
            angle_offset=0.1,
            max_dist=1000.0,
            road_polygon=boundary_polygon
        )
        
        if vis_polygon and vis_polygon['area_ratio'] is not None:
            point_results.append({
                'index': idx,
                'point': test_point,
                'area': vis_polygon['area'],
                'area_ratio': vis_polygon['area_ratio'],
                'perimeter': vis_polygon['perimeter'],
                'polygon_points': vis_polygon['polygon_points'],
                'shapely_polygon': vis_polygon['shapely_polygon']
            })

print(f"\n成功分析 {len(point_results)} 個有效點")

# === 特殊排名方式：選出面積最大的點，剔除可以見到它的點，再排名 ===
print("\n=== 特殊排名方式 ===")
print("步驟 1: 找出面積最大的點")

# 按面積排序找出最大點
point_results_by_area = sorted(point_results, key=lambda x: x['area'], reverse=True)
max_area_point = point_results_by_area[0]
max_area_point_idx = max_area_point['index']
max_area_point_coords = max_area_point['point']
max_area_point_polygon = max_area_point['shapely_polygon']

print(f"面積最大的點: 索引 {max_area_point_idx}, 面積: {max_area_point['area']:.6f}, 座標: ({max_area_point_coords[0]:.2f}, {max_area_point_coords[1]:.2f})")

# 步驟 2: 逐次選出最大面積的點，並剔除能看到它的所有點
print("步驟 2: 逐次選出最大面積的點...")
remaining_points = point_results.copy()  # 剩餘未被選中的點
final_ranking = []  # 最終排名結果

rank = 1
while remaining_points:
    # 按面積排序，選出最大的
    remaining_points.sort(key=lambda x: x['area'], reverse=True)
    selected_point = remaining_points.pop(0)  # 選出最大面積的點
    final_ranking.append(selected_point)
    
    selected_point_coords = selected_point['point']
    selected_point_polygon = selected_point['shapely_polygon']
    selected_point_2d = ShPoint(selected_point_coords[0], selected_point_coords[1])
    
    print(f"  排名 {rank}: 選出點 {selected_point['index']}，面積 {selected_point['area']:.6f}")
    
    # 檢查並剔除能看到這個點的所有點
    points_to_remove = []
    for point_data in remaining_points:
        point_polygon = point_data['shapely_polygon']
        # 如果選中的點在該點的可見多邊形內，則該點被剔除
        if point_polygon.contains(selected_point_2d):
            points_to_remove.append(point_data)
    
    # 從剩餘點中移除被剔除的點
    for point in points_to_remove:
        remaining_points.remove(point)
    
    print(f"    剔除了 {len(points_to_remove)} 個能看到它的點，剩餘 {len(remaining_points)} 個點")
    rank += 1

print(f"\n最終產生 {len(final_ranking)} 個排名點")

# === 輸出最終排名 ===
print("\n最終排名（逐次剔除可見點）:")
print(f"{'排名':<5} {'點索引':<8} {'XY座標':<25} {'可見面積':<12} {'面積比':<12} {'百分比':<10}")
print("=" * 85)

for rank, point_data in enumerate(final_ranking, 1):
    x, y = point_data['point'][:2]
    tag = " [SELECTED]" if rank == 1 else ""
    print(f"{rank:<5} {point_data['index']:<8} ({x:7.2f}, {y:7.2f}){'':<8} {point_data['area']:<12.6f} {point_data['area_ratio']:<12.6f} {point_data['area_ratio']*100:>8.2f}%{tag}")

# === 可視化前 6 名的點 ===
print("\n=== 可視化新排名前 6 名的點 ===")
viewer = Viewer()

# 添加骨架
for edge in Skeleton.edges():
    line = Skeleton.edge_line(edge)
    if Skeleton.edge_attribute(edge, "inner_bisector"):
        viewer.scene.add(line, linecolor=(0.5, 0.5, 0.5), linewidth=1)
    elif Skeleton.edge_attribute(edge, "bisector"):
        viewer.scene.add(line, linecolor=(0.3, 0.3, 0.3), linewidth=0.5)

for pt in random_points:
    viewer.scene.add(cg.Point(*pt), pointcolor=(0.8, 0.8, 0.8), pointsize=1.5)

# 準備要可視化的前 6 個點：從最終排名中選取
top_6_points_to_display = final_ranking[:min(6, len(final_ranking))]

# 定義顏色
colors = [
    (1.0, 0.0, 0.0),  # 紅色 - 排名 1
    (1.0, 0.5, 0.0),  # 橙色 - 排名 2
    (1.0, 1.0, 0.0),  # 黃色 - 排名 3
    (0.0, 1.0, 0.0),  # 綠色 - 排名 4
    (0.0, 0.0, 1.0),  # 藍色 - 排名 5
    (1.0, 0.0, 1.0),  # 紫色 - 排名 6
]

print(f"\n可視化最終排名前 {len(top_6_points_to_display)} 名的點及其可見多邊形：")
for rank, (point_data, color) in enumerate(zip(top_6_points_to_display, colors), 1):
    test_point = point_data['point']
    area = point_data['area']
    
    tag = " [SELECTED]" if rank == 1 else ""
    print(f"  排名 {rank}: 點 {point_data['index']} - 面積: {area:.6f}{tag}")
    
    # 添加測試點（大號標記）
    viewer.scene.add(
        cg.Point(*test_point),
        pointcolor=color,
        pointsize=20
    )
    
    # 添加可見多邊形邊界
    poly_points = point_data['polygon_points']
    for i in range(len(poly_points)):
        p1 = poly_points[i]
        p2 = poly_points[(i + 1) % len(poly_points)]
        edge_line = cg.Line(
            cg.Point(p1[0], p1[1], 0.05),
            cg.Point(p2[0], p2[1], 0.05)
        )
        viewer.scene.add(edge_line, linecolor=color, linewidth=2)
    
    # 添加多邊形頂點
    for pt in poly_points:
        viewer.scene.add(
            cg.Point(pt[0], pt[1], 0.05),
            pointcolor=color,
            pointsize=6
        )

# === 輸出圖例 ===
print(f"\n{'='*70}")
print("可視化圖例：")
print(f"{'='*70}")
print(f"  排名 1 - 紅色大點 (最大面積: {final_ranking[0]['area']:.6f}) [SELECTED]")
for rank, (color_name, point_data) in enumerate(zip(['橙色', '黃色', '綠色', '藍色', '紫色'], final_ranking[1:6]), 2):
    print(f"  排名 {rank} - {color_name}大點 (面積: {point_data['area']:.6f})")
print(f"\n  彩虹色細線與小點 = 所有 {len(point_results)} 個點的可見多邊形")
print(f"  灰色小點 = 所有隨機採樣點")
print(f"  灰色線 = 骨架結構")
print(f"\n  排名方式: 逐次選出最大面積的點，並剔除能看到它的所有點")
print(f"  共產生 {len(final_ranking)} 個排名點")
viewer.show()

