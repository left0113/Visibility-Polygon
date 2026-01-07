import compas.datastructures as cd
import compas.geometry as cg
import pathlib
from shapely.geometry import Polygon, LineString, Point as ShPoint
from shapely.ops import unary_union
from compas_viewer import Viewer
import isovist
import buildSkeleton as skel
import booleanVist

# === Load Road Mesh ===
filepath_road = pathlib.Path(__file__).parent / "Roads.stl"
RoadMesh = cd.Mesh.from_stl(filepath_road)

# === Find Boundary Lines ===
RoadLines = skel.find_boundary_lines(RoadMesh)

# === 製作圖形骨架 ===
Skeleton = skel.costume_straight_skeleton(RoadMesh)

# === 獲取邊界多邊形 ===
outer_poly, hole_polys = skel.mesh_boundaries_to_polygons(RoadMesh)

# === 創建 Viewer ===
viewer = Viewer()

# === 可視化骨架 ===
for edge in Skeleton.edges():
    line = Skeleton.edge_line(edge)
    if Skeleton.edge_attribute(edge, "inner_bisector"):
        viewer.scene.add(line, linecolor=(0.5, 0.5, 0.5), linewidth=1)
    elif Skeleton.edge_attribute(edge, "bisector"):
        viewer.scene.add(line, linecolor=(0.3, 0.3, 0.3), linewidth=0.5)

# === 定義測試點 ===
test_points = [
    (7.8, -4.8, 0),  # 測試點1
    (-0.2, 0.2, 0)   # 測試點2
]

test_colors = [
    (1.0, 1.0, 0.0),  # 黃色
    (1.0, 0.5, 0.0)   # 橙色
]

if outer_poly:
    # 使用 Shapely Polygon
    boundary_polygon = Polygon(outer_poly, holes=hole_polys)
    
    # === 使用函數計算可見多邊形聯集 ===
    result = booleanVist.compute_visibility_union(
        points=test_points,
        skeleton=Skeleton,
        boundary_polygon=boundary_polygon,
        boundary_lines=RoadLines
    )
    
    visibility_polygons = result['visibility_polygons']
    union_polygon = result['union_polygon']
    
    # === 可視化測試點和單個可見多邊形 ===
    for vp in visibility_polygons:
        idx = vp['index']
        test_color = test_colors[idx] if idx < len(test_colors) else (0.5, 0.5, 1.0)
        
        # 可視化測試點
        viewer.scene.add(
            cg.Point(*vp['test_point']),
            pointcolor=test_color,
            pointsize=25
        )
        
        # 可視化單個可見多邊形（半透明邊界）
        poly_points = vp['polygon_points']
        for i in range(len(poly_points)):
            p1 = poly_points[i]
            p2 = poly_points[(i + 1) % len(poly_points)]
            edge_line = cg.Line(
                cg.Point(p1[0], p1[1], 0.05),
                cg.Point(p2[0], p2[1], 0.05)
            )
            viewer.scene.add(edge_line, linecolor=test_color, linewidth=2)
    
    # === 可視化聯集多邊形（使用 Shapely）===
    if union_polygon:
        print(f"\n可視化聯集多邊形...")
        
        if union_polygon.geom_type == 'Polygon':
            # 單一多邊形
            union_coords = list(union_polygon.exterior.coords)
            print(f"  聯集多邊形頂點數: {len(union_coords)}")
            
            for i in range(len(union_coords) - 1):
                p1 = union_coords[i]
                p2 = union_coords[i + 1]
                edge_line = cg.Line(
                    cg.Point(p1[0], p1[1], 0.15),
                    cg.Point(p2[0], p2[1], 0.15)
                )
                viewer.scene.add(edge_line, linecolor=(0.0, 1.0, 0.0), linewidth=4)
            
            # 可視化聯集多邊形的頂點
            for pt in union_coords[:-1]:  # 排除最後一個重複點
                viewer.scene.add(
                    cg.Point(pt[0], pt[1], 0.15),
                    pointcolor=(0.0, 1.0, 0.0),
                    pointsize=8
                )
        
        elif union_polygon.geom_type == 'MultiPolygon':
            # 多個多邊形
            print(f"  聯集包含 {len(union_polygon.geoms)} 個多邊形")
            for geom in union_polygon.geoms:
                union_coords = list(geom.exterior.coords)
                for i in range(len(union_coords) - 1):
                    p1 = union_coords[i]
                    p2 = union_coords[i + 1]
                    edge_line = cg.Line(
                        cg.Point(p1[0], p1[1], 0.15),
                        cg.Point(p2[0], p2[1], 0.15)
                    )
                    viewer.scene.add(edge_line, linecolor=(0.0, 1.0, 0.0), linewidth=4)
        
        # === 輸出圖例 ===
        print(f"\n{'='*70}")
        print("可視化圖例：")
        print(f"{'='*70}")
        for idx, vp in enumerate(visibility_polygons):
            color_name = "黃色" if idx == 0 else ("橙色" if idx == 1 else "藍色")
            print(f"  ● {color_name}大點 = 測試點{idx+1} {vp['test_point'][:2]}")
            print(f"  ━ {color_name}細線 = 測試點{idx+1}的可見多邊形")
        print(f"  ━ 綠色粗線 = 聯集多邊形 (Union)")
        print(f"  ━ 灰色線 = 骨架結構")
        print("="*70 + "\n")

else:
    print("錯誤：無法提取邊界多邊形")

# === 顯示 Viewer ===
print("啟動 Viewer...")
viewer.show()
