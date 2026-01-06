import compas.datastructures as cd
import compas.geometry as cg
import compas_cgal.straight_skeleton_2 as skeleton
import pathlib
import numpy as np
from shapely.geometry import Polygon, LineString, Point as ShPoint
from shapely.ops import unary_union
from compas_viewer import Viewer
from compas_viewer.scene import Tag
import isovist
import buildSkeleton as skel

# === Load Meshes ===
filepath_road = pathlib.Path(__file__).parent / "Roads.stl"
filepath_block = pathlib.Path(__file__).parent / "Block.stl"

RoadMesh = cd.Mesh.from_stl(filepath_road)
BlockMesh = cd.Mesh.from_stl(filepath_block)

# === Find Boundary Lines ===
BlockLines = isovist.find_boundary_lines(BlockMesh)
RoadLines = isovist.find_boundary_lines(RoadMesh)


Skeleton = skel.costume_straight_skeleton(RoadMesh)


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
outer_poly, hole_polys = skel.mesh_boundaries_to_polygons(RoadMesh)
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
        result = isovist.analyze_point_visibility(
            point=test_point,
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
            vis_polygon = isovist.compute_visibility_polygon(
                points=test_point,
                visible_boundary_coords=result['visible_boundary_coords'],
                boundary_lines=RoadLines,
                angle_offset=0.1,  # ±0.1度
                max_dist=1000.0,
                road_polygon=boundary_polygon
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
    result = isovist.analyze_point_visibility(
        point=test_point,
        skeleton=Skeleton,
        boundary_polygon=boundary_polygon,
        boundary_lines=RoadLines
    )
    
    if result['is_inside'] and result['visible_boundary_coords'] and len(result['visible_boundary_coords']) >= 3:
        vis_polygon = isovist.compute_visibility_polygon(
            points=test_point,
            visible_boundary_coords=result['visible_boundary_coords'],
            boundary_lines=RoadLines,
            angle_offset=0.1,
            max_dist=1000.0,
            road_polygon=boundary_polygon
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
            print(f"  - 比值{vis_polygon['area_ratio']:.6f}")

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