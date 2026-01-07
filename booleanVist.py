from shapely.geometry import Polygon, LineString, Point as ShPoint
from shapely.ops import unary_union
import isovist

# === 定義計算多點可見多邊形聯集的函數 ===
def compute_visibility_union(points, skeleton, boundary_polygon, boundary_lines):
    """
    計算多個點的可見多邊形並返回其聯集
    
    Parameters:
    -----------
    test_points : list of tuple
        測試點座標列表 [(x1, y1, z1), (x2, y2, z2), ...]
    skeleton : compas.datastructures.Network
        骨架網絡
    boundary_polygon : shapely.geometry.Polygon
        邊界多邊形
    boundary_lines : list
        邊界線列表
    
    Returns:
    --------
    dict : 包含以下鍵值
        - 'visibility_polygons': list，每個點的可見多邊形資訊（Shapely格式）
        - 'union_polygon': shapely.geometry.Polygon，所有可見多邊形的聯集
        - 'total_union_area': float，聯集總面積
        - 'individual_areas': list，各個可見多邊形的面積
    """
    visibility_polygons = []
    
    print(f"\n{'='*70}")
    print(f"開始計算 {len(points)} 個點的可見多邊形")
    print(f"{'='*70}")
    
    # 對每個測試點進行分析
    for idx, test_point in enumerate(points):
        print(f"\n處理測試點 {idx + 1}: {test_point[:2]}")
        
        # 使用統一分析函數
        result = isovist.analyze_point_visibility(
            point=test_point,
            skeleton=skeleton,
            boundary_polygon=boundary_polygon,
            boundary_lines=boundary_lines
        )
        
        # 如果點在圖形內且找到可見邊界節點
        if result['is_inside'] and result['visible_boundary_coords'] and len(result['visible_boundary_coords']) >= 3:
            vis_polygon = isovist.compute_visibility_polygon(
                points=test_point,
                visible_boundary_coords=result['visible_boundary_coords'],
                boundary_lines=boundary_lines,
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
                    'area_ratio': vis_polygon['area_ratio'],
                    'index': idx,
                    'polygon_points': vis_polygon['polygon_points']
                })
                
                print(f"   可見多邊形計算完成")
                print(f"    - 頂點數: {len(vis_polygon['polygon_points'])}")
                print(f"    - 面積: {vis_polygon['area']:.6f}")
                if vis_polygon['area_ratio'] is not None:
                    print(f"    - 占道路面積比: {vis_polygon['area_ratio']:.6f}")
                print(f"    - 周長: {vis_polygon['perimeter']:.6f}")
            else:
                print(f"無法計算可見多邊形")
        else:
            if not result['is_inside']:
                print(f"點不在邊界內")
            else:
                print(f"找不到足夠的可見邊界節點")
    
    # 執行聯集運算
    union_polygon = None
    total_union_area = 0.0
    individual_areas = []
    
    if len(visibility_polygons) >= 1:
        print(f"\n{'='*70}")
        print(f"執行聯集運算：合併 {len(visibility_polygons)} 個可見多邊形")
        print(f"{'='*70}")
        
        # 收集所有 Shapely 多邊形
        shapely_polygons = [vp['shapely_polygon'] for vp in visibility_polygons]
        individual_areas = [vp['area'] for vp in visibility_polygons]
        
        # 使用 Shapely 計算聯集
        union_polygon = unary_union(shapely_polygons)
        total_union_area = union_polygon.area
        
    else:
        print(f"\n⚠ 警告：沒有有效的可見多邊形")
    
    return {
        'visibility_polygons': visibility_polygons,
        'union_polygon': union_polygon,
        'total_union_area': total_union_area,
        'individual_areas': individual_areas
    }


# === 定義計算覆蓋率的函數 ===
def union_coverage(road_polygon, union_polygon):
    
    road_area = road_polygon.area
    
    if union_polygon is None:
        union_area = 0.0
    else:
        union_area = union_polygon.area
    
    if road_area > 0:
        coverage_ratio = union_area / road_area
    else:
        coverage_ratio = 0.0
    
    coverage_percentage = coverage_ratio * 100
    
    return {
        'road_area': road_area,
        'union_area': union_area,
        'coverage_ratio': coverage_ratio,
        'coverage_percentage': coverage_percentage
    }
