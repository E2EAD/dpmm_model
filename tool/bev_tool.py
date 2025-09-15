import numpy as np


def calculate_pixel_per_meter(camera_params):
    """
    计算俯视相机单位像素对应的实际距离（米/像素）

    参数:
    camera_params (dict): 包含相机参数的字典，格式如下:
        {
            "location": [x, y, z],
            "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "world2cam": 4x4变换矩阵,
            "rotation": [pitch, yaw, roll]  # 可选，用于倾斜修正
        }

    返回:
    float: 单位像素代表的实际距离（米/像素）
    """
    try:
        # 解析相机参数
        h = camera_params["location"][2]  # 相机高度（z坐标）
        intrinsic = np.array(camera_params["intrinsic"])
        fx = intrinsic[0, 0]  # x轴焦距
        fy = intrinsic[1, 1]  # y轴焦距

        # 基础计算：高度/焦距
        meter_per_pixel_x = h / fx
        meter_per_pixel_y = h / fy

        # 检查相机倾斜情况（如果有旋转参数）
        if "rotation" in camera_params:
            # 提取旋转角度（通常为俯仰角pitch）
            pitch = camera_params["rotation"][0]

            # 计算倾斜修正因子（90°表示完全垂直）
            tilt_angle = 90 + pitch  # 实际倾斜角=90°+俯仰角
            tilt_rad = np.deg2rad(tilt_angle)
            correction_factor = 1 / np.cos(tilt_rad)

            # 应用倾斜修正
            meter_per_pixel_x *= correction_factor
            meter_per_pixel_y *= correction_factor

        # 返回平均值（通常x/y方向值接近）
        return 1 / ((meter_per_pixel_x + meter_per_pixel_y) / 2.0)

    except KeyError as e:
        raise ValueError(f"缺少必要的相机参数: {e}")
    except Exception as e:
        raise RuntimeError(f"计算错误: {e}")


def calculate_meter_per_pixel(camera_params):
    """
    计算俯视相机单位像素对应的实际距离（米/像素）

    参数:
    camera_params (dict): 包含相机参数的字典，格式如下:
        {
            "location": [x, y, z],
            "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "world2cam": 4x4变换矩阵,
            "rotation": [pitch, yaw, roll]  # 可选，用于倾斜修正
        }

    返回:
    float: 单位像素代表的实际距离（米/像素）
    """
    try:
        # 解析相机参数
        h = camera_params["location"][2]  # 相机高度（z坐标）
        intrinsic = np.array(camera_params["intrinsic"])
        fx = intrinsic[0, 0]  # x轴焦距
        fy = intrinsic[1, 1]  # y轴焦距

        # 基础计算：高度/焦距
        meter_per_pixel_x = h / fx
        meter_per_pixel_y = h / fy

        # 检查相机倾斜情况（如果有旋转参数）
        if "rotation" in camera_params:
            # 提取旋转角度（通常为俯仰角pitch）
            pitch = camera_params["rotation"][0]

            # 计算倾斜修正因子（90°表示完全垂直）
            tilt_angle = 90 + pitch  # 实际倾斜角=90°+俯仰角
            tilt_rad = np.deg2rad(tilt_angle)
            correction_factor = 1 / np.cos(tilt_rad)

            # 应用倾斜修正
            meter_per_pixel_x *= correction_factor
            meter_per_pixel_y *= correction_factor

        # 返回平均值（通常x/y方向值接近）
        return (meter_per_pixel_x + meter_per_pixel_y) / 2.0

    except KeyError as e:
        raise ValueError(f"缺少必要的相机参数: {e}")
    except Exception as e:
        raise RuntimeError(f"计算错误: {e}")
    

def calculate_pixel_per_meter_v2(camera_config):
    """
    根据完整相机配置字典计算单位像素对应的实际距离
    
    参数:
        camera_config (dict): 相机配置字典，包含以下键:
            - intrinsic: 3x3内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            - location: [x, y, z] 相机在世界坐标系中的位置（米）
    
    返回:
        float: 单位像素对应的实际距离（米/像素）
    
    抛出:
        KeyError: 如果配置字典缺少必要键
        ValueError: 如果内参矩阵格式错误
    """
    # 验证必要键值
    required_keys = ['intrinsic', 'location']
    for key in required_keys:
        if key not in camera_config:
            raise KeyError(f"Missing required key '{key}' in camera_config")

    # 提取参数
    intrinsic = camera_config['intrinsic']
    location = camera_config['location']
    
    # 验证内参矩阵格式
    if len(intrinsic) != 3 or any(len(row) != 3 for row in intrinsic):
        raise ValueError("Intrinsic matrix must be a 3x3 matrix")
    
    # 提取焦距（像素单位）
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    
    # 计算平均焦距（处理可能的微小差异）
    focal_length_pixels = (fx + fy) / 2
    
    # 获取相机高度（米）
    camera_height = location[2]
    
    # 计算像素/米
    return  focal_length_pixels / camera_height


# 示例用法
if __name__ == "__main__":
    # 使用问题中的相机参数
    camera_config = {
        "location": [46.35805892944336, 207.41505432128906, 50.14830017089844],
        "rotation": [-89.10604858398438, -0.015657298266887665, -0.12697848677635193],
        "intrinsic": [
            [560.1660305677678, 0.0, 800.0],
            [0.0, 560.1660305677678, 450.0],
            [0.0, 0.0, 1.0]
        ],
        "world2cam": [
            [0.015601754188537598, -3.457654747762717e-05, -0.9998782873153687, 49.42610168457031],
            [0.002489428035914898, 0.9999969005584717, 4.263523806002922e-06, -207.530029296875],
            [0.9998751878738403, -0.0024891914799809456, 0.015601791441440582, -46.618377685546875],
            [0.0, 0.0, 0.0, 1.0]
        ]
    }

    ppm = calculate_pixel_per_meter(camera_config)
    print(f"单位像素距离: {ppm:.5f} 像素/米")
    ppm = calculate_pixel_per_meter_v2(camera_config)
    print(f"单位像素距离: {ppm:.5f} 像素/米")