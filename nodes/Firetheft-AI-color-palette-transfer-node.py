import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
import torch.nn.functional as F
import re
import ast
from scipy.cluster.vq import vq
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def Hex_to_RGB(inhex: str) -> tuple:
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code: {inhex}')
    else:
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        rgb = (int(rval, 16), int(gval, 16), int(bval, 16))
    return tuple(rgb)

def parse_color_palette(color_palette: str):
    hex_pattern = r'#([A-Fa-f0-9]{6})'
    rgb_pattern = r'\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})\)'

    if re.search(hex_pattern, color_palette):
        hex_colors = [color.strip() for color in color_palette.split(',')]
        return [Hex_to_RGB(color) for color in hex_colors]

    elif re.search(rgb_pattern, color_palette):
        rgb_colors_str = f"[{color_palette}]"
        return ast.literal_eval(rgb_colors_str)

    else:
        raise ValueError("Invalid color palette format. Please use hex format '#RRGGBB' or RGB format '(R, G, B)'.")


class ColorPaletteTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        data_in = {
            "required": {
                "image": ("IMAGE",),
                "cluster_method": (["Kmeans", "Mini batch Kmeans"], {'default': 'Kmeans'}),
                "distance_method": (["Euclidean", "Manhattan"], {'default': 'Euclidean'}),
                # 增加了随机打乱模式
                "match_mode": (["Force All Colors (1-to-1)", "Closest Match (Many-to-1)", "Random Shuffle (1-to-1)"], {'default': 'Force All Colors (1-to-1)'}),
                # 新增了和选择器一样的随机种子参数
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # 插画风格强度：0为关闭，1-10为不同程度的平滑和块面融合
                "illustration_level": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            },
            "optional": {
                "color_palette": ("STRING", {"forceInput": True}),
            }
        }
        return data_in

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_transfer"
    CATEGORY = "📜Firetheft AI Tools"

    def color_transfer(self, image, cluster_method, distance_method, match_mode, seed, illustration_level=0, color_palette="", **kwargs):
        if not color_palette:
            return (image,)

        target_colors = parse_color_palette(color_palette)

        if len(target_colors) == 0:
            return (image,)

        # 固定 numpy 的全局随机种子，确保整个抽卡过程是受控且可复现的
        # numpy 的 seed 最大值是 2**32 - 1，所以对超大 seed 做了取模处理
        np_seed = seed % (2**32)
        np.random.seed(np_seed)

        target_colors_np = np.array(target_colors)
        num_targets = len(target_colors)
        processedImages = []
        
        # 降采样提速参数
        scale_factor = 0.5
        sampling_rate = 0.25

        for img in image:
            H, W, C = img.shape
            
            # 1. 降采样
            img_tensor = img.unsqueeze(0).permute(0, 3, 1, 2)
            resized_img = F.interpolate(img_tensor, scale_factor=scale_factor, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).squeeze(0)
            pixels_for_clustering = resized_img.reshape(-1, C).cpu().numpy() * 255.0
            
            # 2. 随机像素采样（现在受到 seed 控制，采样更加稳定）
            num_samples = int(pixels_for_clustering.shape[0] * sampling_rate)
            if num_samples > 0:
                random_indices = np.random.choice(pixels_for_clustering.shape[0], size=num_samples, replace=False)
                pixels_for_clustering = pixels_for_clustering[random_indices]

            # 3. 极速提取原图主色（传入 random_state=np_seed 保证 Kmeans 结果可复现）
            cluster_methods = {
                "Kmeans": KMeans,
                "Mini batch Kmeans": MiniBatchKMeans
            }
            clustering_model = cluster_methods.get(cluster_method)(n_clusters=num_targets, n_init='auto', random_state=np_seed)
            clustering_model.fit(pixels_for_clustering)
            detected_colors = clustering_model.cluster_centers_

            # 4. 颜色匹配逻辑
            if match_mode == "Force All Colors (1-to-1)":
                # 计算距离矩阵（匈牙利算法最优解）
                metric = 'euclidean' if distance_method == "Euclidean" else 'cityblock'
                cost_matrix = cdist(detected_colors, target_colors_np, metric=metric)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                closest_colors = np.zeros_like(detected_colors)
                closest_colors[row_ind] = target_colors_np[col_ind]

            elif match_mode == "Random Shuffle (1-to-1)":
                # 无视色彩距离，直接用 seed 将目标颜色随机打乱后强制分配！
                shuffled_targets = np.random.permutation(target_colors_np)
                closest_colors = shuffled_targets[:len(detected_colors)]
                
            else:
                # 原始的贪心匹配算法（可能丢失颜色）
                closest_colors = []
                for color in detected_colors:
                    if distance_method == "Euclidean":
                        distances = np.linalg.norm(color - target_colors_np, axis=1)
                    else:
                        distances = np.sum(np.abs(color - target_colors_np), axis=1)
                    closest_colors.append(target_colors_np[np.argmin(distances)])
                closest_colors = np.array(closest_colors)

            # 5. 使用 vq 极速分配全分辨率图像的每一个像素
            original_img_np = (img.cpu().numpy() * 255.0).astype(np.uint8)
            edges_mask = None
            
            if illustration_level > 0:
                try:
                    import cv2
                    # 核心新增：提取边缘保护蒙版（在被双边滤波平滑前拦截最原始的线条细节）
                    gray = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2GRAY)
                    
                    # 使用较敏感的 Canny 阈值提取结构线
                    edges = cv2.Canny(gray, 30, 100)
                    
                    # 对线条边缘进行轻微膨胀保护（随过滤强度增大而扩大保护区，严防中值滤波从两侧侵蚀断线）
                    dilate_iter = 1 if illustration_level <= 3 else 2
                    edges_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilate_iter) > 0

                    # 动态计算双边滤波强度。Level 1-10 映射到 sigma 15-150
                    sigma = int(15 * illustration_level)
                    # 提前做一次双边滤波平滑，统一色块，保留边缘，有助于后续分类
                    original_img_np = cv2.bilateralFilter(original_img_np, d=9, sigmaColor=sigma, sigmaSpace=sigma)
                except ImportError:
                    pass

            original_pixels = original_img_np.reshape(-1, C).astype(np.float32)
            labels, _ = vq(original_pixels, detected_colors)
            
            if illustration_level > 0:
                try:
                    import cv2
                    # 动态计算中值滤波的核大小，确保是奇数（Level 1 -> 3, Level 2 -> 5, ..., Level 10 -> 21）
                    k_size = int(illustration_level * 2 + 1)
                    
                    # 把 labels 转为 2D 图像进行中值滤波，消除孤岛和毛刺
                    label_img = labels.reshape((H, W)).astype(np.uint8)
                    label_img_filtered = cv2.medianBlur(label_img, k_size)
                    
                    # 核心改动：用边缘蒙版无伤保护线条！
                    if edges_mask is not None:
                        # 仅在非边缘（面区域）应用强烈的中值滤波融化斑块...
                        # 而原有的结构线条（edges_mask 为真处）强制使用第一手原始调色板分配索引，绝对不断线！
                        label_img = np.where(edges_mask, label_img, label_img_filtered)
                    else:
                        label_img = label_img_filtered

                    labels = label_img.flatten()
                except ImportError:
                    pass

            
            # 6. 生成新图
            new_pixels = closest_colors[labels]
            new_img = new_pixels.reshape((H, W, C)).astype(np.float32) / 255.0
            processedImages.append(torch.from_numpy(new_img).unsqueeze(0))

        output = torch.cat(processedImages, dim=0)
        return (output, )

NODE_CLASS_MAPPINGS = {
    "ColorPaletteTransferNode": ColorPaletteTransferNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPaletteTransferNode": "Color Palette Transfer（调色板应用器）"
}