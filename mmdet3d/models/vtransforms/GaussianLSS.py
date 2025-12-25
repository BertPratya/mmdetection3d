import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer



class BEVCamera:
    def __init__(self, x_range, y_range, img_width, img_height):
        """
        For 3d coordinate: +X is forward, +Y is left, +Z is up
        For 2d BEV space: +X is move up (-row), +Y is move left (-col)
        """
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
            
        self.FoVx = self.x_max - self.x_min
        self.FoVy = self.y_max - self.y_min
        
        self.img_width = img_width
        self.img_height = img_height
        self.camera_center = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.set_transform()
        
    def set_transform(self):
        sh = self.img_height / self.FoVx
        sw = self.img_width / self.FoVy
        
        # Matrix M where M @ [x, y, z, 1] = [col, row, z, 1]
        # Row 0 (Col): W/2 - y * scale_y
        # Row 1 (Row): H/2 - x * scale_x
        
        self.full_proj_transform = torch.tensor([
            [0,   -sw, 0, self.img_width / 2.0],  # Col Index
            [-sh,   0, 0, self.img_height / 2.0], # Row Index
            [0,     0, 0, 1],                     
            [0,     0, 0, 1]                      
        ], dtype=torch.float32)
    
        self.world_view_transform = torch.tensor([
            [ 0.,  sw,  0.,         0.],
            [ sh,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
            [ 0.,  0.,  0.,         0.],
        ], dtype=torch.float32)

class GaussianRenderer(nn.Module):
    def __init__(self, embed_dims, grid_width, grid_height, x_range, y_range, threshold=0.05):
        super().__init__()
        self.viewpoint_camera = BEVCamera(x_range, y_range, grid_width, grid_height)
        self.rasterizer = GaussianRasterizer()
        self.embed_dims = embed_dims
        self.threshold = threshold
        self.grid_width, self.grid_height = grid_width, grid_height

    def forward(self, features, means3D, cov3D, opacities):
        """
        features: b G d
        means3D: b G 3
        cov3D:    b G 6
        opacities: b G 1
        """ 
        b = features.shape[0]
        device = means3D.device
        
        bev_out = []
        mask = (opacities > self.threshold)
        mask = mask.squeeze(-1)
        self.set_Rasterizer(device)
        
        for i in range(b):
            if mask[i].sum() > 0:
                rendered_bev, _ = self.rasterizer(
                    means3D=means3D[i][mask[i]],
                    means2D=None,
                    shs=None, 
                    colors_precomp=features[i][mask[i]],
                    opacities=opacities[i][mask[i]],
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D[i][mask[i]]
                )
                bev_out.append(rendered_bev)
            else:
                bev_out.append(torch.zeros((self.embed_dims, self.grid_height, self.grid_width), device=device))
            
        x = torch.stack(bev_out, dim=0) # b d h w
        num_gaussians = (mask.detach().float().sum(1)).mean().cpu()

        return x, num_gaussians
        
    @torch.no_grad()
    def set_Rasterizer(self, device):
        tanfovx = math.tan(self.viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(self.viewpoint_camera.FoVy * 0.5)

        bg_color = torch.zeros((self.embed_dims)).to(device) 
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.viewpoint_camera.img_height),
            image_width=int(self.viewpoint_camera.img_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1,
            viewmatrix=self.viewpoint_camera.world_view_transform.to(device),
            projmatrix=self.viewpoint_camera.full_proj_transform.to(device),
            sh_degree=0,  # No SHs used 
            campos=self.viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=False
        )
        self.rasterizer.set_raster_settings(raster_settings)

@MODELS.register_module()
class GaussianLSSTransform(BaseModule):
    def __init__(self,
                 img_h,
                 img_w,
                 depth_num,
                 depth_start,
                 depth_max,
                 embed_dims,
                 bev_h,
                 bev_w,
                 x_range,
                 y_range,
                 error_tolerance=1.0,
                 opacity_filter=0.05):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.depth_num = depth_num
        self.depth_start = depth_start
        self.depth_max = depth_max
        self.embed_dims = embed_dims
        self.error_tolerance = error_tolerance
        self.opacity_filter = opacity_filter
        
        self.gs_render = GaussianRenderer(embed_dims=embed_dims,
                                          grid_width=bev_w,
                                          grid_height=bev_h,
                                          x_range=x_range,
                                          y_range=y_range,
                                          threshold=opacity_filter)
        
        bins = self.init_bin_centers()
        self.register_buffer('bins', bins, persistent=False)

    def init_bin_centers(self):
        """
        Calculate depth bin centers.
        """
        depth_range = self.depth_max - self.depth_start
        interval = depth_range / self.depth_num
        interval = interval * torch.ones((self.depth_num+1))
        interval[0] = self.depth_start
        bin_edges = torch.cumsum(interval, 0)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers 
    
    def pred_depth(self, lidar2img, depth, img_h, img_w, coords_3d=None):
        if coords_3d is None:
            coords_3d, coords_d = get_pixel_coords_3d(self.bins, depth, lidar2img, img_h, img_w)
            coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
            
        pred_coords_3d = (depth.unsqueeze(-1) * coords_3d).sum(1)
        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale
        return pred_coords_3d, cov

    def forward(self, features, depth, opacity, lidar2img):
        """
        Args:
            features: (B*N, C, H, W) - Semantic/Color features
            depth: (B*N, D, H, W) - Depth probabilities
            opacity: (B*N, 1, H, W) - Opacity
            lidar2img: (B, N, 4, 4) - Calibration matrices
        """
        B, N = lidar2img.shape[:2]
        
        # 1. Lift features to 3D (Calculate Mean and Covariance)
        means3D, cov3D_full = self.pred_depth(lidar2img, depth, self.img_h, self.img_w)

        # 2. Process Covariance for Gaussian Splatting
        cov3D_flat = cov3D_full.flatten(-2, -1) # (B*N, H, W, 9)
        
        # Extract the upper triangular unique elements (xx, xy, xz, yy, yz, zz)
        cov3D = torch.cat((
            cov3D_flat[..., 0:3], 
            cov3D_flat[..., 4:6], 
            cov3D_flat[..., 8:9]
        ), dim=-1) # (B*N, H, W, 6)

        # 3. Reshape all tensors for the Rasterizer
        # (B*N, C, H, W) -> (B, N*H*W, C)
        features = rearrange(features, '(b n) c h w -> b (n h w) c', b=B, n=N)
        
        # (B*N, H, W, 3) -> (B, N*H*W, 3)
        means3D = rearrange(means3D, '(b n) h w d -> b (n h w) d', b=B, n=N)
        
        # (B*N, H, W, 6) -> (B, N*H*W, 6)
        cov3D = rearrange(cov3D, '(b n) h w d -> b (n h w) d', b=B, n=N)
        
        # (B*N, 1, H, W) -> (B, N*H*W, 1)
        opacity = rearrange(opacity, '(b n) c h w -> b (n h w) c', b=B, n=N)
        # Ensure all inputs are float32
        features = features.float()
        means3D = means3D.float()
        cov3D = cov3D.float()
        opacity = opacity.float()
        # 4. Render BEV
        # Returns: x (B, C, BEV_H, BEV_W), num_gaussians
        x, num_gaussians = self.gs_render(features, means3D, cov3D, opacity)
        
        
        
        return x, num_gaussians

# --- Helper Functions ---

@torch.no_grad()
def get_pixel_coords_3d(coords_d, depth, lidar2img, img_h, img_w):
    eps = 1e-5
    B, N = lidar2img.shape[:2]
    H, W = depth.shape[-2:]
    
    # Mapping matrix to map from pixel in depth map to image pixel coordinate
    coords_h = torch.linspace(0, 1, H, device=depth.device).float() * img_h
    coords_w = torch.linspace(0, 1, W, device=depth.device).float() * img_w
    
    num_depth = coords_d.shape[0]
    
    # Generate all points stacking into [u,v,d]
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3, 0)
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    
    # Prevent numerical instability with small depth values
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
    
    img2lidars = lidar2img.inverse()
    
    coords = coords.view(1, 1, W, H, num_depth, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
    img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, num_depth, 1, 1)
    
    # Transform to world coordinates
    # Output shape: B N W H D 3
    coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] 
    
    return coords3d, coords_d

