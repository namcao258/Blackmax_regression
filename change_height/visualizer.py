import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import matplotlib.pyplot as plt
import pandas as pd
from inference import config

class TactileVisualize():
    def __init__(self, skin_path, init_position, depth_range=[-5, 100]):
        self.skin_path = skin_path
        self.init_position = np.array(init_position)  # Input as a single point
        self.deviations = np.zeros(3)  # Running deviations for a single point
        self.depth_range = depth_range
        self.plot_initialize()

    def plot_initialize(self):
    # Set camera position
        self.cpos = np.array([[-57.34826013561122, -661.242740117575, 352.06298028880514],
                            [-0.056080635365844955, -1.209135079294247, 118.18100834810429],
                            [0.024125960714330284, 0.3320410642889663, 0.9429563455672065]])
        
        self.plotter = pvqt.BackgroundPlotter()
        self.plotter.set_background("white", top="white")
        pv.global_theme.font.color = 'black' 
        pv.global_theme.font.title_size = 16 
        pv.global_theme.font.label_size = 16  
        boring_cmap = plt.cm.get_cmap("bwr")  
        
        self.plotter.subplot(0, 0)
        self.plotter.camera_position = self.cpos
        self.plotter.show_axes()
        self.skin_est = pv.read(self.skin_path)  # For PyVista visualization
        
        # Sửa lại để tạo một mảng có số phần tử bằng số điểm trong mesh
        norm_deviations = np.linalg.norm(self.deviations)
        
        # Tạo một mảng với giá trị norm_deviations cho mỗi điểm trong skin_est
        depth_values = np.full(self.skin_est.n_points, norm_deviations)
        
        # Gán mảng depth_values cho thuộc tính 'contact depth (unit:mm)'
        self.skin_est['contact depth (unit:mm)'] = depth_values  # For contact depth visualization
        
        self.plotter.add_mesh(self.skin_est, cmap=boring_cmap, clim=self.depth_range)



    def update_plot(self, deformation, positions):
        # Kiểm tra nếu deformation là một giá trị duy nhất và tạo mảng với số phần tử bằng số điểm
        if np.isscalar(deformation):
            deformation = np.full(self.skin_est.n_points, deformation)
        
        # Gán mảng deformation cho thuộc tính 'contact depth (unit:mm)'
        self.skin_est['contact depth (unit:mm)'] = deformation  # For contact depth visualization
        
        # Cập nhật các điểm trong mesh với positions mới
        self.skin_est.points = positions

