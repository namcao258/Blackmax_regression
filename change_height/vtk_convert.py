import pyvista as pv

# Đọc file STL
mesh_original = pv.read("/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/skin_5.STL")

# Lưu dưới định dạng VTK
mesh_original.save("/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/skin_5.vtk")
