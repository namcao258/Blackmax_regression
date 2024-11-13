# Đảm bảo đã cài đặt tất cả thư viện trước
from visualizer import TactileVisualize
import numpy as np

# Khởi tạo đối tượng với tệp skin và tọa độ ban đầu
skin_path = "/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/skin_5.STL"  # Đặt đường dẫn chính xác đến tệp skin của bạn
init_position = [0.0, 0.0, 0.0]  # Ví dụ về tọa độ ban đầu (x, y, z)

# Khởi tạo đối tượng TactileVisualize
visualizer = TactileVisualize(skin_path, init_position)

# Cập nhật vị trí và độ sâu (deformation) khi cần
new_position = [1.0, 2.0, 3.0]  # Tọa độ mới cho điểm
deformation = 10.0  # Độ biến dạng (depth)

# Cập nhật đồ họa với dữ liệu mới
visualizer.update_plot(deformation, new_position)


