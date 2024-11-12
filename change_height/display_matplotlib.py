import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def display_model_with_camera(file_path, height, z, theta, depth):
    fig = plt.figure(figsize=(12, 6))

    # Hiển thị mô hình 3D trong subplot (0, 0)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_box_aspect([1, 1, 1])

    # Tạo mô hình giả lập (mô phỏng với một hình hộp)
    bounds = [0, 100, 0, 100, 0, 50]  # Giả sử đây là bounds của mô hình
    x = np.linspace(bounds[0], bounds[1], 10)
    y = np.linspace(bounds[2], bounds[3], 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # Z là chiều cao của mô hình

    ax1.plot_surface(X, Y, Z, color='lightblue', alpha=0.5, rstride=100, cstride=100)

    # Thêm điểm đen ở gốc đáy
    ax1.scatter(50, 50, 0, color="black", s=100)

    # Vẽ trục tọa độ
    ax1.quiver(50, 50, 0, 10, 0, 0, color="red", length=20, linewidth=2)
    ax1.quiver(50, 50, 0, 0, 10, 0, color="green", length=20, linewidth=2)
    ax1.quiver(50, 50, 0, 0, 0, 10, color="blue", length=20, linewidth=2)

    # Vẽ vòng tròn tại đáy mô hình
    circle_points = np.array([[50 + 20 * np.cos(i * 2 * np.pi / 24),
                               50 + 20 * np.sin(i * 2 * np.pi / 24), 0] for i in range(24)])
    for i in range(24):
        ax1.plot([circle_points[i][0], circle_points[(i + 1) % 24][0]],
                 [circle_points[i][1], circle_points[(i + 1) % 24][1]],
                 [circle_points[i][2], circle_points[(i + 1) % 24][2]], color="purple", linewidth=2)

    # Vẽ đường tròn ở trên cùng của mô hình
    circle_points_top = np.array([[50 + 20 * np.cos(i * 2 * np.pi / 24),
                                   50 + 20 * np.sin(i * 2 * np.pi / 24), 50] for i in range(24)])
    for i in range(24):
        ax1.plot([circle_points_top[i][0], circle_points_top[(i + 1) % 24][0]],
                 [circle_points_top[i][1], circle_points_top[(i + 1) % 24][1]],
                 [circle_points_top[i][2], circle_points_top[(i + 1) % 24][2]], color="green", linewidth=2)

    # Vẽ các điểm và đường nối cho mô hình hình nón
    radius_bottom = 20
    radius_top = 10
    for i in range(24):
        x = 50 + radius_bottom * np.cos(i * 2 * np.pi / 24)
        y = 50 + radius_bottom * np.sin(i * 2 * np.pi / 24)
        z = 0
        ax1.scatter(x, y, z, color="red", s=30)

    # Hiển thị giá trị dự đoán
    print("Predicted Output (Original Scale):")
    print(f"x_position: {50-20}")
    print(f"z_position: {0}")
    print(f"rotation_angle: {theta}")
    print(f"robot_height: {height}")

    # Hiển thị camera view trong subplot (0, 1)
    ax2 = fig.add_subplot(122)
    ax2.imshow(np.random.random((200, 200)), cmap='gray')  # Giả sử là một ảnh ngẫu nhiên
    ax2.set_title("Camera View")
    ax2.axis('off')  # Tắt các trục

    plt.show()

# Gọi hàm với các tham số
file_path = "/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/skin_5.vtk"
display_model_with_camera(file_path, 270, 150, 270, 10)
