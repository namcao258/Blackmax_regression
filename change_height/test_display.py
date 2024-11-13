import pyvista as pv
import numpy as np
from pyvistaqt import BackgroundPlotter

def display_model(file_path, plotter):
    mesh = pv.read(file_path)
    center_bottom = [(mesh.bounds[0] + mesh.bounds[1]) / 2, (mesh.bounds[2] + mesh.bounds[3]) / 2, mesh.bounds[4]]
    mesh.translate([-coord for coord in center_bottom])
    plotter.add_axes()
    plotter.view_isometric()
    return mesh, center_bottom

def get_base_center_coordinates(file_path):
    mesh = pv.read(file_path)
    bounds = mesh.bounds
    return [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, bounds[4]]

def add_red_dot_at_base_center(plotter, center_bottom):
    plotter.add_text("Visualization 3D View", font_size=10)
    plotter.add_points(pv.PolyData([center_bottom]), color="black", point_size=15)

def add_coordinate_axes(plotter, center_bottom, axis_lengthX, axis_lengthY, axis_lengthZ):
    plotter.add_mesh(pv.Line(center_bottom, [center_bottom[0] + axis_lengthX, center_bottom[1], center_bottom[2]]), color="red", line_width=5)
    plotter.add_mesh(pv.Line(center_bottom, [center_bottom[0], center_bottom[1] + axis_lengthY, center_bottom[2]]), color="green", line_width=5)
    plotter.add_mesh(pv.Line(center_bottom, [center_bottom[0], center_bottom[1], center_bottom[2] + axis_lengthZ]), color="blue", line_width=5)

def add_blank_box(plotter, depth,z,theta,height):
    plotter.add_text(f"Depth: {round(depth,2)}", position=(10, 600), font_size=10, color="black", name="depth_text")
    plotter.add_text(f"Z: {round(z,2)}", position=(10, 570), font_size=10, color="black", name="z_text")
    plotter.add_text(f"Theta: {round(theta,2)}", position=(10, 540), font_size=10, color="black", name="theta_text")
    plotter.add_text(f"Height: {round(height,2)}", position=(10, 510), font_size=10, color="black", name="height_text")

def initialize_plotter_and_model(file_path):
    plotter = BackgroundPlotter(shape=(1, 2))
    mesh_original, center_bottom = display_model(file_path, plotter)
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_original, color="black", show_edges=True, opacity=0.2)
    add_red_dot_at_base_center(plotter, center_bottom)
    add_coordinate_axes(plotter, center_bottom, axis_lengthX=110, axis_lengthY=110, axis_lengthZ=350)
    return plotter, mesh_original, center_bottom

def update_conic_surface_point(plotter, mesh_original, center_bottom, z, theta, height, depth):
    radius_bot = (mesh_original.bounds[1] - mesh_original.bounds[0]) / 2
    radius_top = 50
    theta_radian = np.radians(theta)
    radius_at_z = radius_bot + (z / height) * (radius_top - radius_bot)
    x = center_bottom[0] + radius_at_z * np.cos(theta_radian)
    y = center_bottom[1] + radius_at_z * np.sin(theta_radian)
    point = np.array([x, y, z])
    vector_length = depth
    vector = np.array([vector_length * np.cos(theta_radian), vector_length * np.sin(theta_radian), 0])
    new_point = point - vector
    print(point)
    print(new_point)
    print(0)
    point_polydata = pv.PolyData(point.reshape(1, 3))
    new_point_polydata = pv.PolyData(new_point.reshape(1, 3))
    point_actor = plotter.add_points(point_polydata, color="red", point_size=20, opacity=1.0,render_points_as_spheres=True)
    new_point_actor = plotter.add_points(new_point_polydata, color="blue", point_size=0, opacity=0)
    line = pv.Line(point, new_point)
    line_actor = plotter.add_mesh(line, color="green", line_width=10)
    vector = point -new_point
    arrow = pv.Arrow(start=new_point, direction=-vector, tip_length=10, tip_radius=10)
    arrow_actor = plotter.add_mesh(arrow, color="orange", line_width=2, opacity=1.0)
    plotter.remove_actor(new_point_actor)
    plotter.remove_actor(line_actor)
    plotter.remove_actor(point_actor)
    plotter.remove_actor(arrow_actor)
    add_blank_box(plotter,depth,z,theta,height)
    plotter.update()

def display_model_with_camera(height, z, theta, depth):
    update_conic_surface_point(plotter, mesh_original, center_bottom, z, theta, height, depth)
    

def display_predict(x, z, theta, height):
    print("Predicted Output (Original Scale):")
    print(f"x_position: {x}")
    print(f"z_position: {z}")
    print(f"rotation_angle: {theta}")
    print(f"robot_height: {height}")

file_path = "/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/skin_5.vtk"
plotter, mesh_original, center_bottom = initialize_plotter_and_model(file_path)

# while True:
#     z = random.uniform(0, 300)
#     height = random.uniform(0, 300)
#     theta = random.uniform(0, 360)
#     depth = random.uniform(0, 45)

#     display_model_with_camera(file_path, z, theta, height, depth)
    # display_predict(depth, z, theta, height)
