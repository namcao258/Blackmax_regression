import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider

# Parameters for the cylinder
radius = 110      # Radius of the cylinder
height = 200      # Height of the cylinder
theta_resolution = 360  # Resolution along the theta axis
z_resolution = 301       # Resolution along the z axis
contact_depth_max = 45  # Maximum contact depth

# Create arrays for theta and z
theta = np.linspace(0, 2 * np.pi, theta_resolution)
z = np.linspace(0, height, z_resolution)

# Create a meshgrid for theta and z
theta_grid, z_grid = np.meshgrid(theta, z)

# Convert polar coordinates to Cartesian coordinates
x_grid = radius * np.cos(theta_grid)
y_grid = radius * np.sin(theta_grid)

# Initialize the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

# Plot the cylinder surface in Cartesian coordinates
surface = ax.plot_surface(x_grid, y_grid, z_grid, color='cyan', alpha=0.4, edgecolor='gray')

# Plot polar coordinate lines on the base of the cylinder to represent theta
for t in np.linspace(0, 2 * np.pi, 18):  # 12 radial lines for better visualization
    x_theta = radius * np.cos(t)
    y_theta = radius * np.sin(t)
    ax.plot([0, x_theta], [0, y_theta], [0, 0], color='blue', linestyle='--', alpha=0.6)

# Initial values for sliders
init_theta_idx = 0
init_z_idx = 0
init_contact_depth = 0

# Keep references to the previous plotted elements
highlighted_sphere = None
arrow = None

def plot_sphere(center, radius, color):
    """Helper function to plot a sphere at a given center with a given radius."""
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    return ax.plot_surface(x, y, z, color=color, alpha=0.6)

def update_plot(val):
    global highlighted_sphere, arrow
    
    # Remove the previous sphere and arrow if they exist
    if highlighted_sphere:
        highlighted_sphere.remove()
    if arrow:
        arrow.remove()
    
    # Get the current values from sliders
    node_theta_index = int(theta_slider.val)
    node_z_index = int(z_slider.val)
    contact_depth = contact_slider.val
    
    # Normalize the contact depth to get a color intensity
    norm = mcolors.Normalize(vmin=0, vmax=contact_depth_max)
    cmap = plt.get_cmap('Reds')
    node_color = cmap(norm(contact_depth))

    # Get the Cartesian coordinates of the selected node
    node_x = x_grid[node_z_index, node_theta_index]
    node_y = y_grid[node_z_index, node_theta_index]
    node_z = z_grid[node_z_index, node_theta_index]

    # Plot a small sphere to represent the contact depth
    sphere_radius = contact_depth / contact_depth_max * 20  # Sphere radius proportional to contact depth
    highlighted_sphere = plot_sphere(center=(node_x, node_y, node_z), radius=sphere_radius, color=node_color)

    # Add an arrow from the contact point to the center, with length proportional to contact depth
    arrow_length = contact_depth / contact_depth_max * radius
    arrow_dx = -node_x * (arrow_length / radius)
    arrow_dy = -node_y * (arrow_length / radius)
    arrow = ax.quiver(node_x, node_y, node_z, arrow_dx, arrow_dy, 0, color='black', linewidth=1.5)

    fig.canvas.draw_idle()

# Set plot labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Cylinder with Contact Depth Represented by Sphere and Directional Vector')

# Set the view so that theta=0 is in front
ax.view_init(elev=30, azim=0)

# Set aspect ratio for better visualization
ax.set_box_aspect([1, 1, height / radius])

# Add sliders for theta, z, and contact depth
ax_theta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_z = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_contact = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Create sliders
theta_slider = Slider(ax_theta, 'Theta', 0, theta_resolution - 1, valinit=init_theta_idx, valstep=1)
z_slider = Slider(ax_z, 'Z', 0, z_resolution - 1, valinit=init_z_idx, valstep=1)
contact_slider = Slider(ax_contact, 'Contact Depth', 0, contact_depth_max, valinit=init_contact_depth)

# Update the plot when the slider value changes
theta_slider.on_changed(update_plot)
z_slider.on_changed(update_plot)
contact_slider.on_changed(update_plot)

# Show the plot
plt.show()
