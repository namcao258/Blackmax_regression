import pyvista as pv
import numpy as np

class FrustumCone:
    def __init__(self, height, radius_base, radius_top, sphere_radius, stl_path=None):
        # Initialize the cone with given height and radii, and optional STL path
        self.height = height
        self.radius_base = radius_base
        self.radius_top = radius_top
        self.sphere_radius = sphere_radius
        self.stl_path = stl_path  # Path to the STL file (optional)
        self.cone = pv.Cone(height=self.height, radius=self.radius_base, direction=(0, 0, 1))
        
    def plot(self, z, theta_deg):
        # Step 1: Calculate the 3D coordinates of a point on the frustum at height z
        theta_rad = np.radians(theta_deg)  # Convert the angle to radians
        r = self.radius_top + (self.radius_base - self.radius_top) * (z / self.height)  # Calculate radius at height z
        x = r * np.cos(theta_rad)  # X-coordinate based on radius and angle
        y = r * np.sin(theta_rad)  # Y-coordinate based on radius and angle

        # Step 2: Transform the cone and adjust its height
        self.cone.scale([1, 1, 1])  # Scale cone to match given dimensions
        self.cone.points[:, 2] += self.height  # Move the bottom to the correct height
        self.cone.points[:, 2] -= self.height / 2  # Adjust the top part of the cone
        
        # Step 3: Initialize the plotter for visualization
        plotter = pv.Plotter()
        plotter.add_mesh(self.cone, color="lightblue", opacity=0.1)  # Add the cone to the plot
        plotter.add_axes()  # Add axes for orientation
        plotter.show_bounds(grid='front', location='outer', all_edges=True, xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')

        # Step 4: Plot the specific point on the cone (at height z)
        points = pv.PolyData([[x, y, z]])
        plotter.add_points(points, color='red', point_size=10)

        # Step 5: Calculate and plot the points for the base circle
        theta_base = np.linspace(0, 2*np.pi, 100)  # Create angles for the base circle
        x_base = self.radius_base * np.cos(theta_base)  # X-coordinates of base
        y_base = self.radius_base * np.sin(theta_base)  # Y-coordinates of base
        z_base = np.zeros_like(theta_base)  # Z-coordinates of base (all at height 0)
        base_points = np.vstack((x_base, y_base, z_base)).T  # Stack the points together
        plotter.add_points(base_points, color='green', point_size=5)  # Add base points to the plot

        # Step 6: Calculate and plot the points for the top circle
        theta_top = np.linspace(0, 2*np.pi, 100)  # Create angles for the top circle
        x_top = self.radius_top * np.cos(theta_top)  # X-coordinates of top
        y_top = self.radius_top * np.sin(theta_top)  # Y-coordinates of top
        z_top = np.full_like(theta_top, self.height)  # Z-coordinates of top (all at height)
        top_points = np.vstack((x_top, y_top, z_top)).T  # Stack the points together
        plotter.add_points(top_points, color='blue', point_size=5)  # Add top points to the plot

        # Step 7: Draw line between the center of the base and the center of the top
        line = pv.Line([0, 0, 0], [0, 0, self.height])  # Create line from base center to top center
        plotter.add_mesh(line, color="blue", line_width=5)  # Add the center line to the plot

        # Step 8: Draw lines connecting each point on the base circle to the corresponding point on the top circle
        self.plot_lines_between_circles(plotter, x_base, y_base, z_base, x_top, y_top, z_top)

        # Step 9: Add a sphere on top of the cone at the top center
        sphere_center = [0, 0, self.height]  # Set the center of the sphere at the top of the cone
        sphere = pv.Sphere(radius=self.sphere_radius, center=sphere_center)  # Create the sphere at the center
        plotter.add_mesh(sphere, color="red")  # Add the sphere to the plot

        # Step 10: Add STL mesh if the path is provided
        if self.stl_path:
            self.add_stl_to_plot(plotter)

        # Step 11: Display the plot with all elements
        plotter.show()

    def plot_lines_between_circles(self, plotter, x_base, y_base, z_base, x_top, y_top, z_top):
        # Step 12: Loop through each point on the base and top circles and draw lines between them
        for i in range(len(x_base)):
            # Create a line between corresponding points on the base and top
            line = pv.Line([x_base[i], y_base[i], z_base[i]], [x_top[i], y_top[i], z_top[i]])
            plotter.add_mesh(line, color="purple", line_width=2)  # Add the connecting line to the plot

    def add_stl_to_plot(self, plotter):
        # Step 13: Load and position the STL file if the path is provided
        stl_mesh = pv.read(self.stl_path)  # Load the STL mesh from the path
        
        # Step 14: Scale and position the STL mesh to wrap around the cone and sphere
        stl_mesh.scale([1, 1, 1])  # Adjust the scale as needed
        stl_mesh.translate([0, 0, self.height / 2])  # Adjust the position to cover the cone and sphere

        # Step 15: Add the STL mesh as a cover on top of the cone and sphere
        plotter.add_mesh(stl_mesh, color="orange", opacity=0.5)  # Use opacity for better view

# Step 16: Create the FrustumCone object with specified parameters, including sphere radius and STL path
stl_file_path = '/media/namcao/NAM CAO/Ubuntu/Blackmax_regression/change_height/skin_5.STL'  # Replace with your STL file path
frustum_cone = FrustumCone(height=250, radius_base=100, radius_top=30, sphere_radius=20, stl_path=stl_file_path)

# Step 17: Call the plot method to display the frustum and plot the lines (with z=200 and theta_deg=0)
frustum_cone.plot(z=200, theta_deg=0)
