import numpy as np
import open3d as o3d
import pyvista as pv


# Specify the path to your .npy file
file_path = '/home/lj/PycharmProjects/APES/data/modelnet/pcd/test/0001.npy'

# Load the .npy file
data = np.load(file_path)

# Print the loaded data
print(data.shape)
# Create an Open3D point cloud object
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(data)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])


# Create a PyVista point cloud
point_cloud = pv.PolyData(data)

# Plot the point cloud
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, point_size=2, render_points_as_spheres=True)
plotter.show()
