from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'acit4820_project'
data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.*')),
        (os.path.join('share', package_name, 'rviz'  ), glob('rviz/*.*'  )),
        (os.path.join('share', package_name, 'urdf'  ), glob('urdf/*.*'  )),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.*')),
    ]

# Recursively include all model directories
for root, _, files in os.walk('models'):
    if files:
        # The destination path inside the install directory
        install_dir = os.path.join('share', package_name, root)
        # The source file paths
        files_list = [os.path.join(root, f) for f in files]
        data_files.append((install_dir, files_list))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='james',
    maintainer_email='james@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_node            = acit4820_project.aruco_node:main',
            'bluerov_pose_node     = acit4820_project.bluerov_pose_node:main',
            'thruster_manager_node = acit4820_project.thruster_manager_node:main',
            'rov_control_node      = acit4820_project.rov_control_node:main',
        ],
    },
)
# maybe add current node in the future
