#!/usr/bin/env python3
# Code taken and readapted from:
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/tree/main
# and from turlab ntnu aruco pose estimation repository
# https://turlab.itk.ntnu.no/turlab/ros2-aruco-pose-estimation/-/blob/main/aruco_pose_estimation/scripts/aruco_node.py?ref_type=heads

# Python imports
import numpy as np
import cv2
import transforms3d
import math
# ROS2 imports
from rclpy.impl import rcutils_logger
from geometry_msgs.msg import Pose

# utils import python code
from acit4820_project.utils import aruco_display

def get_marker_size(marker_id: int) -> float:
    '''
    Returns the marker size (in meters) depending on the marker id
    None if the size is unknown
    '''
    if (marker_id == 0) or (marker_id == 1):
        return 0.05
    elif (marker_id == 2) or (marker_id == 3) or (marker_id == 4) or (marker_id == 5) or (marker_id == 6) or (marker_id == 7):
        return 0.25
    else:
        return None

def pose_estimation(frame, depth_frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, pose_array, markers):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    parameters             = cv2.aruco.DetectorParameters_create()
    corners, marker_ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict_type, parameters=parameters)
    
    if corners is None or len(corners) == 0:
        return frame, pose_array, markers # nothing detected

    for c in corners:
        if np.any(np.isnan(c)) or np.any(np.isinf(c)):
            continue  # skip invalid markers

    
    logger = rcutils_logger.RcutilsLogger(name="aruco_node")

    frame_processed = frame.copy()

    # If markers are detected
    if len(corners) > 0:
        logger.debug("Detected {} markers.".format(len(corners)))

        for i, marker_id_arr in enumerate(marker_ids):
            marker_id = int(marker_id_arr[0]) if hasattr(marker_id_arr, '__len__') else int(marker_id_arr)
            marker_size = get_marker_size(marker_id)
            if marker_size is None:
                logger.warning(f"Marker ID {marker_id} unknown, skipping...")
                continue

            # optionally keep aruco's estimator first (but don't rely on it)
            try:
                # corners[i] is usually shape (1,4,2) or (4,1,2)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners=np.array([corners[i]]),  # ensure expected shape (n_markers, 1, 4, 2) or similar
                    markerLength=marker_size,
                    cameraMatrix=matrix_coefficients,
                    distCoeffs=distortion_coefficients)
            except Exception as e:
                logger.debug(f"aruco.estimatePoseSingleMarkers failed: {e}")
                rvecs = None; tvecs = None

            # use our robust solver
            tvec, rvec, quat = my_estimatePoseSingleMarkers(corners=corners[i],
                                                            marker_size=marker_size,
                                                            camera_matrix=matrix_coefficients,
                                                            distortion=distortion_coefficients)
            if tvec is None or rvec is None or quat is None:
                logger.warn(f"Pose estimation failed for marker {marker_id} (skipping this marker)")
                continue

            # draw and publish safely
            frame_processed = aruco_display(corners=corners, ids=marker_ids, image=frame_processed)
            try:
                frame_processed = cv2.drawFrameAxes(image=frame_processed,
                                                    cameraMatrix=matrix_coefficients,
                                                    distCoeffs=distortion_coefficients,
                                                    rvec=rvec, tvec=tvec,
                                                    length=0.05, thickness=3)
            except Exception as e:
                logger.warn(f"drawFrameAxes failed: {e}")

            # populate pose...
            pose = Pose()
            pose.position.x = float(tvec[0])
            pose.position.y = float(tvec[1])
            pose.position.z = float(tvec[2])
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            pose_array.poses.append(pose)
            markers.poses.append(pose)
            markers.marker_ids.append(marker_id)
            
    return frame_processed, pose_array, markers


def my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion):
    """
    Robust wrapper around solvePnP for a single square ArUco marker.
    Returns (tvec, rvec, quaternion) or (None, None, None) if estimation failed.
    """

    # corners: usually shape (1,4,2) from detectMarkers
    if marker_size is None:
        return None, None, None

    # sanitize imagePoints: make shape (4,2) and dtype=float64
    image_pts = np.array(corners, dtype=np.float64)
    if image_pts.ndim == 3 and image_pts.shape[0] == 1:
        image_pts = image_pts.squeeze(0)   # (4,2)
    elif image_pts.ndim == 2 and image_pts.shape == (4,2):
        pass
    else:
        # try to reshape/adapt or fail
        try:
            image_pts = image_pts.reshape((4,2))
        except Exception:
            return None, None, None

    # quick validity checks
    if np.any(np.isnan(image_pts)) or np.any(np.isinf(image_pts)):
        return None, None, None

    # compute polygon area — reject degenerate / tiny markers
    def polygon_area(pts):
        x = pts[:,0]; y = pts[:,1]
        return 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))
    area_px = polygon_area(image_pts)
    if area_px < 10.0:   # tune threshold (pixels^2)
        return None, None, None

    # object points in marker frame (consistent ordering)
    obj_pts = np.array([[-marker_size/2, marker_size/2, 0.0],
                        [ marker_size/2, marker_size/2, 0.0],
                        [ marker_size/2,-marker_size/2, 0.0],
                        [-marker_size/2,-marker_size/2, 0.0]], dtype=np.float64)

    # try different solvePnP flags if one fails
    solvepnp_flags = [cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE]
    for flag in solvepnp_flags:
        try:
            retval, rvec, tvec = cv2.solvePnP(objectPoints=obj_pts,
                                              imagePoints=image_pts,
                                              cameraMatrix=camera_matrix,
                                              distCoeffs=distortion,
                                              flags=flag)
        except cv2.error:
            retval = False
            rvec = None
            tvec = None

        if not retval or rvec is None or tvec is None:
            continue

        # reshape and sanity-check
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        # Rodrigues -> rotation matrix
        try:
            rot3x3, jac = cv2.Rodrigues(rvec)
        except cv2.error:
            continue

        # ensure rot3x3 is a valid rotation: orthonormalize with SVD
        try:
            U, S, Vt = np.linalg.svd(rot3x3)
            rot_ortho = U.dot(Vt)
            # ensure right-handed (determinant = +1)
            if np.linalg.det(rot_ortho) < 0:
                # fix sign
                Vt[-1,:] *= -1
                rot_ortho = U.dot(Vt)
        except np.linalg.LinAlgError:
            continue

        # convert to quaternion robustly — mat2quat expects (3,3)
        try:
            quat = transforms3d.quaternions.mat2quat(rot_ortho)
            # normalize
            nq = np.linalg.norm(quat)
            if nq == 0 or np.isnan(nq):
                continue
            quat = quat / nq
        except Exception:
            # fallback: compute quaternion from rotation vector (Rodrigues)
            try:
                angle = np.linalg.norm(rvec)
                if angle == 0:
                    quat = np.array([0, 0, 0, 1], dtype=np.float64)
                else:
                    axis = (rvec.flatten() / angle)
                    qw = math.cos(angle/2.0)
                    qxyz = axis * math.sin(angle/2.0)
                    quat = np.array([qxyz[0], qxyz[1], qxyz[2], qw], dtype=np.float64)
            except Exception:
                continue

        # everything succeeded
        return tvec, rvec, quat

    # if we reach here, all attempts failed
    return None, None, None


def depth_to_pointcloud_centroid(depth_image: np.array, intrinsic_matrix: np.array,
                                 corners: np.array) -> np.array:
    """
    This function takes a depth image and the corners of a quadrilateral as input,
    and returns the centroid of the corresponding pointcloud.

    Args:
        depth_image: A 2D numpy array representing the depth image.
        corners: A list of 4 tuples, each representing the (x, y) coordinates of a corner.

    Returns:
        A tuple (x, y, z) representing the centroid of the segmented pointcloud.
    """

    # Get image parameters
    height, width = depth_image.shape
    

    # Check if all corners are within image bounds
    # corners has shape (1, 4, 2)
    corners_indices = np.array([(int(x), int(y)) for x, y in corners[0]])

    for x, y in corners_indices:
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError("One or more corners are outside the image bounds.")

    # bounding box of the polygon
    x_min = int(min(corners_indices[:, 0]))
    x_max = int(max(corners_indices[:, 0]))
    y_min = int(min(corners_indices[:, 1]))
    y_max = int(max(corners_indices[:, 1]))

    # create array of pixels inside the polygon defined by the corners
    # search for pixels inside the squared bounding box of the polygon
    points = []
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            if is_pixel_in_polygon(pixel=(x, y), corners=corners_indices):
                # add point to the list of points
                points.append([x, y, depth_image[y, x]])

    # Convert points to numpy array
    points = np.array(points, dtype=np.uint16)
   
    # convert to open3d image
    #depth_segmented = geometry.Image(points)
    # create pinhole camera model
    #pinhole_matrix = camera.PinholeCameraIntrinsic(width=width, height=height, 
    #                                               intrinsic_matrix=intrinsic_matrix)
    # Convert points to Open3D pointcloud
    #pointcloud = geometry.PointCloud.create_from_depth_image(depth=depth_segmented, intrinsic=pinhole_matrix,
    #                                                         depth_scale=1000.0)

    # apply formulas to pointcloud, where 
    # fx = intrinsic_matrix[0, 0], fy = intrinsic_matrix[1, 1]
    # cx = intrinsic_matrix[0, 2], cy = intrinsic_matrix[1, 2], 
    # u = x, v = y, d = depth_image[y, x], depth_scale = 1000.0,
    # z = d / depth_scale
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy

    # create pointcloud
    pointcloud = []
    for x, y, d in points:
        z = d / 1000.0
        x = (x - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
        y = (y - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]
        pointcloud.append([x, y, z])

    # Calculate centroid from pointcloud
    centroid = np.mean(np.array(pointcloud, dtype=np.uint16), axis=0)

    return centroid


def is_pixel_in_polygon(pixel: tuple, corners: np.array) -> bool:
    num_intersections = 0
    for i in range(len(corners)):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % len(corners)]
        if (y1 <= pixel[1] < y2) or (y2 <= pixel[1] < y1):
            x_intersection = (x2 - x1) * (pixel[1] - y1) / (y2 - y1) + x1
            if x_intersection > pixel[0]:
                num_intersections += 1
    return (num_intersections % 2) == 1

