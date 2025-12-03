"""
Bundle Adjustment Helper for 1DSfM/COLMAP Dataset Format
Handles COLMAP bundle.out format with comprehensive functionality
"""

import numpy as np
import gtsam
from gtsam import symbol_shorthand
import time
import psutil
import threading
import pickle
import os
import gc

# Symbol shortcuts
C = symbol_shorthand.C  # Cameras  
P = symbol_shorthand.P  # Points (landmarks)


def _read_line(f):
    """Read next non-comment, non-empty line from file."""
    while True:
        line = f.readline()
        if not line:  # EOF
            return None
        line = line.strip()
        if line and not line.startswith('#'):
            return line


class ResourceMonitor:
    """Monitor CPU and RAM usage during optimization."""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.cpu_samples = []
        self.ram_samples = []
        self.running = False
        self.thread = None
        self.process = psutil.Process()
    
    def _sample(self):
        while self.running:
            self.cpu_samples.append(psutil.cpu_percent())
            self.ram_samples.append(self.process.memory_info().rss / (1024**3))  # GB
            time.sleep(self.interval)
    
    def start(self):
        self.cpu_samples = []
        self.ram_samples = []
        self.running = True
        self.thread = threading.Thread(target=self._sample)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        
        return {
            'max_cpu': max(self.cpu_samples) if self.cpu_samples else 0,
            'avg_cpu': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'max_ram_gb': max(self.ram_samples) if self.ram_samples else 0,
            'avg_ram_gb': np.mean(self.ram_samples) if self.ram_samples else 0,
        }


def load_1dsfm_data(bundle_file, list_file=None):
    """
    Load 1DSfM/COLMAP bundle.out file format.
    Handles comment lines starting with '#'.
    
    Args:
        bundle_file: path to bundle.out file
        list_file: optional path to list.txt file (for image names)
    
    Returns:
        cameras: numpy structured array with camera parameters
        points: numpy array of 3D points
        observations: numpy array of observations (cam_id, point_id, x, y)
        image_names: list of image names (if list_file provided)
    """
    print(f"  Loading 1DSfM file: {bundle_file}")
    
    with open(bundle_file, 'r') as f:
        # Read header (skip comments)
        header = _read_line(f)
        if header is None:
            raise ValueError("Empty file or only comments")
        
        num_cameras, num_points = map(int, header.split())
        
        print(f"  Parsing: {num_cameras} cameras, {num_points} points")
        
        # Parse cameras - structured array for efficiency
        camera_dtype = np.dtype([
            ('f', np.float64),
            ('k1', np.float64),
            ('k2', np.float64),
            ('R', np.float64, (3, 3)),
            ('t', np.float64, 3)
        ])
        cameras = np.empty(num_cameras, dtype=camera_dtype)
        
        for cam_id in range(num_cameras):
            # Focal length and distortion
            line = _read_line(f)
            if line is None:
                raise ValueError(f"Unexpected EOF while reading camera {cam_id}")
            f_val, k1, k2 = map(float, line.split())
            
            # Rotation matrix (3 lines)
            R = np.zeros((3, 3))
            for i in range(3):
                line = _read_line(f)
                if line is None:
                    raise ValueError(f"Unexpected EOF while reading camera {cam_id} rotation")
                R[i] = list(map(float, line.split()))
            
            # Translation vector
            line = _read_line(f)
            if line is None:
                raise ValueError(f"Unexpected EOF while reading camera {cam_id} translation")
            t = np.array(list(map(float, line.split())))
            
            cameras[cam_id]['f'] = f_val
            cameras[cam_id]['k1'] = k1
            cameras[cam_id]['k2'] = k2
            cameras[cam_id]['R'] = R
            cameras[cam_id]['t'] = t
        
        # Parse points with observations
        points = np.empty((num_points, 3), dtype=np.float64)
        observations = []
        
        for point_id in range(num_points):
            # 3D position
            line = _read_line(f)
            if line is None:
                raise ValueError(f"Unexpected EOF while reading point {point_id}")
            position = np.array(list(map(float, line.split())))
            points[point_id] = position
            
            # Color (read but ignore)
            line = _read_line(f)
            if line is None:
                raise ValueError(f"Unexpected EOF while reading point {point_id} color")
            
            # View list
            line = _read_line(f)
            if line is None:
                raise ValueError(f"Unexpected EOF while reading point {point_id} views")
            view_list = list(map(float, line.split()))
            
            num_views = int(view_list[0])
            for i in range(num_views):
                cam_idx = int(view_list[1 + 4*i])
                key_idx = int(view_list[2 + 4*i])
                x = view_list[3 + 4*i]
                y = view_list[4 + 4*i]
                observations.append((cam_idx, point_id, x, y))
    
    # Convert to numpy array
    observations = np.array(observations, dtype=np.float64)
    
    # Load image names if provided
    image_names = None
    if list_file and os.path.exists(list_file):
        with open(list_file, 'r') as f:
            image_names = [line.strip() for line in f]
        print(f"  Loaded {len(image_names)} image names")
    
    print(f"  Loaded {num_cameras} cameras, {num_points} points, {len(observations)} observations")
    
    return cameras, points, observations, image_names


def build_factor_graph(cameras, points, observations, use_robust=True, 
                        filter_behind_camera=False, use_bundler_camera=False,
                        min_observations_per_point=2):
    """
    Build GTSAM factor graph for bundle adjustment.
    
    Args:
        cameras: numpy structured array from load_1dsfm_data
        points: numpy array of 3D points
        observations: numpy array of observations
        use_robust: if True, use Huber robust loss (recommended for outliers)
        filter_behind_camera: if True, skip points behind cameras (usually False for COLMAP)
        use_bundler_camera: if True, use Cal3Bundler with distortion (for COLMAP Bundler exports)
        min_observations_per_point: minimum observations required per point (default: 2)
    
    Returns:
        graph: gtsam.NonlinearFactorGraph
        initial: gtsam.Values with initial estimates
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    num_cameras = len(cameras)
    num_points = len(points)
    num_obs = len(observations)
    
    print(f"Building factor graph...")
    print(f"  {num_cameras} cameras, {num_points} points, {num_obs} observations")
    
    # Check observation ranges for diagnostics
    if num_obs > 0:
        x_vals = observations[:, 2]
        y_vals = observations[:, 3]
        print(f"  Observation ranges: X=[{x_vals.min():.1f}, {x_vals.max():.1f}], Y=[{y_vals.min():.1f}, {y_vals.max():.1f}]")
        
        # Check if coordinates look centered
        coords_are_centered = (x_vals.min() < 0 or y_vals.min() < 0)
        if coords_are_centered:
            print(f"  -> Observations are already centered")
        else:
            print(f"  -> Observations are in pixel coordinates (all positive)")
    
    # Choose camera model
    if use_bundler_camera:
        # Use Cal3Bundler: f, k1, k2, u0, v0 (principal point at center for COLMAP)
        print(f"  Using Cal3Bundler camera model (with distortion)")
        # For now, average the calibration across all cameras
        avg_f = np.mean([cam['f'] for cam in cameras])
        avg_k1 = np.mean([cam['k1'] for cam in cameras])
        avg_k2 = np.mean([cam['k2'] for cam in cameras])
        K = gtsam.Cal3Bundler(avg_f, avg_k1, avg_k2, 0.0, 0.0)
        print(f"  Average calibration: f={avg_f:.1f}, k1={avg_k1:.6f}, k2={avg_k2:.6f}")
        camera_model = "bundler"
    else:
        # Use simplified Cal3_S2: fx, fy, s=0, px=0, py=0
        print(f"  Using Cal3_S2 camera model (no distortion)")
        K = gtsam.Cal3_S2(cameras[0]['f'], cameras[0]['f'], 0, 0, 0)
        camera_model = "s2"
    
    # First pass: filter observations and count per camera/point
    valid_observations = []
    camera_obs_count = {}
    point_obs_count = {}
    num_filtered_behind = 0
    
    # Add all camera poses/cameras first
    for cam_id in range(num_cameras):
        R = gtsam.Rot3(cameras[cam_id]['R'])
        t = gtsam.Point3(cameras[cam_id]['t'])
        pose = gtsam.Pose3(R, t)
        
        if camera_model == "bundler":
            # Create individual calibration for each camera
            K_cam = gtsam.Cal3Bundler(
                cameras[cam_id]['f'], 
                cameras[cam_id]['k1'], 
                cameras[cam_id]['k2'], 
                0.0, 0.0
            )
            camera = gtsam.PinholeCameraCal3Bundler(pose, K_cam)
            initial.insert(C(cam_id), camera)
        else:
            initial.insert(C(cam_id), pose)
        
        camera_obs_count[cam_id] = 0
    
    # Add all 3D points first
    for pt_id in range(num_points):
        initial.insert(P(pt_id), gtsam.Point3(points[pt_id]))
        point_obs_count[pt_id] = 0
    
    # Filter observations
    for obs in observations:
        cam_id, pt_id, x, y = int(obs[0]), int(obs[1]), obs[2], obs[3]
        
        # Optional: Filter points behind camera (cheirality check)
        if filter_behind_camera:
            try:
                camera_pose = initial.atPose3(C(cam_id))
                point_3d = initial.atPoint3(P(pt_id))
                
                # Transform point to camera frame
                local_point = camera_pose.transformTo(point_3d)
                
                # Check if point is in front of camera (positive Z)
                if local_point[2] <= 0:
                    num_filtered_behind += 1
                    continue
            except:
                num_filtered_behind += 1
                continue
        
        valid_observations.append((cam_id, pt_id, x, y))
        camera_obs_count[cam_id] = camera_obs_count.get(cam_id, 0) + 1
        point_obs_count[pt_id] = point_obs_count.get(pt_id, 0) + 1
    
    # Identify valid cameras and points (those with enough observations)
    valid_cameras = set(cam_id for cam_id, count in camera_obs_count.items() if count > 0)
    valid_points = set(pt_id for pt_id, count in point_obs_count.items() 
                       if count >= min_observations_per_point)
    
    if num_filtered_behind > 0:
        print(f"  Filtered {num_filtered_behind} observations (points behind camera)")
    print(f"  Valid cameras: {len(valid_cameras)}/{num_cameras}")
    print(f"  Valid points: {len(valid_points)}/{num_points}")
    
    # Create noise model - ROBUST for outlier rejection
    if use_robust:
        base_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        measurement_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            base_noise
        )
        print(f"  Using Huber robust loss function")
    else:
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    
    # Add projection factors only for valid cameras and points
    num_added = 0
    for cam_id, pt_id, x, y in valid_observations:
        if cam_id in valid_cameras and pt_id in valid_points:
            measurement = gtsam.Point2(x, y)
            
            if camera_model == "bundler":
                # Use GeneralSFMFactorCal3Bundler (same as BAL format)
                factor = gtsam.GeneralSFMFactorCal3Bundler(
                    measurement, measurement_noise, C(cam_id), P(pt_id)
                )
            else:
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    measurement, measurement_noise, C(cam_id), P(pt_id), K
                )
            graph.add(factor)
            num_added += 1
    
    print(f"  Added {num_added} projection factors")
    
    # Add prior on first valid camera to fix gauge freedom
    first_cam = min(valid_cameras)
    if camera_model == "bundler":
        first_camera = initial.atPinholeCameraCal3Bundler(C(first_cam))
        graph.push_back(
            gtsam.PriorFactorPinholeCameraCal3Bundler(
                C(first_cam), first_camera, gtsam.noiseModel.Isotropic.Sigma(9, 0.1)
            )
        )
    else:
        first_pose = initial.atPose3(C(first_cam))
        graph.push_back(
            gtsam.PriorFactorPose3(
                C(first_cam), first_pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
            )
        )
    
    # Add prior on first valid landmark to fix scale
    first_point = min(valid_points)
    first_point_3d = initial.atPoint3(P(first_point))
    graph.push_back(
        gtsam.PriorFactorPoint3(
            P(first_point), first_point_3d, gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        )
    )
    
    # Compute and print initial error
    initial_error = graph.error(initial)
    print(f"  Graph size: {graph.size()} factors")
    print(f"  Values size: {initial.size()} variables")
    print(f"  Initial error: {initial_error:.2e}")
    
    if initial_error > 1e8:
        print(f"  NOTE: High initial error is expected for COLMAP Bundler exports")
        print(f"        Huber loss will handle outliers during optimization")
    
    return graph, initial
    """
    Build GTSAM factor graph for bundle adjustment.
    
    Args:
        cameras: numpy structured array from load_1dsfm_data
        points: numpy array of 3D points
        observations: numpy array of observations
        use_robust: if True, use Huber robust loss (recommended for outliers)
        filter_behind_camera: if True, skip points behind cameras
        image_width: optional, for coordinate conversion (if observations not centered)
        image_height: optional, for coordinate conversion (if observations not centered)
        min_observations_per_point: minimum observations required per point (default: 2)
    
    Returns:
        graph: gtsam.NonlinearFactorGraph
        initial: gtsam.Values with initial estimates
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    num_cameras = len(cameras)
    num_points = len(points)
    num_obs = len(observations)
    
    print(f"Building factor graph...")
    print(f"  {num_cameras} cameras, {num_points} points, {num_obs} observations")
    
    # Check observation ranges for diagnostics
    if num_obs > 0:
        x_vals = observations[:, 2]
        y_vals = observations[:, 3]
        print(f"  Observation ranges: X=[{x_vals.min():.1f}, {x_vals.max():.1f}], Y=[{y_vals.min():.1f}, {y_vals.max():.1f}]")
        
        # Check if coordinates look centered
        coords_are_centered = (x_vals.min() < 0 or y_vals.min() < 0)
        if coords_are_centered:
            print(f"  -> Observations are already centered (contains negative values)")
            if image_width is not None or image_height is not None:
                print(f"  -> WARNING: Ignoring image_width/height - coordinates already centered!")
                image_width = None  # Don't apply conversion
                image_height = None
        else:
            print(f"  -> Observations are in pixel coordinates (all positive)")
            if image_width is None or image_height is None:
                print(f"  -> HINT: Consider providing image_width and image_height for conversion")
    
    # Use first camera's focal length for calibration
    K = gtsam.Cal3_S2(cameras[0]['f'], cameras[0]['f'], 0, 0, 0)
    
    # First pass: filter observations and count per camera/point
    valid_observations = []
    camera_obs_count = {}
    point_obs_count = {}
    num_filtered_behind = 0
    
    # Add all camera poses first
    for cam_id in range(num_cameras):
        R = gtsam.Rot3(cameras[cam_id]['R'])
        t = gtsam.Point3(cameras[cam_id]['t'])
        pose = gtsam.Pose3(R, t)
        initial.insert(C(cam_id), pose)
        camera_obs_count[cam_id] = 0
    
    # Add all 3D points first
    for pt_id in range(num_points):
        initial.insert(P(pt_id), gtsam.Point3(points[pt_id]))
        point_obs_count[pt_id] = 0
    
    # Filter observations
    for obs in observations:
        cam_id, pt_id, x, y = int(obs[0]), int(obs[1]), obs[2], obs[3]
        
        # Optional: Convert observations from top-left to centered coordinates
        if image_width is not None and image_height is not None:
            x_centered = x - image_width / 2.0
            y_centered = y - image_height / 2.0
        else:
            x_centered = x
            y_centered = y
        
        # Optional: Filter points behind camera (cheirality check)
        if filter_behind_camera:
            try:
                camera_pose = initial.atPose3(C(cam_id))
                point_3d = initial.atPoint3(P(pt_id))
                
                # Transform point to camera frame
                local_point = camera_pose.transformTo(point_3d)
                
                # Check if point is in front of camera (positive Z)
                if local_point[2] <= 0:
                    num_filtered_behind += 1
                    continue
            except:
                num_filtered_behind += 1
                continue
        
        valid_observations.append((cam_id, pt_id, x_centered, y_centered))
        camera_obs_count[cam_id] = camera_obs_count.get(cam_id, 0) + 1
        point_obs_count[pt_id] = point_obs_count.get(pt_id, 0) + 1
    
    # Identify valid cameras and points (those with enough observations)
    valid_cameras = set(cam_id for cam_id, count in camera_obs_count.items() if count > 0)
    valid_points = set(pt_id for pt_id, count in point_obs_count.items() 
                       if count >= min_observations_per_point)
    
    if num_filtered_behind > 0:
        print(f"  Filtered {num_filtered_behind} observations (points behind camera)")
    if image_width is not None and image_height is not None:
        print(f"  Converted coordinates: center = ({image_width/2:.1f}, {image_height/2:.1f})")
    print(f"  Valid cameras: {len(valid_cameras)}/{num_cameras}")
    print(f"  Valid points: {len(valid_points)}/{num_points}")
    
    # Create noise model - ROBUST for outlier rejection
    if use_robust:
        base_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        measurement_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            base_noise
        )
        print(f"  Using Huber robust loss function")
    else:
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    
    # Add projection factors only for valid cameras and points
    num_added = 0
    for cam_id, pt_id, x, y in valid_observations:
        if cam_id in valid_cameras and pt_id in valid_points:
            measurement = gtsam.Point2(x, y)
            factor = gtsam.GenericProjectionFactorCal3_S2(
                measurement, measurement_noise, C(cam_id), P(pt_id), K
            )
            graph.add(factor)
            num_added += 1
    
    print(f"  Added {num_added} projection factors")
    
    # Add prior on first valid camera to fix gauge freedom
    first_cam = min(valid_cameras)
    first_pose = initial.atPose3(C(first_cam))
    graph.push_back(
        gtsam.PriorFactorPose3(
            C(first_cam), first_pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        )
    )
    
    # Add prior on first valid landmark to fix scale
    first_point = min(valid_points)
    first_point_3d = initial.atPoint3(P(first_point))
    graph.push_back(
        gtsam.PriorFactorPoint3(
            P(first_point), first_point_3d, gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        )
    )
    
    # Compute and print initial error
    initial_error = graph.error(initial)
    print(f"  Graph size: {graph.size()} factors")
    print(f"  Values size: {initial.size()} variables")
    print(f"  Initial error: {initial_error:.2e}")
    
    if initial_error > 1e10:
        print(f"  WARNING: Very high initial error! Possible coordinate system mismatch.")
        print(f"  Consider setting image_width/image_height for coordinate conversion.")
    
    return graph, initial


def run_bundle_adjustment(graph, initial, max_iterations=50, verbose=True,
                           lambda_initial=1.0, lambda_upper_bound=1e9):
    """
    Run Levenberg-Marquardt bundle adjustment with resource monitoring.
    
    Args:
        graph: gtsam.NonlinearFactorGraph
        initial: gtsam.Values
        max_iterations: maximum LM iterations
        verbose: if True, print detailed iteration info
        lambda_initial: initial damping parameter (default: 1.0)
        lambda_upper_bound: maximum lambda value (default: 1e9)
    
    Returns:
        result: optimized gtsam.Values
        metrics: dict with timing and resource metrics
    """
    monitor = ResourceMonitor()
    
    # Setup LM parameters
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    
    # Better tolerances for large problems
    params.setRelativeErrorTol(1e-5)   
    params.setAbsoluteErrorTol(1e-5)
    
    # Set lambda parameters - more aggressive start for large problems
    params.setlambdaInitial(lambda_initial)
    params.setlambdaUpperBound(lambda_upper_bound)
    params.setlambdaLowerBound(1e-10)
    
    # Use better linear solver for large sparse problems
    params.setLinearSolverType("MULTIFRONTAL_CHOLESKY")
    
    # Use GTSAM defaults for lambda - they're well tested
    # Default: lambdaInitial=1e-5, lambdaFactor=10, bounds are reasonable
    
    # Set verbosity level
    if verbose:
        params.setVerbosity("ERROR")
        params.setVerbosityLM("SUMMARY")
    else:
        params.setVerbosity("SILENT")
        params.setVerbosityLM("SILENT")
    
    # Create optimizer
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    
    # Print LM settings
    if verbose:
        print(f"\nLM Parameters:")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Initial lambda: {params.getlambdaInitial()}")
        print(f"  Lambda factor: {params.getlambdaFactor()}")
        print(f"  Lambda upper bound: {params.getlambdaUpperBound()}")
        print(f"  Lambda lower bound: {params.getlambdaLowerBound()}")
        print(f"  Relative error tol: {params.getRelativeErrorTol()}")
        print(f"  Absolute error tol: {params.getAbsoluteErrorTol()}")
    
    initial_error = graph.error(initial)
    print(f"\nRunning Bundle Adjustment...")
    print(f"  Initial error: {initial_error:.6e}")
    print("-" * 60)
    
    # Track per-iteration errors
    iteration_errors = [initial_error]
    iteration_times = [0.0]
    
    monitor.start()
    start_time = time.time()
    
    try:
        # Manual iteration for detailed logging
        current_error = initial_error
        iteration = 0
        
        while iteration < max_iterations:
            iter_start = time.time()
            
            # Perform one iteration
            try:
                optimizer.iterate()
            except Exception as e:
                print(f"  Iteration {iteration + 1}: Exception - {e}")
                break
            
            iteration += 1
            new_error = optimizer.error()
            iter_time = time.time() - iter_start
            
            iteration_errors.append(new_error)
            iteration_times.append(time.time() - start_time)
            
            # Print iteration info
            error_reduction = (current_error - new_error) / current_error * 100 if current_error > 0 else 0
            try:
                lambda_val = optimizer.lambda_()
                print(f"  Iter {iteration:3d}: error = {new_error:.6e}, "
                      f"reduction = {error_reduction:7.3f}%, "
                      f"lambda = {lambda_val:.2e}, "
                      f"time = {iter_time:.2f}s")
            except:
                print(f"  Iter {iteration:3d}: error = {new_error:.6e}, "
                      f"reduction = {error_reduction:7.3f}%, "
                      f"time = {iter_time:.2f}s")
            
            # Check convergence
            if abs(current_error - new_error) < params.getAbsoluteErrorTol():
                print(f"  Converged: absolute error tolerance reached")
                break
            if current_error > 0 and abs(current_error - new_error) / current_error < params.getRelativeErrorTol():
                print(f"  Converged: relative error tolerance reached")
                break
            
            current_error = new_error
        
        result = optimizer.values()
        success = True
        
    except Exception as e:
        print(f"  Optimization exception: {e}")
        result = initial
        success = False
    
    elapsed = time.time() - start_time
    resource_metrics = monitor.stop()
    
    final_error = graph.error(result)
    iterations = len(iteration_errors) - 1
    
    print("-" * 60)
    print(f"  Final error: {final_error:.6e}")
    print(f"  Total iterations: {iterations}")
    print(f"  Total time: {elapsed:.2f}s")
    
    # Check improvement
    if initial_error > 1e-10:
        improvement = (initial_error - final_error) / initial_error
        if improvement > 1e-6:
            print(f"  Error reduction: {improvement*100:.4f}%")
        elif improvement < -1e-6:
            print(f"  Warning: Error increased!")
            success = False
    
    metrics = {
        'time': elapsed,
        'initial_error': initial_error,
        'final_error': final_error,
        'iterations': iterations,
        'success': success,
        'iteration_errors': iteration_errors,
        'iteration_times': iteration_times,
        **resource_metrics
    }
    
    # Aggressive cleanup
    del optimizer, params, monitor
    gc.collect()
    
    return result, metrics


def print_results(dataset_name, cameras, points, observations, metrics):
    """Print formatted results summary."""
    print("\n" + "=" * 70)
    print(f"RESULTS FOR {dataset_name}")
    print("=" * 70)
    print(f"  Dataset Size:")
    print(f"    - Cameras: {len(cameras)}")
    print(f"    - Points: {len(points)}")
    print(f"    - Observations: {len(observations)}")
    print(f"  Performance:")
    print(f"    - Success: {metrics['success']}")
    print(f"    - Time: {metrics['time']:.2f} seconds")
    print(f"    - Max CPU: {metrics['max_cpu']:.1f}%")
    print(f"    - Avg CPU: {metrics['avg_cpu']:.1f}%")
    print(f"    - Max RAM: {metrics['max_ram_gb']:.2f} GB")
    print(f"    - Avg RAM: {metrics['avg_ram_gb']:.2f} GB")
    print(f"  Optimization:")
    print(f"    - Initial error: {metrics['initial_error']:.2e}")
    print(f"    - Final error: {metrics['final_error']:.2e}")
    print(f"    - Iterations: {metrics['iterations']}")
    print("=" * 70)


def save_results(all_results, filename='1dsfm_results.pkl'):
    """Save results to pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  Results saved to {filename}")


def load_results(filename='1dsfm_results.pkl'):
    """Load results from pickle file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return []


def extract_cameras_and_points(cameras, points, values=None):
    """
    Extract camera poses and 3D points from initial data or optimized values.
    
    Args:
        cameras: numpy structured array
        points: numpy array of 3D points
        values: gtsam.Values (if None, use initial values)
    
    Returns:
        camera_positions: numpy array (N, 3) of camera centers
        camera_orientations: list of 3x3 rotation matrices
        points_3d: numpy array (M, 3) of 3D points
    """
    num_cameras = len(cameras)
    num_points = len(points)
    
    camera_positions = []
    camera_orientations = []
    
    for i in range(num_cameras):
        if values is not None:
            pose = values.atPose3(C(i))
            position = pose.translation()
            R = pose.rotation().matrix()
        else:
            R = cameras[i]['R']
            t = cameras[i]['t']
            pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))
            position = pose.translation()
        
        # Handle both gtsam.Point3 and numpy array
        if hasattr(position, 'x'):
            camera_positions.append([position.x(), position.y(), position.z()])
        else:
            camera_positions.append([position[0], position[1], position[2]])
        
        camera_orientations.append(R)
    
    camera_positions = np.array(camera_positions)
    
    # Extract 3D points
    points_3d = []
    for j in range(num_points):
        if values is not None:
            point = values.atPoint3(P(j))
        else:
            point = points[j]
        
        # Handle both gtsam.Point3 and numpy array
        if hasattr(point, 'x'):
            points_3d.append([point.x(), point.y(), point.z()])
        else:
            points_3d.append([point[0], point[1], point[2]])
    
    points_3d = np.array(points_3d)
    
    return camera_positions, camera_orientations, points_3d


def plot_reconstruction(cameras, points, initial_values, optimized_values, dataset_name, 
                        max_points=10000, save_path=None):
    """
    Plot 3D reconstruction before and after optimization using Plotly only.
    
    Args:
        cameras: numpy structured array
        points: numpy array of 3D points
        initial_values: gtsam.Values before optimization
        optimized_values: gtsam.Values after optimization  
        dataset_name: string for title
        max_points: maximum number of points to plot (for performance)
        save_path: if provided, save figure to this path
    """
    # Extract initial reconstruction
    init_cam_pos, init_cam_rot, init_points = extract_cameras_and_points(cameras, points, initial_values)
    
    # Extract optimized reconstruction
    opt_cam_pos, opt_cam_rot, opt_points = extract_cameras_and_points(cameras, points, optimized_values)
    
    # Subsample points if too many
    if len(init_points) > max_points:
        indices = np.random.choice(len(init_points), max_points, replace=False)
        init_points_plot = init_points[indices]
        opt_points_plot = opt_points[indices]
    else:
        init_points_plot = init_points
        opt_points_plot = opt_points
    
    plot_reconstruction_plotly(
        init_cam_pos, init_cam_rot, init_points_plot,
        opt_cam_pos, opt_cam_rot, opt_points_plot,
        dataset_name, save_path
    )


def plot_reconstruction_plotly(init_cam_pos, init_cam_rot, init_points,
                                opt_cam_pos, opt_cam_rot, opt_points,
                                dataset_name, save_path=None):
    """Create interactive 3D plot using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  Plotly not installed. Install with: pip install plotly")
        print("  Skipping visualization.")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(f'{dataset_name} - Before Optimization', 
                       f'{dataset_name} - After Optimization'),
        horizontal_spacing=0.05
    )
    
    # Helper function to add traces for one reconstruction
    def add_reconstruction_traces(fig, points, col):
        # Add 3D points only
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=1, color='blue', opacity=0.3),
                name='3D Points',
                showlegend=(col == 1)
            ),
            row=1, col=col
        )
    
    # Add traces for both reconstructions
    add_reconstruction_traces(fig, init_points, 1)
    add_reconstruction_traces(fig, opt_points, 2)
    
    # Update layout
    fig.update_layout(
        title=dict(text=f'3D Reconstruction: {dataset_name}', x=0.5),
        height=700,
        width=1400,
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h')
    )
    
    # Set equal aspect ratio for both subplots based on 90th percentile
    all_points = np.vstack([init_points, opt_points])
    mid = np.mean(all_points, axis=0)
    
    # Calculate distances from centroid
    distances = np.sqrt(np.sum((all_points - mid)**2, axis=1))
    percentile_90 = np.percentile(distances, 90)
    
    # Use 90th percentile distance as range (with small padding)
    max_range = percentile_90 * 1.1
    
    for col in [1, 2]:
        fig.update_scenes(
            dict(
                xaxis=dict(range=[mid[0] - max_range, mid[0] + max_range]),
                yaxis=dict(range=[mid[1] - max_range, mid[1] + max_range]),
                zaxis=dict(range=[mid[2] - max_range, mid[2] + max_range]),
                aspectmode='cube'
            ),
            row=1, col=col
        )
    
    # Save as HTML for interactivity
    if save_path:
        html_path = save_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"  Interactive plot saved to {html_path}")
        
        # Also save static image if kaleido is installed
        try:
            fig.write_image(save_path)
            print(f"  Static plot saved to {save_path}")
        except Exception:
            pass
    
    # Show the plot
    fig.show()
    
    # Clean up
    del fig, all_points, distances
    gc.collect()


def check_memory_available(required_gb=4.0):
    """
    Check if enough memory is available before running optimization.
    
    Args:
        required_gb: estimated memory required in GB
    
    Returns:
        bool: True if enough memory available
    """
    available = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"  Available RAM: {available:.1f} GB, Required: ~{required_gb:.1f} GB")
    
    if available < required_gb:
        print(f"  WARNING: Low memory! May crash. Consider closing other applications.")
        return False
    return True


def estimate_memory_requirement(num_cameras, num_points, num_observations):
    """
    Estimate memory requirement for bundle adjustment.
    
    Returns:
        float: estimated memory in GB
    """
    camera_mem = num_cameras * 72
    point_mem = num_points * 24
    factor_mem = num_observations * 200
    
    base_mem = (camera_mem + point_mem + factor_mem) / (1024**3)
    estimated = base_mem * 6  # Factor graph + optimization overhead
    
    return max(estimated, 1.0)  # At least 1 GB


def run_safe_bundle_adjustment(cameras, points, observations, max_iterations=50, 
                                 verbose=True, lambda_initial=1.0, lambda_upper_bound=1e9):
    """
    Wrapper that checks memory before running bundle adjustment.
    
    Args:
        cameras: numpy structured array
        points: numpy array of 3D points
        observations: numpy array of observations
        max_iterations: maximum iterations
        verbose: print detailed output
        lambda_initial: initial damping parameter
        lambda_upper_bound: maximum lambda value
    
    Returns:
        result: optimized values
        metrics: performance metrics
    """
    # Force garbage collection before starting
    gc.collect()
    
    num_cameras = len(cameras)
    num_points = len(points)
    num_obs = len(observations)
    
    # Estimate memory
    estimated_mem = estimate_memory_requirement(num_cameras, num_points, num_obs)
    
    # Check available memory
    available_mem = psutil.virtual_memory().available / (1024**3)
    
    print(f"\nMemory Analysis:")
    print(f"  Available RAM: {available_mem:.1f} GB")
    print(f"  Estimated need: {estimated_mem:.1f} GB")
    print(f"  Using Levenberg-Marquardt optimizer")
    print(f"  Lambda initial: {lambda_initial:.2e}")
    print(f"  Lambda upper bound: {lambda_upper_bound:.2e}")
    
    if estimated_mem > available_mem * 0.8:
        print(f"  WARNING: High memory usage expected!")
    
    # Build graph
    graph, initial = build_factor_graph(cameras, points, observations)
    
    # Run optimization
    result, metrics = run_bundle_adjustment(
        graph, initial, 
        max_iterations=max_iterations,
        verbose=verbose,
        lambda_initial=lambda_initial,
        lambda_upper_bound=lambda_upper_bound
    )
    
    # Aggressive cleanup
    del graph, initial
    gc.collect()
    
    return result, metrics


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python helper_1dsfm.py <path_to_bundle_file> [list_file]")
        print("Example: python helper_1dsfm.py colmap_bundle.out.bundle.out colmap_bundle.out.list.txt")
        sys.exit(1)
    
    bundle_file = sys.argv[1]
    list_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    dataset_name = os.path.basename(bundle_file).replace('.bundle.out', '')
    
    print("=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)
    
    # Load data
    cameras, points, observations, image_names = load_1dsfm_data(bundle_file, list_file)
    
    # Run optimization
    result, metrics = run_safe_bundle_adjustment(
        cameras, points, observations,
        max_iterations=50,
        verbose=True
    )
    
    # Print results
    print_results(dataset_name, cameras, points, observations, metrics)