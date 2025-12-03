"""
Bundle Adjustment Helper for BAL Dataset Format
Uses GTSAM's native readBal function which handles all coordinate conversions
"""

import numpy as np
import gtsam
from gtsam import symbol_shorthand
import time
import psutil
import threading
import pickle
import os

# Symbol shortcuts - using same as GTSAM examples
C = symbol_shorthand.C  # Cameras  
P = symbol_shorthand.P  # Points (landmarks)


def load_bal_data(filepath):
    """
    Load BAL file using GTSAM's native reader.
    This handles all coordinate system conversions internally.
    
    Returns:
        scene_data: gtsam.SfmData object
    """
    print(f"  Loading BAL file: {filepath}")
    scene_data = gtsam.readBal(filepath)
    print(f"  Loaded {scene_data.numberCameras()} cameras, {scene_data.numberTracks()} tracks")
    return scene_data


def build_factor_graph(scene_data):
    """
    Build GTSAM factor graph for bundle adjustment using SfmData.
    
    Following GTSAM's SFMExample_bal.py example exactly.
    
    Args:
        scene_data: gtsam.SfmData from readBal
    
    Returns:
        graph: gtsam.NonlinearFactorGraph
        initial: gtsam.Values with initial estimates
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    num_cameras = scene_data.numberCameras()
    num_tracks = scene_data.numberTracks()
    
    # Count total observations
    total_obs = 0
    for j in range(num_tracks):
        track = scene_data.track(j)
        total_obs += track.numberMeasurements()
    
    print(f"Building factor graph...")
    print(f"  {num_cameras} cameras, {num_tracks} tracks, {total_obs} observations")
    
    # Noise model for measurements (1 pixel standard deviation)
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    
    # Add factors for all tracks
    for j in range(num_tracks):
        track = scene_data.track(j)
        
        # Add measurements for this track
        for k in range(track.numberMeasurements()):
            # Get camera index and measurement
            cam_idx, measurement = track.measurement(k)
            
            # Create factor using GeneralSFMFactorCal3Bundler
            # This factor type works with PinholeCameraCal3Bundler
            factor = gtsam.GeneralSFMFactorCal3Bundler(
                measurement, measurement_noise, C(cam_idx), P(j)
            )
            graph.add(factor)
    
    # Add prior on the first camera to fix gauge freedom
    # This indirectly specifies where the origin is
    first_camera = scene_data.camera(0)
    graph.push_back(
        gtsam.PriorFactorPinholeCameraCal3Bundler(
            C(0), first_camera, gtsam.noiseModel.Isotropic.Sigma(9, 0.1)
        )
    )
    
    # Add prior on the first landmark to fix scale
    first_track = scene_data.track(0)
    graph.push_back(
        gtsam.PriorFactorPoint3(
            P(0), first_track.point3(), gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        )
    )
    
    # Create initial estimate from scene data
    # Add cameras
    for i in range(num_cameras):
        camera = scene_data.camera(i)
        initial.insert(C(i), camera)
    
    # Add 3D points from tracks
    for j in range(num_tracks):
        track = scene_data.track(j)
        initial.insert(P(j), track.point3())
    
    # Compute and print initial error
    initial_error = graph.error(initial)
    print(f"  Graph size: {graph.size()} factors")
    print(f"  Values size: {initial.size()} variables")
    print(f"  Initial error: {initial_error:.2e}")
    
    return graph, initial


def get_schur_sparsity(graph, initial):
    """
    Compute Schur complement sparsity (camera-camera block after marginalizing landmarks).
    
    Returns:
        sparsity: float between 0 (sparse) and 1 (dense)
    """
    try:
        # Linearize at initial estimate
        linear_graph = graph.linearize(initial)
        
        # Get the Hessian
        hessian = linear_graph.hessian()
        A = hessian[0]  # Information matrix
        
        # Count non-zeros in upper triangle
        n = A.shape[0]
        if n == 0:
            return 0.0
        
        nnz = np.count_nonzero(np.triu(A))
        max_nnz = n * (n + 1) // 2
        
        return nnz / max_nnz if max_nnz > 0 else 0.0
    except Exception as e:
        print(f"  Warning: Could not compute sparsity: {e}")
        return 0.0


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
    
    # Practical tolerances
    params.setRelativeErrorTol(1e-5)   
    params.setAbsoluteErrorTol(1e-5)
    
    # Set lambda parameters
    params.setlambdaInitial(lambda_initial)
    params.setlambdaUpperBound(lambda_upper_bound)   
        

    
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
    import gc
    gc.collect()
    
    return result, metrics


def print_results(dataset_name, scene_data, metrics):
    """Print formatted results summary."""
    # Count observations
    total_obs = 0
    for j in range(scene_data.numberTracks()):
        track = scene_data.track(j)
        total_obs += track.numberMeasurements()
    
    print("\n" + "=" * 70)
    print(f"RESULTS FOR {dataset_name}")
    print("=" * 70)
    print(f"  Dataset Size:")
    print(f"    - Cameras: {scene_data.numberCameras()}")
    print(f"    - Points: {scene_data.numberTracks()}")
    print(f"    - Observations: {total_obs}")
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


def save_results(all_results, filename='bal_results.pkl'):
    """Save results to pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  Results saved to {filename}")


def load_results(filename='bal_results.pkl'):
    """Load results from pickle file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return []


def save_results_numpy(all_results, filename='bal_results.npz'):
    """Save results to numpy file."""
    # Convert to format suitable for numpy
    save_dict = {}
    for i, r in enumerate(all_results):
        for key, value in r.items():
            if isinstance(value, (list, np.ndarray)):
                save_dict[f'{i}_{key}'] = np.array(value)
            else:
                save_dict[f'{i}_{key}'] = np.array([value])
    save_dict['num_results'] = np.array([len(all_results)])
    np.savez(filename, **save_dict)
    print(f"  Results saved to {filename}")


def extract_cameras_and_points(scene_data, values=None):
    """
    Extract camera poses and 3D points from scene_data or optimized values.
    
    Args:
        scene_data: gtsam.SfmData object
        values: gtsam.Values (if None, use initial values from scene_data)
    
    Returns:
        camera_positions: numpy array (N, 3) of camera centers
        camera_orientations: list of 3x3 rotation matrices
        points_3d: numpy array (M, 3) of 3D points
    """
    num_cameras = scene_data.numberCameras()
    num_points = scene_data.numberTracks()
    
    camera_positions = []
    camera_orientations = []
    
    for i in range(num_cameras):
        if values is not None:
            camera = values.atPinholeCameraCal3Bundler(C(i))
        else:
            camera = scene_data.camera(i)
        
        pose = camera.pose()
        # Camera center in world coordinates
        position = pose.translation()
        # Handle both gtsam.Point3 and numpy array
        if hasattr(position, 'x'):
            camera_positions.append([position.x(), position.y(), position.z()])
        else:
            camera_positions.append([position[0], position[1], position[2]])
        
        # Rotation matrix
        R = pose.rotation().matrix()
        camera_orientations.append(R)
    
    camera_positions = np.array(camera_positions)
    
    # Extract 3D points
    points_3d = []
    for j in range(num_points):
        if values is not None:
            point = values.atPoint3(P(j))
        else:
            track = scene_data.track(j)
            point = track.point3()
        
        # Handle both gtsam.Point3 and numpy array
        if hasattr(point, 'x'):
            points_3d.append([point.x(), point.y(), point.z()])
        else:
            points_3d.append([point[0], point[1], point[2]])
    
    points_3d = np.array(points_3d)
    
    return camera_positions, camera_orientations, points_3d


def plot_reconstruction(scene_data, initial_values, optimized_values, dataset_name, 
                        max_points=10000, save_path=None):
    """
    Plot 3D reconstruction before and after optimization using Plotly only.
    
    Args:
        scene_data: gtsam.SfmData object
        initial_values: gtsam.Values before optimization
        optimized_values: gtsam.Values after optimization  
        dataset_name: string for title
        max_points: maximum number of points to plot (for performance)
        save_path: if provided, save figure to this path
    """
    # Extract initial reconstruction
    init_cam_pos, init_cam_rot, init_points = extract_cameras_and_points(scene_data, initial_values)
    
    # Extract optimized reconstruction
    opt_cam_pos, opt_cam_rot, opt_points = extract_cameras_and_points(scene_data, optimized_values)
    
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
    import gc
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
    
    Rough estimate based on:
    - Each camera: ~9 parameters * 8 bytes = 72 bytes
    - Each point: ~3 parameters * 8 bytes = 24 bytes
    - Each observation factor: ~200 bytes (includes Jacobians)
    - Factor graph overhead: ~2x
    - Optimization workspace: ~3x
    
    Returns:
        float: estimated memory in GB
    """
    camera_mem = num_cameras * 72
    point_mem = num_points * 24
    factor_mem = num_observations * 200
    
    base_mem = (camera_mem + point_mem + factor_mem) / (1024**3)
    estimated = base_mem * 6  # Factor graph + optimization overhead
    
    return max(estimated, 1.0)  # At least 1 GB


def run_safe_bundle_adjustment(scene_data, max_iterations=50, verbose=True,
                                 lambda_initial=1.0, lambda_upper_bound=1e9):
    """
    Wrapper that checks memory before running bundle adjustment.
    Always uses Levenberg-Marquardt for consistent comparison.
    
    Args:
        scene_data: gtsam.SfmData object
        max_iterations: maximum iterations
        verbose: print detailed output
        lambda_initial: initial damping parameter (default: 1.0)
        lambda_upper_bound: maximum lambda value (default: 1e9)
    
    Returns:
        result: optimized values
        metrics: performance metrics
    """
    import gc
    
    # Force garbage collection before starting
    gc.collect()
    
    # Count observations
    total_obs = 0
    for j in range(scene_data.numberTracks()):
        track = scene_data.track(j)
        total_obs += track.numberMeasurements()
    
    num_cameras = scene_data.numberCameras()
    num_points = scene_data.numberTracks()
    
    # Estimate memory
    estimated_mem = estimate_memory_requirement(num_cameras, num_points, total_obs)
    
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
    graph, initial = build_factor_graph(scene_data)
    
    # Run optimization (always LM)
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


def run_dataset_for_notebook(name, bal_file, max_iterations=50, verbose=True, 
                               cleanup=True, lambda_initial=1.0, lambda_upper_bound=1e9):
    """
    Notebook-friendly function to run a single dataset with aggressive cleanup.
    Always uses Levenberg-Marquardt for consistent comparison.
    
    Args:
        name: dataset name
        bal_file: path to BAL file
        max_iterations: max optimization iterations
        verbose: print detailed output
        cleanup: if True, aggressively clean up after completion
        lambda_initial: initial damping parameter (default: 1.0)
        lambda_upper_bound: maximum lambda value (default: 1e9)
    
    Returns:
        result_entry: dict with results, or None if failed
    """
    import gc
    
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print(f"{'='*70}")
    
    try:
        # Load data
        scene_data = load_bal_data(bal_file)
        
        # Run safe optimization (always LM)
        result, metrics = run_safe_bundle_adjustment(
            scene_data,
            max_iterations=max_iterations,
            verbose=verbose,
            lambda_initial=lambda_initial,
            lambda_upper_bound=lambda_upper_bound
        )
        
        # Print results
        print_results(name, scene_data, metrics)
        
        # Count observations for storage
        total_obs = sum(scene_data.track(j).numberMeasurements() 
                       for j in range(scene_data.numberTracks()))
        
        # Store results
        result_entry = {
            'name': name,
            'num_cameras': scene_data.numberCameras(),
            'num_points': scene_data.numberTracks(),
            'num_observations': total_obs,
            **metrics
        }
        
        if cleanup:
            # Aggressive cleanup
            del scene_data, result, metrics
            gc.collect()
            print(f"\n  Memory cleaned up")
        
        return result_entry
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {bal_file}")
        return None
    except MemoryError:
        print(f"ERROR: Out of memory!")
        gc.collect()
        return None
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python helper_bal.py <path_to_bal_file> [--dogleg] [--max-iter N]")
        print("Example: python helper_bal.py problem-49-7776-pre.txt")
        print("Options:")
        print("  --dogleg: Force use of Dogleg optimizer (more memory efficient)")
        print("  --max-iter N: Set maximum iterations (default: 50)")
        sys.exit(1)
    
    filepath = sys.argv[1]
    dataset_name = filepath.split('/')[-1].replace('.txt', '')
    
    # Parse optional arguments
    max_iterations = 50
    force_dogleg = False
    for i, arg in enumerate(sys.argv[2:]):
        if arg == '--dogleg':
            force_dogleg = True
        elif arg == '--max-iter' and i+3 < len(sys.argv):
            max_iterations = int(sys.argv[i+3])
    
    print("=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)
    
    # Load using GTSAM's native BAL reader
    scene_data = load_bal_data(filepath)
    
    # Use safe wrapper that auto-selects optimizer
    if force_dogleg:
        print("WARNING: --dogleg flag ignored, using LM for consistent comparison")
    
    result, metrics = run_safe_bundle_adjustment(
        scene_data,
        max_iterations=max_iterations,
        verbose=True
    )
    
    # Print results
    print_results(dataset_name, scene_data, metrics)