import numpy as np
import gtsam
from gtsam import symbol_shorthand
import psutil
import time
import threading
from scipy.sparse import lil_matrix, csr_matrix

L = symbol_shorthand.L  # Landmarks
C = symbol_shorthand.C  # Cameras

# Monitors CPU, RAM usage in separate thread during BA optimization
class ResourceMonitor:
    def __init__(self):
        self.cpu_samples = []
        self.ram_samples = []
        self.monitoring = False
        self.thread = None
        self.process = psutil.Process()
    
    def _monitor(self):
        while self.monitoring:
            self.cpu_samples.append(psutil.cpu_percent(interval=0.1))
            self.ram_samples.append(self.process.memory_info().rss / 1024**3)  # GB
            time.sleep(0.1)
    
    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return {
            'max_cpu': max(self.cpu_samples) if self.cpu_samples else 0,
            'avg_cpu': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'max_ram': max(self.ram_samples) if self.ram_samples else 0,
            'avg_ram': np.mean(self.ram_samples) if self.ram_samples else 0
        }

# Parse COLMAP bundle.out file format - optimized streaming
def parse_colmap_bundle(bundle_file, list_file):
    with open(bundle_file, 'r') as f:
        # First line: num_cameras num_points
        header = f.readline().strip().split()
        num_cameras, num_points = map(int, header)
        
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
            f_val, k1, k2 = map(float, f.readline().split())
            R = np.array([list(map(float, f.readline().split())) for _ in range(3)])
            t = np.array(list(map(float, f.readline().split())))
            
            cameras[cam_id]['f'] = f_val
            cameras[cam_id]['k1'] = k1
            cameras[cam_id]['k2'] = k2
            cameras[cam_id]['R'] = R
            cameras[cam_id]['t'] = t
        
        # Parse points with observations
        points = np.empty((num_points, 3), dtype=np.float64)
        observations = []
        
        for point_id in range(num_points):
            position = np.array(list(map(float, f.readline().split())))
            points[point_id] = position
            
            color = list(map(int, f.readline().split()))
            view_list = list(map(float, f.readline().split()))
            
            num_views = int(view_list[0])
            for i in range(num_views):
                cam_idx = int(view_list[1 + 4*i])
                key_idx = int(view_list[2 + 4*i])
                x = view_list[3 + 4*i]
                y = view_list[4 + 4*i]
                observations.append((cam_idx, point_id, x, y))
    
    # Convert to numpy array
    observations = np.array(observations, dtype=np.float32)
    
    return cameras, points, observations

# Create GTSAM factor graph - optimized for large problems
def create_gtsam_graph(cameras, points, observations):
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    print(f"  Building graph with {len(observations)} factors...")
    
    # Camera calibration (use first camera's focal length)
    K = gtsam.Cal3_S2(cameras[0]['f'], cameras[0]['f'], 0, 0, 0)
    
    # Add camera poses
    for cam_id in range(len(cameras)):
        R = gtsam.Rot3(cameras[cam_id]['R'])
        t = gtsam.Point3(cameras[cam_id]['t'])
        pose = gtsam.Pose3(R, t)
        initial.insert(C(cam_id), pose)
    
    # Add 3D points
    for pt_id in range(len(points)):
        initial.insert(L(pt_id), gtsam.Point3(points[pt_id]))
    
    # Add projection factors - shared noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    
    for obs in observations:
        cam_id, pt_id, x, y = int(obs[0]), int(obs[1]), obs[2], obs[3]
        measurement = gtsam.Point2(x, y)
        factor = gtsam.GenericProjectionFactorCal3_S2(
            measurement, measurement_noise, C(cam_id), L(pt_id), K
        )
        graph.add(factor)
    
    return graph, initial

# Run BA optimization and collect metrics
def run_bundle_adjustment(graph, initial):
    monitor = ResourceMonitor()
    
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity("ERROR")
    params.setMaxIterations(50)  # Limit iterations for large problems
    params.setRelativeErrorTol(1e-5)
    params.setAbsoluteErrorTol(1e-5)
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    
    print(f"  Starting optimization...")
    monitor.start()
    start_time = time.time()
    
    try:
        result = optimizer.optimize()
        success = True
    except Exception as e:
        print(f"  Optimization failed: {e}")
        result = initial
        success = False
    
    elapsed = time.time() - start_time
    metrics = monitor.stop()
    
    metrics['time'] = elapsed
    metrics['initial_error'] = graph.error(initial)
    metrics['final_error'] = graph.error(result) if success else float('inf')
    metrics['success'] = success
    
    return result, metrics

# Compute Schur complement sparsity using sparse matrices - most efficient
def compute_schur_sparsity(graph, initial):
    # Count cameras
    num_cameras = sum(1 for key in initial.keys() if gtsam.Symbol(key).chr() == ord('c'))
    
    print(f"  Computing sparsity pattern for {num_cameras} cameras...")
    
    # Use sparse matrix (6 DOF per camera)
    schur_size = 6 * num_cameras
    pattern = lil_matrix((schur_size, schur_size), dtype=bool)
    
    # Build camera-to-landmark connectivity map
    cam_to_landmarks = {}
    for i in range(graph.size()):
        factor = graph.at(i)
        keys = factor.keys()
        
        if len(keys) == 2:  # Projection factor
            cam_key = next((k for k in keys if gtsam.Symbol(k).chr() == ord('c')), None)
            lm_key = next((k for k in keys if gtsam.Symbol(k).chr() == ord('l')), None)
            
            if cam_key and lm_key:
                cam_idx = gtsam.Symbol(cam_key).index()
                lm_idx = gtsam.Symbol(lm_key).index()
                
                if cam_idx not in cam_to_landmarks:
                    cam_to_landmarks[cam_idx] = set()
                cam_to_landmarks[cam_idx].add(lm_idx)
    
    # For each landmark, find all cameras observing it
    lm_to_cams = {}
    for cam_idx, landmarks in cam_to_landmarks.items():
        for lm_idx in landmarks:
            if lm_idx not in lm_to_cams:
                lm_to_cams[lm_idx] = []
            lm_to_cams[lm_idx].append(cam_idx)
    
    # Build Schur complement sparsity: cameras connected through shared landmarks
    for lm_idx, cam_indices in lm_to_cams.items():
        for i, cam_i in enumerate(cam_indices):
            for cam_j in cam_indices[i:]:  # Include diagonal
                # Mark 6x6 block
                pattern[cam_i*6:(cam_i+1)*6, cam_j*6:(cam_j+1)*6] = True
                if cam_i != cam_j:  # Symmetric
                    pattern[cam_j*6:(cam_j+1)*6, cam_i*6:(cam_i+1)*6] = True
    
    # Convert to CSR for efficient counting
    pattern_csr = pattern.tocsr()
    return pattern_csr