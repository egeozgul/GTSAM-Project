import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_graph_file(filename):
    """Parse the graph file and extract vertices and edges."""
    vertices = {}
    edges = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                if parts[0] == 'VERTEX2':
                    # Format: VERTEX2 id x y z
                    vertex_id = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    vertices[vertex_id] = np.array([x, y, z])
                
                elif parts[0] == 'EDGE2':
                    # Format: EDGE2 start end ...
                    start = int(parts[1])
                    end = int(parts[2])
                    edges.append((start, end))
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    
    return vertices, edges

def visualize_graph(vertices, edges, filename, max_vertices=100):
    """Create 3D visualization of the graph."""
    if vertices is None or edges is None:
        return
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract vertex coordinates (first max_vertices)
    vertex_ids = sorted(vertices.keys())[:max_vertices]
    coords = np.array([vertices[vid] for vid in vertex_ids])
    # Set z-coordinates to 0
    coords[:, 2] = 0
    
    # Plot vertices
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
               c='red', marker='o', s=50, label='Vertices', zorder=5)
    
    # Plot edges (only between vertices in the first 100)
    vertex_ids_set = set(vertex_ids)
    for start, end in edges:
        if start in vertex_ids_set and end in vertex_ids_set:
            p1 = vertices[start].copy()
            p2 = vertices[end].copy()
            p1[2] = 0
            p2[2] = 0
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   'b-', alpha=0.6, linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Graph Visualization\n({len(vertex_ids)} vertices, {len(edges)} edges)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = "M10000_P_toro.graph"
    vertices, edges = read_graph_file(filename)
    
    if vertices and edges:
        print(f"Loaded {len(vertices)} vertices and {len(edges)} edges")
        visualize_graph(vertices, edges, filename, max_vertices=1000)
    else:
        print("Failed to load graph data.")
