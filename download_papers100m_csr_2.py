import numpy as np
import scipy.sparse as sp
import os
from pathlib import Path

def download_and_convert_papers100m_compatible():
    """
    Download ogbn-papers100M and save in format compatible with your C++ program.
    Your program expects:
    - data: float32
    - indices: int32 
    - indptr: int32
    """
    try:
        from ogb.nodeproppred import NodePropPredDataset
        import torch
    except ImportError:
        print("Error: Required packages not found.")
        print("Please install with: pip install ogb torch")
        return False

    # Set up paths
    dataset_dir = Path("/home/labuser/shiju/dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading ogbn-papers100M dataset...")
    
    # Load the dataset
    dataset = NodePropPredDataset(name="ogbn-papers100M", root=str(dataset_dir))
    graph, labels = dataset[0]
    
    print("Dataset loaded successfully!")
    
    # Extract data
    edge_index = graph['edge_index']  # Shape: [2, num_edges]
    num_nodes = graph['num_nodes']
    
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {edge_index.shape[1]}")
    
    # Convert to numpy if it's a tensor
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()
    
    # Create adjacency matrix in COO format
    print("Creating sparse adjacency matrix...")
    
    # Extract source and destination nodes
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # Create data array (all ones for unweighted graph)
    data = np.ones(len(src_nodes), dtype=np.float32)  # IMPORTANT: float32
    
    # Create COO sparse matrix
    adj_coo = sp.coo_matrix((data, (src_nodes, dst_nodes)), 
                           shape=(num_nodes, num_nodes), 
                           dtype=np.float32)
    
    # Convert to CSR format
    print("Converting to CSR format...")
    adj_csr = adj_coo.tocsr()
    
    print("Saving in format compatible with your C++ program...")
    
    # CRITICAL: Save with exact data types your program expects
    csr_file = dataset_dir / "ogbn_papers100m_adj_csr.npz"
    
    # Your program expects:
    # - data as float32
    # - indices as int32
    # - indptr as int32
    np.savez_compressed(str(csr_file),
                       data=adj_csr.data.astype(np.float32),      # float32
                       indices=adj_csr.indices.astype(np.int32),  # int32
                       indptr=adj_csr.indptr.astype(np.int32))    # int32
    
    print(f"✓ CSR matrix saved to: {csr_file}")
    print(f"  Data type: {adj_csr.data.dtype} -> float32")
    print(f"  Indices type: {adj_csr.indices.dtype} -> int32") 
    print(f"  Indptr type: {adj_csr.indptr.dtype} -> int32")
    print(f"  File size: {csr_file.stat().st_size / (1024**3):.2f} GB")
    
    # Verify the saved file
    print("\nVerifying saved file...")
    verify_npz_file(str(csr_file))
    
    return True

def verify_npz_file(npz_path):
    """Verify the NPZ file matches your program's expectations"""
    print(f"Loading and verifying: {npz_path}")
    
    try:
        data = np.load(npz_path)
        
        print("✓ File loaded successfully")
        print("Available keys:", list(data.keys()))
        
        # Check data array
        data_array = data['data']
        print(f"Data array: shape={data_array.shape}, dtype={data_array.dtype}")
        if data_array.dtype != np.float32:
            print(f"⚠ Warning: Data dtype is {data_array.dtype}, expected float32")
        
        # Check indices array  
        indices_array = data['indices']
        print(f"Indices array: shape={indices_array.shape}, dtype={indices_array.dtype}")
        if indices_array.dtype != np.int32:
            print(f"⚠ Warning: Indices dtype is {indices_array.dtype}, expected int32")
        
        # Check indptr array
        indptr_array = data['indptr']
        print(f"Indptr array: shape={indptr_array.shape}, dtype={indptr_array.dtype}")
        if indptr_array.dtype != np.int32:
            print(f"⚠ Warning: Indptr dtype is {indptr_array.dtype}, expected int32")
        
        # Calculate matrix dimensions
        num_rows = len(indptr_array) - 1
        nnz = len(indices_array)
        
        print(f"Matrix: {num_rows} x {num_rows}")
        print(f"Non-zeros: {nnz}")
        print(f"Density: {nnz / (num_rows * num_rows) * 100:.6f}%")
        
        # Test data access (like your C++ program does)
        print("\nTesting data access (simulating C++ program):")
        print(f"First few data values: {data_array[:5]}")
        print(f"First few indices: {indices_array[:5]}")
        print(f"First few indptr: {indptr_array[:5]}")
        print(f"Last indptr: {indptr_array[-1]}")
        
        # Verify CSR format integrity
        if indptr_array[0] != 0:
            print(f"⚠ Warning: First indptr should be 0, got {indptr_array[0]}")
        if indptr_array[-1] != nnz:
            print(f"⚠ Warning: Last indptr should be {nnz}, got {indptr_array[-1]}")
        
        print("✓ Verification complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error verifying file: {e}")
        return False

def create_smaller_test_matrix():
    """Create a smaller test matrix for debugging"""
    print("Creating smaller test matrix for debugging...")
    
    # Create a small 1000x1000 matrix for testing
    size = 1000
    density = 0.01  # 1% density
    
    # Generate random sparse matrix
    nnz = int(size * size * density)
    row_indices = np.random.randint(0, size, nnz)
    col_indices = np.random.randint(0, size, nnz)
    data = np.random.rand(nnz).astype(np.float32)
    
    # Create COO matrix and convert to CSR
    coo_matrix = sp.coo_matrix((data, (row_indices, col_indices)), 
                              shape=(size, size), dtype=np.float32)
    csr_matrix = coo_matrix.tocsr()
    
    # Remove duplicates and sort
    csr_matrix.eliminate_zeros()
    csr_matrix.sort_indices()
    
    # Save in compatible format
    test_file = "/home/labuser/shiju/dataset/test_small_csr.npz"
    np.savez_compressed(test_file,
                       data=csr_matrix.data.astype(np.float32),
                       indices=csr_matrix.indices.astype(np.int32),
                       indptr=csr_matrix.indptr.astype(np.int32))
    
    print(f"✓ Small test matrix saved to: {test_file}")
    print(f"  Size: {size} x {size}")
    print(f"  NNZ: {csr_matrix.nnz}")
    print(f"  Density: {csr_matrix.nnz / (size * size) * 100:.2f}%")
    
    # Verify
    verify_npz_file(test_file)
    
    return test_file

def convert_existing_dataset(input_path, output_path):
    """Convert an existing NPZ file to the format your program expects"""
    print(f"Converting {input_path} to compatible format...")
    
    try:
        # Load existing file
        data = np.load(input_path)
        
        print("Original file contents:")
        for key in data.keys():
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Extract arrays
        if 'data' in data:
            data_array = data['data'].astype(np.float32)
        else:
            print("No 'data' array found")
            return False
            
        if 'indices' in data:
            indices_array = data['indices'].astype(np.int32)
        else:
            print("No 'indices' array found")
            return False
            
        if 'indptr' in data:
            indptr_array = data['indptr'].astype(np.int32)
        else:
            print("No 'indptr' array found")
            return False
        
        # Save in compatible format
        np.savez_compressed(output_path,
                           data=data_array,
                           indices=indices_array,
                           indptr=indptr_array)
        
        print(f"✓ Converted file saved to: {output_path}")
        print("New data types:")
        print(f"  data: {data_array.dtype}")
        print(f"  indices: {indices_array.dtype}")
        print(f"  indptr: {indptr_array.dtype}")
        
        # Verify the converted file
        verify_npz_file(output_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Error converting file: {e}")
        return False

if __name__ == "__main__":
    print("NPZ Data Generator for C++ SpMV Program")
    print("=" * 50)
    
    # Create dataset directory
    os.makedirs("/home/labuser/shiju/dataset", exist_ok=True)
    
    print("Choose an option:")
    print("1. Download ogbn-papers100M and convert to compatible format")
    print("2. Create small test matrix for debugging")
    print("3. Convert existing NPZ file to compatible format")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        success = download_and_convert_papers100m_compatible()
        if success:
            print("\n" + "=" * 50)
            print("SUCCESS! Your C++ program should now work with:")
            print("/home/labuser/shiju/dataset/ogbn_papers100m_adj_csr.npz")
            print("\nRun your program:")
            print("sudo ./spmv_multi 72 /home/labuser/shiju/dataset/ogbn_papers100m_adj_csr.npz")
        
    elif choice == "2":
        test_file = create_smaller_test_matrix()
        print("\n" + "=" * 50)
        print("SUCCESS! Test with the small matrix first:")
        print(f"sudo ./spmv_multi 72 {test_file}")
        
    elif choice == "3":
        input_file = input("Enter path to existing NPZ file: ").strip()
        output_file = input("Enter path for converted file: ").strip()
        
        if not output_file:
            output_file = "/home/labuser/shiju/dataset/converted_csr.npz"
        
        success = convert_existing_dataset(input_file, output_file)
        if success:
            print(f"\nSUCCESS! Use converted file: {output_file}")
    
    else:
        print("Invalid choice")