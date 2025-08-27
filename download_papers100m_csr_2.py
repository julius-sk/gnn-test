import numpy as np
import scipy.sparse as sp
import os
from pathlib import Path

def load_and_convert_papers100m_compatible():
    """
    Load existing ogbn-papers100M dataset and save in format compatible with your C++ program.
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

    # Set up paths - check common download locations
    possible_paths = [
        Path("/home/labuser/shiju/dataset"),
        Path("./dataset"),
        Path("~/dataset").expanduser(),
        Path("~/.ogb/datasets").expanduser(),
        Path("./ogb_datasets"),
    ]
    
    dataset_dir = None
    for path in possible_paths:
        if path.exists():
            # Check if ogbn-papers100M exists in this directory
            papers100m_path = path / "ogbn_papers100M"
            if papers100m_path.exists():
                dataset_dir = path
                print(f"Found existing dataset at: {dataset_dir}")
                break
    
    if dataset_dir is None:
        # Default to the first path and let OGB find it
        dataset_dir = possible_paths[0]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using dataset directory: {dataset_dir}")
    
    print("Loading existing ogbn-papers100M dataset (no download)...")
    
    try:
        # Load the dataset - OGB will use existing files if available
        dataset = NodePropPredDataset(name="ogbn-papers100M", root=str(dataset_dir))
        graph, labels = dataset[0]
        print("✓ Dataset loaded successfully from existing files!")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("Make sure the dataset is already downloaded in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}/ogbn_papers100M/")
        return False
    
    print("Dataset loaded successfully!")
    
    # Extract data
    edge_index = graph['edge_index']  # Shape: [2, num_edges]
    num_nodes = graph['num_nodes']
    
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {edge_index.shape[1]}")
    
    # Convert to numpy if it's a tensor
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()
    
    print("Creating sparse adjacency matrix...")
    
    # Extract source and destination nodes
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # IMPORTANT: Create symmetric adjacency matrix for undirected graph
    # Add both directions: (i,j) and (j,i)
    all_src = np.concatenate([src_nodes, dst_nodes])
    all_dst = np.concatenate([dst_nodes, src_nodes])
    
    # Create data array (all ones for unweighted graph)
    data = np.ones(len(all_src), dtype=np.float32)
    
    # DEBUG: Check input data before creating COO matrix
    print("\n=== DEBUG: Input data analysis ===")
    print(f"all_src shape: {all_src.shape}, dtype: {all_src.dtype}")
    print(f"all_dst shape: {all_dst.shape}, dtype: {all_dst.dtype}")
    print(f"data shape: {data.shape}, dtype: {data.dtype}")
    print(f"Source range: [{np.min(all_src)}, {np.max(all_src)}]")
    print(f"Dest range: [{np.min(all_dst)}, {np.max(all_dst)}]")
    print(f"Data range: [{np.min(data)}, {np.max(data)}]")
    print(f"Matrix shape will be: {num_nodes} x {num_nodes}")
    
    # Check for invalid indices
    if np.min(all_src) < 0 or np.max(all_src) >= num_nodes:
        print(f"❌ ERROR: Source indices out of bounds!")
        return False
    if np.min(all_dst) < 0 or np.max(all_dst) >= num_nodes:
        print(f"❌ ERROR: Destination indices out of bounds!")
        return False
    
    # Create COO sparse matrix
    print("\n=== Creating COO matrix ===")
    adj_coo = sp.coo_matrix((data, (all_src, all_dst)), 
                           shape=(num_nodes, num_nodes), 
                           dtype=np.float32)
    
    print(f"COO matrix created:")
    print(f"  Shape: {adj_coo.shape}")
    print(f"  NNZ: {adj_coo.nnz}")
    print(f"  Data dtype: {adj_coo.data.dtype}")
    print(f"  Row dtype: {adj_coo.row.dtype}")
    print(f"  Col dtype: {adj_coo.col.dtype}")
    print(f"  Data range: [{np.min(adj_coo.data)}, {np.max(adj_coo.data)}]")
    print(f"  Row range: [{np.min(adj_coo.row)}, {np.max(adj_coo.row)}]")
    print(f"  Col range: [{np.min(adj_coo.col)}, {np.max(adj_coo.col)}]")
    
    # Sample some entries
    print(f"  First 5 COO entries:")
    for i in range(min(5, adj_coo.nnz)):
        print(f"    ({adj_coo.row[i]}, {adj_coo.col[i]}) = {adj_coo.data[i]}")
    
    # CRITICAL: Remove duplicates and sort indices
    print("\n=== Removing duplicates ===")
    print(f"Before sum_duplicates: NNZ = {adj_coo.nnz}")
    adj_coo.sum_duplicates()
    print(f"After sum_duplicates: NNZ = {adj_coo.nnz}")
    
    # DEBUG: Check COO after duplicate removal
    print(f"COO after duplicate removal:")
    print(f"  Data dtype: {adj_coo.data.dtype}")
    print(f"  Data range: [{np.min(adj_coo.data)}, {np.max(adj_coo.data)}]")
    print(f"  First 5 entries after duplicate removal:")
    for i in range(min(5, adj_coo.nnz)):
        print(f"    ({adj_coo.row[i]}, {adj_coo.col[i]}) = {adj_coo.data[i]}")
    
    # Convert to CSR format
    print("\n=== Converting COO to CSR ===")
    print(f"Starting conversion...")
    
    try:
        adj_csr = adj_coo.tocsr()
        print(f"✓ COO to CSR conversion successful")
    except Exception as e:
        print(f"❌ COO to CSR conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # DEBUG: Immediately check CSR after conversion
    print(f"\n=== CSR Matrix Analysis (Raw) ===")
    print(f"CSR shape: {adj_csr.shape}")
    print(f"CSR NNZ: {adj_csr.nnz}")
    print(f"CSR format: {adj_csr.format}")
    print(f"CSR has_sorted_indices: {adj_csr.has_sorted_indices}")
    print(f"CSR has_canonical_format: {adj_csr.has_canonical_format}")
    
    print(f"\nCSR data array:")
    print(f"  Shape: {adj_csr.data.shape}")
    print(f"  Dtype: {adj_csr.data.dtype}")
    print(f"  Size in bytes: {adj_csr.data.nbytes}")
    print(f"  Min: {np.min(adj_csr.data) if len(adj_csr.data) > 0 else 'N/A'}")
    print(f"  Max: {np.max(adj_csr.data) if len(adj_csr.data) > 0 else 'N/A'}")
    print(f"  First 10 values: {adj_csr.data[:10]}")
    print(f"  Last 10 values: {adj_csr.data[-10:]}")
    
    print(f"\nCSR indices array:")
    print(f"  Shape: {adj_csr.indices.shape}")
    print(f"  Dtype: {adj_csr.indices.dtype}")
    print(f"  Size in bytes: {adj_csr.indices.nbytes}")
    print(f"  Min: {np.min(adj_csr.indices) if len(adj_csr.indices) > 0 else 'N/A'}")
    print(f"  Max: {np.max(adj_csr.indices) if len(adj_csr.indices) > 0 else 'N/A'}")
    print(f"  First 10 values: {adj_csr.indices[:10]}")
    print(f"  Last 10 values: {adj_csr.indices[-10:]}")
    
    print(f"\nCSR indptr array:")
    print(f"  Shape: {adj_csr.indptr.shape}")
    print(f"  Dtype: {adj_csr.indptr.dtype}")
    print(f"  Size in bytes: {adj_csr.indptr.nbytes}")
    print(f"  Min: {np.min(adj_csr.indptr)}")
    print(f"  Max: {np.max(adj_csr.indptr)}")
    print(f"  First 10 values: {adj_csr.indptr[:10]}")
    print(f"  Last 10 values: {adj_csr.indptr[-10:]}")
    
    # Check for obvious corruption
    if adj_csr.indptr[0] != 0:
        print(f"❌ ERROR: First indptr should be 0, got {adj_csr.indptr[0]}")
    if adj_csr.indptr[-1] != adj_csr.nnz:
        print(f"❌ ERROR: Last indptr should be {adj_csr.nnz}, got {adj_csr.indptr[-1]}")
    
    # Check if indices are reasonable
    if len(adj_csr.indices) > 0:
        if np.min(adj_csr.indices) < 0:
            print(f"❌ ERROR: Negative column index found: {np.min(adj_csr.indices)}")
        if np.max(adj_csr.indices) >= num_nodes:
            print(f"❌ ERROR: Column index too large: {np.max(adj_csr.indices)} >= {num_nodes}")
    
    # Test matrix-vector multiplication
    print(f"\n=== Testing matrix structure ===")
    try:
        # Create a simple test vector
        test_vec = np.ones(num_nodes, dtype=np.float32)
        print("Testing SpMV with ones vector...")
        
        # Try matrix-vector multiplication
        result = adj_csr.dot(test_vec)
        print(f"✓ SpMV successful, result shape: {result.shape}")
        print(f"Result dtype: {result.dtype}")
        print(f"Result range: [{np.min(result)}, {np.max(result)}]")
        print(f"First 5 result values: {result[:5]}")
        
    except Exception as e:
        print(f"❌ SpMV test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Additional cleanup for CSR format
    print(f"\n=== CSR Cleanup Phase ===")
    print("Before cleanup:")
    print(f"  NNZ: {adj_csr.nnz}")
    print(f"  Has sorted indices: {adj_csr.has_sorted_indices}")
    print(f"  Data sample: {adj_csr.data[:5] if len(adj_csr.data) > 0 else 'empty'}")
    
    # Remove zeros first
    adj_csr.eliminate_zeros()
    print(f"After eliminate_zeros: NNZ = {adj_csr.nnz}")
    
    # Sort indices
    if not adj_csr.has_sorted_indices:
        print("Sorting indices...")
        adj_csr.sort_indices()
        print(f"After sort_indices: has_sorted_indices = {adj_csr.has_sorted_indices}")
    
    # Make it canonical format
    if not adj_csr.has_canonical_format:
        print("Converting to canonical format...")
        adj_csr.eliminate_zeros()
        adj_csr.sort_indices()
        print(f"After canonical format: has_canonical_format = {adj_csr.has_canonical_format}")
    
    # Final check of CSR arrays
    print(f"\n=== Final CSR Arrays Check ===")
    print(f"Data array - shape: {adj_csr.data.shape}, dtype: {adj_csr.data.dtype}")
    print(f"  Min/Max: [{np.min(adj_csr.data) if len(adj_csr.data) > 0 else 'N/A'}, {np.max(adj_csr.data) if len(adj_csr.data) > 0 else 'N/A'}]")
    print(f"  Sample values: {adj_csr.data[:10] if len(adj_csr.data) >= 10 else adj_csr.data}")
    
    print(f"Indices array - shape: {adj_csr.indices.shape}, dtype: {adj_csr.indices.dtype}")
    print(f"  Min/Max: [{np.min(adj_csr.indices) if len(adj_csr.indices) > 0 else 'N/A'}, {np.max(adj_csr.indices) if len(adj_csr.indices) > 0 else 'N/A'}]")
    print(f"  Sample values: {adj_csr.indices[:10] if len(adj_csr.indices) >= 10 else adj_csr.indices}")
    
    print(f"Indptr array - shape: {adj_csr.indptr.shape}, dtype: {adj_csr.indptr.dtype}")
    print(f"  Min/Max: [{np.min(adj_csr.indptr)}, {np.max(adj_csr.indptr)}]")
    print(f"  First 10: {adj_csr.indptr[:10]}")
    print(f"  Last 10: {adj_csr.indptr[-10:]}")
    
    # Check for the suspicious numbers you mentioned
    suspicious_data = adj_csr.data[adj_csr.data > 1e10]
    if len(suspicious_data) > 0:
        print(f"❌ WARNING: Found {len(suspicious_data)} suspiciously large data values!")
        print(f"  Examples: {suspicious_data[:5]}")
    
    suspicious_indices = adj_csr.indices[adj_csr.indices > num_nodes]
    if len(suspicious_indices) > 0:
        print(f"❌ WARNING: Found {len(suspicious_indices)} out-of-bounds indices!")
        print(f"  Examples: {suspicious_indices[:5]}")
    
    suspicious_indptr = adj_csr.indptr[adj_csr.indptr > adj_csr.nnz]
    if len(suspicious_indptr) > 0:
        print(f"❌ WARNING: Found {len(suspicious_indptr)} invalid indptr values!")
        print(f"  Examples: {suspicious_indptr[:5]}")
    
    # Ensure data types are exactly what your C++ program expects
    print("\n=== Data Type Conversion ===")
    print("Converting to required data types...")
    
    # Force explicit conversion with checks
    try:
        print(f"Converting data from {adj_csr.data.dtype} to float32...")
        csr_data = adj_csr.data.astype(np.float32, copy=True)
        print(f"  ✓ Data conversion successful")
        print(f"  New shape: {csr_data.shape}, dtype: {csr_data.dtype}")
        print(f"  New range: [{np.min(csr_data) if len(csr_data) > 0 else 'N/A'}, {np.max(csr_data) if len(csr_data) > 0 else 'N/A'}]")
        print(f"  Sample: {csr_data[:5] if len(csr_data) >= 5 else csr_data}")
    except Exception as e:
        print(f"❌ Data conversion failed: {e}")
        return False
    
    try:
        print(f"Converting indices from {adj_csr.indices.dtype} to int32...")
        csr_indices = adj_csr.indices.astype(np.int32, copy=True)
        print(f"  ✓ Indices conversion successful")
        print(f"  New shape: {csr_indices.shape}, dtype: {csr_indices.dtype}")
        print(f"  New range: [{np.min(csr_indices) if len(csr_indices) > 0 else 'N/A'}, {np.max(csr_indices) if len(csr_indices) > 0 else 'N/A'}]")
        print(f"  Sample: {csr_indices[:5] if len(csr_indices) >= 5 else csr_indices}")
    except Exception as e:
        print(f"❌ Indices conversion failed: {e}")
        return False
    
    try:
        print(f"Converting indptr from {adj_csr.indptr.dtype} to int32...")
        csr_indptr = adj_csr.indptr.astype(np.int32, copy=True)
        print(f"  ✓ Indptr conversion successful")
        print(f"  New shape: {csr_indptr.shape}, dtype: {csr_indptr.dtype}")
        print(f"  New range: [{np.min(csr_indptr)}, {np.max(csr_indptr)}]")
        print(f"  First 5: {csr_indptr[:5]}")
        print(f"  Last 5: {csr_indptr[-5:]}")
    except Exception as e:
        print(f"❌ Indptr conversion failed: {e}")
        return False
    
    # Check for suspicious values in converted arrays
    print(f"\n=== Final Converted Arrays Verification ===")
    
    # Check for the specific suspicious numbers you mentioned
    suspicious_value = 4607182418800017408
    
    # Check data array
    if len(csr_data) > 0:
        suspicious_data_mask = np.abs(csr_data.view(np.int64)) == suspicious_value
        if np.any(suspicious_data_mask):
            print(f"❌ FOUND suspicious value {suspicious_value} in data array!")
            print(f"  Count: {np.sum(suspicious_data_mask)}")
            print(f"  Positions: {np.where(suspicious_data_mask)[0][:5]}")
    
    # Check indices array  
    if len(csr_indices) > 0:
        suspicious_indices_mask = csr_indices.view(np.int64) == suspicious_value
        if np.any(suspicious_indices_mask):
            print(f"❌ FOUND suspicious value {suspicious_value} in indices array!")
            print(f"  Count: {np.sum(suspicious_indices_mask)}")
    
    # Check indptr array
    suspicious_indptr_mask = csr_indptr.view(np.int64) == suspicious_value
    if np.any(suspicious_indptr_mask):
        print(f"❌ FOUND suspicious value {suspicious_value} in indptr array!")
        print(f"  Count: {np.sum(suspicious_indptr_mask)}")
    
    # Additional memory/corruption checks
    print(f"\n=== Memory and Corruption Checks ===")
    
    # Check if arrays are contiguous
    print(f"Array memory layout:")
    print(f"  Data contiguous: {csr_data.flags['C_CONTIGUOUS']}")
    print(f"  Indices contiguous: {csr_indices.flags['C_CONTIGUOUS']}")
    print(f"  Indptr contiguous: {csr_indptr.flags['C_CONTIGUOUS']}")
    
    # Check for NaN or Inf values
    if len(csr_data) > 0:
        nan_count = np.sum(np.isnan(csr_data))
        inf_count = np.sum(np.isinf(csr_data))
        if nan_count > 0:
            print(f"❌ Found {nan_count} NaN values in data array")
        if inf_count > 0:
            print(f"❌ Found {inf_count} Inf values in data array")
    
    # Memory usage summary
    print(f"\n=== Memory Usage Summary ===")
    data_mb = csr_data.nbytes / (1024**2)
    indices_mb = csr_indices.nbytes / (1024**2)
    indptr_mb = csr_indptr.nbytes / (1024**2)
    total_mb = data_mb + indices_mb + indptr_mb
    
    print(f"Data array: {data_mb:.2f} MB")
    print(f"Indices array: {indices_mb:.2f} MB") 
    print(f"Indptr array: {indptr_mb:.2f} MB")
    print(f"Total: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
    
    # Test array access patterns
    print(f"\n=== Testing Array Access Patterns ===")
    try:
        # Test reading first few elements
        print("Testing array access...")
        if len(csr_data) > 0:
            print(f"  First data element: {csr_data[0]} (type: {type(csr_data[0])})")
        if len(csr_indices) > 0:
            print(f"  First indices element: {csr_indices[0]} (type: {type(csr_indices[0])})")
        print(f"  First indptr element: {csr_indptr[0]} (type: {type(csr_indptr[0])})")
        
        # Test basic slicing
        if len(csr_data) > 10:
            slice_test = csr_data[:10]
            print(f"  Data slice [0:10]: {slice_test}")
        
    except Exception as e:
        print(f"❌ Array access test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Array access tests passed")
    
    # Validate the CSR format
    print(f"\n=== CSR Format Validation ===")
    if not validate_csr_format(csr_data, csr_indices, csr_indptr, num_nodes):
        print("❌ CSR validation failed!")
        return False
    
    print(f"\n=== Pre-Save Final Check ===")
    print(f"About to save arrays with:")
    print(f"  csr_data: shape={csr_data.shape}, dtype={csr_data.dtype}, size={csr_data.nbytes} bytes")
    print(f"  csr_indices: shape={csr_indices.shape}, dtype={csr_indices.dtype}, size={csr_indices.nbytes} bytes")  
    print(f"  csr_indptr: shape={csr_indptr.shape}, dtype={csr_indptr.dtype}, size={csr_indptr.nbytes} bytes")
    
    # Double-check for corruption one more time
    print("Final corruption check...")
    if len(csr_data) > 0:
        print(f"  Data - min: {np.min(csr_data)}, max: {np.max(csr_data)}, first: {csr_data[0]}")
    if len(csr_indices) > 0:
        print(f"  Indices - min: {np.min(csr_indices)}, max: {np.max(csr_indices)}, first: {csr_indices[0]}")
    print(f"  Indptr - min: {np.min(csr_indptr)}, max: {np.max(csr_indptr)}, first: {csr_indptr[0]}, last: {csr_indptr[-1]}")
    
    print("Saving in format compatible with your C++ program...")
    
    csr_file = dataset_dir / "ogbn_papers100m_adj_csr.npz"
    
    # Save with exact data types your program expects - with additional error handling
    try:
        print(f"Attempting to save to: {csr_file}")
        np.savez_compressed(str(csr_file),
                           data=csr_data,      # float32
                           indices=csr_indices,  # int32
                           indptr=csr_indptr)    # int32
        print("✓ File save operation completed")
        
        # Immediately verify the file was written correctly
        print("Immediately verifying written file...")
        if not csr_file.exists():
            print("❌ File was not created!")
            return False
            
        file_size = csr_file.stat().st_size
        print(f"✓ File exists, size: {file_size} bytes ({file_size/(1024**3):.2f} GB)")
        
    except Exception as e:
        print(f"❌ File save failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"✓ CSR matrix saved to: {csr_file}")
    print(f"  Matrix size: {num_nodes} x {num_nodes}")
    print(f"  Non-zeros: {len(csr_data)}")
    print(f"  Data type: {csr_data.dtype}")
    print(f"  Indices type: {csr_indices.dtype}") 
    print(f"  Indptr type: {csr_indptr.dtype}")
    print(f"  File size: {csr_file.stat().st_size / (1024**3):.2f} GB")
    
    # Verify the saved file
    print("\nVerifying saved file...")
    verify_npz_file(str(csr_file))
    
    return True

def validate_csr_format(data, indices, indptr, num_nodes):
    """Validate CSR format correctness"""
    print("Validating CSR format...")
    
    # Check dimensions
    if len(indptr) != num_nodes + 1:
        print(f"❌ indptr length {len(indptr)} != {num_nodes + 1}")
        return False
    
    # Check that indptr is non-decreasing
    if not np.all(np.diff(indptr) >= 0):
        print("❌ indptr is not non-decreasing")
        return False
    
    # Check first and last indptr values
    if indptr[0] != 0:
        print(f"❌ First indptr value {indptr[0]} != 0")
        return False
    
    if indptr[-1] != len(data):
        print(f"❌ Last indptr value {indptr[-1]} != data length {len(data)}")
        return False
    
    # Check indices bounds
    if len(indices) > 0:
        if np.min(indices) < 0 or np.max(indices) >= num_nodes:
            print(f"❌ Column indices out of bounds: min={np.min(indices)}, max={np.max(indices)}")
            return False
    
    # Check data and indices have same length
    if len(data) != len(indices):
        print(f"❌ Data length {len(data)} != indices length {len(indices)}")
        return False
    
    # Check for sorted indices within each row (optional but recommended)
    for i in range(num_nodes):
        start = indptr[i]
        end = indptr[i + 1]
        if end > start:
            row_indices = indices[start:end]
            if not np.all(np.diff(row_indices) >= 0):
                print(f"⚠️ Warning: Row {i} has unsorted column indices")
                # Don't fail validation for this, but warn
    
    print("✓ CSR format validation passed")
    return True

def verify_npz_file(npz_path):
    """Verify the NPZ file matches your program's expectations"""
    print(f"Loading and verifying: {npz_path}")
    
    try:
        data = np.load(npz_path)
        
        print("✓ File loaded successfully")
        print("Available keys:", list(data.keys()))
        
        # Check required keys exist
        required_keys = ['data', 'indices', 'indptr']
        missing_keys = [key for key in required_keys if key not in data.keys()]
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return False
        
        # Check data array
        data_array = data['data']
        print(f"Data array: shape={data_array.shape}, dtype={data_array.dtype}")
        if data_array.dtype != np.float32:
            print(f"⚠️ Warning: Data dtype is {data_array.dtype}, expected float32")
        
        # Check indices array  
        indices_array = data['indices']
        print(f"Indices array: shape={indices_array.shape}, dtype={indices_array.dtype}")
        if indices_array.dtype != np.int32:
            print(f"⚠️ Warning: Indices dtype is {indices_array.dtype}, expected int32")
        
        # Check indptr array
        indptr_array = data['indptr']
        print(f"Indptr array: shape={indptr_array.shape}, dtype={indptr_array.dtype}")
        if indptr_array.dtype != np.int32:
            print(f"⚠️ Warning: Indptr dtype is {indptr_array.dtype}, expected int32")
        
        # Calculate matrix dimensions
        num_rows = len(indptr_array) - 1
        nnz = len(indices_array)
        
        print(f"Matrix: {num_rows} x {num_rows}")
        print(f"Non-zeros: {nnz}")
        print(f"Density: {nnz / (num_rows * num_rows) * 100:.6f}%")
        
        # Validate CSR format
        if not validate_csr_format(data_array, indices_array, indptr_array, num_rows):
            return False
        
        # Test data access (like your C++ program does)
        print("\nTesting data access (simulating C++ program):")
        print(f"First few data values: {data_array[:min(5, len(data_array))]}")
        print(f"First few indices: {indices_array[:min(5, len(indices_array))]}")
        print(f"First few indptr: {indptr_array[:min(5, len(indptr_array))]}")
        print(f"Last few indptr: {indptr_array[-min(3, len(indptr_array)):]}")
        
        # Test a few rows
        print("\nSample row data:")
        for i in range(min(3, num_rows)):
            start = indptr_array[i]
            end = indptr_array[i + 1]
            row_nnz = end - start
            if row_nnz > 0:
                row_indices = indices_array[start:end]
                row_data = data_array[start:end]
                print(f"  Row {i}: {row_nnz} non-zeros, indices: {row_indices[:min(5, len(row_indices))]}")
        
        print("✓ Verification complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error verifying file: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Remove duplicates by summing
    coo_matrix.sum_duplicates()
    
    # Convert to CSR
    csr_matrix = coo_matrix.tocsr()
    
    # Clean up the matrix
    csr_matrix.eliminate_zeros()
    csr_matrix.sort_indices()
    
    # Ensure correct data types
    csr_data = csr_matrix.data.astype(np.float32)
    csr_indices = csr_matrix.indices.astype(np.int32)
    csr_indptr = csr_matrix.indptr.astype(np.int32)
    
    # Validate
    if not validate_csr_format(csr_data, csr_indices, csr_indptr, size):
        print("❌ Test matrix validation failed!")
        return None
    
    # Save in compatible format
    test_file = "/home/labuser/shiju/dataset/test_small_csr.npz"
    np.savez_compressed(test_file,
                       data=csr_data,
                       indices=csr_indices,
                       indptr=csr_indptr)
    
    print(f"✓ Small test matrix saved to: {test_file}")
    print(f"  Size: {size} x {size}")
    print(f"  NNZ: {len(csr_data)}")
    print(f"  Density: {len(csr_data) / (size * size) * 100:.2f}%")
    
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
        if 'data' not in data:
            print("❌ No 'data' array found")
            return False
        if 'indices' not in data:
            print("❌ No 'indices' array found")
            return False
        if 'indptr' not in data:
            print("❌ No 'indptr' array found")
            return False
            
        # Convert to correct data types
        data_array = data['data'].astype(np.float32)
        indices_array = data['indices'].astype(np.int32)
        indptr_array = data['indptr'].astype(np.int32)
        
        # Validate the format
        num_rows = len(indptr_array) - 1
        if not validate_csr_format(data_array, indices_array, indptr_array, num_rows):
            print("❌ Input file has invalid CSR format!")
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
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("NPZ Data Generator for C++ SpMV Program")
    print("=" * 50)
    
    # Create dataset directory
    os.makedirs("/home/labuser/shiju/dataset", exist_ok=True)
    
    print("Choose an option:")
    print("1. Load existing ogbn-papers100M and convert to compatible format")
    print("2. Create small test matrix for debugging")
    print("3. Convert existing NPZ file to compatible format")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        success = load_and_convert_papers100m_compatible()
        if success:
            print("\n" + "=" * 50)
            print("SUCCESS! Your C++ program should now work with:")
            print("/home/labuser/shiju/dataset/ogbn_papers100m_adj_csr.npz")
            print("\nRun your program:")
            print("sudo ./spmv_multi 72 /home/labuser/shiju/dataset/ogbn_papers100m_adj_csr.npz")
        
    elif choice == "2":
        test_file = create_smaller_test_matrix()
        if test_file:
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
