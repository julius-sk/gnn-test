import numpy as np
import scipy.sparse as sp
import os
from pathlib import Path

def load_and_convert_papers100m_simple():
    """Simple conversion - no overcomplicated debugging"""
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError:
        print("pip install ogb torch")
        return False

    dataset_dir = Path("/home/labuser/shiju/dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    dataset = NodePropPredDataset(name="ogbn-papers100M", root=str(dataset_dir))
    graph, labels = dataset[0]
    
    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']
    
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()
    
    print(f"Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")
    
    # Create symmetric adjacency matrix
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    data = np.ones(len(src), dtype=np.float32)
    
    # COO -> CSR
    coo_matrix = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    csr_matrix = coo_matrix.tocsr()
    
    # Check if we need int64 for large matrices
    use_int64 = csr_matrix.indptr[-1] > 2**31 - 1
    index_dtype = np.int64 if use_int64 else np.int32
    
    print(f"Using {'int64' if use_int64 else 'int32'} for indices/indptr")
    
    # Save
    output_file = dataset_dir / "ogbn_papers100m_adj_csr.npz"
    np.savez_compressed(output_file,
                       data=csr_matrix.data.astype(np.float32),
                       indices=csr_matrix.indices.astype(index_dtype),
                       indptr=csr_matrix.indptr.astype(index_dtype))
    
    print(f"✓ Saved: {output_file}")
    print(f"Size: {output_file.stat().st_size / (1024**3):.2f} GB")
    return True

def create_small_test():
    """Small test matrix"""
    size = 1000
    nnz = 10000
    
    rows = np.random.randint(0, size, nnz)
    cols = np.random.randint(0, size, nnz)
    data = np.ones(nnz, dtype=np.float32)
    
    coo = sp.coo_matrix((data, (rows, cols)), shape=(size, size))
    csr = coo.tocsr()
    
    test_file = "/home/labuser/shiju/dataset/test_small.npz"
    np.savez_compressed(test_file,
                       data=csr.data.astype(np.float32),
                       indices=csr.indices.astype(np.int32),
                       indptr=csr.indptr.astype(np.int32))
    
    print(f"✓ Test file: {test_file}")
    return test_file

if __name__ == "__main__":
    print("1. Convert ogbn-papers100M")
    print("2. Create small test")
    
    choice = input("Choice: ").strip()
    
    if choice == "1":
        load_and_convert_papers100m_simple()
    elif choice == "2":
        create_small_test()
