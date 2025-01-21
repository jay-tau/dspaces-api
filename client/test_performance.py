import numpy as np
import pytest
from client import DSpacesClient, BoundingBox, Interval, DSpacesError

@pytest.fixture(autouse=True)
def setup_teardown():
    """Cleanup before and after each test"""
    client = DSpacesClient()
    # Ensure server is responsive before running tests
    try:
        client.get_variables()
    except Exception as e:
        pytest.skip(f"DSpaces server not available: {e}")
    
    yield
    
    # Cleanup after test
    try:
        variables = client.get_variables()
        for var in variables:
            if var.startswith("perf_test_"):
                objects = client.get_objects(var)
                for obj in objects:
                    client.delete_object(obj.name, obj.version)
    except Exception:
        pass  # Best effort cleanup

@pytest.fixture
def client():
    return DSpacesClient(debug=True)  # Enable debug mode for troubleshooting

def generate_random_array(shape, dtype=np.float32):
    return np.random.random(shape).astype(dtype)

@pytest.mark.parametrize("shape", [(10, 10), (100, 100), (1000, 1000)])
def test_2d_write_performance(benchmark, client, shape):
    """Benchmark 2D array writes of different sizes"""
    def write_array():
        data = generate_random_array(shape)
        box = BoundingBox(bounds=[
            Interval(0, shape[0]),
            Interval(0, shape[1])
        ])
        client.put_object(f"perf_test_2d_{shape[0]}", 1, data, box)
    
    benchmark.extra_info['shape'] = f"{shape[0]}x{shape[1]}"
    benchmark(write_array)

@pytest.mark.parametrize("shape", [(10, 10), (100, 100), (1000, 1000)])
def test_2d_read_performance(benchmark, client, shape):
    """Benchmark 2D array reads of different sizes"""
    # Setup: Write test data first
    data = generate_random_array(shape)
    box = BoundingBox(bounds=[
        Interval(0, shape[0]),
        Interval(0, shape[1])
    ])
    client.put_object(f"perf_test_2d_read_{shape[0]}", 1, data, box)
    
    def read_array():
        return client.get_object(f"perf_test_2d_read_{shape[0]}", 1, box)
    
    benchmark.extra_info['shape'] = f"{shape[0]}x{shape[1]}"
    benchmark(read_array)

@pytest.mark.parametrize("size", [(10, 10, 10), (50, 50, 50)])
def test_3d_slice_performance(benchmark, client, size):
    """Benchmark 3D array slice operations"""
    # Setup: Create and store full array
    shape = (100, 100, 100)
    data = generate_random_array(shape)
    full_box = BoundingBox(bounds=[
        Interval(0, shape[0]),
        Interval(0, shape[1]),
        Interval(0, shape[2])
    ])
    client.put_object("perf_test_3d", 1, data, full_box)
    
    slice_box = BoundingBox(bounds=[
        Interval(0, size[0]),
        Interval(0, size[1]),
        Interval(0, size[2])
    ])
    
    def read_slice():
        return client.get_object("perf_test_3d", 1, slice_box)
    
    benchmark.extra_info['slice_size'] = f"{size[0]}x{size[1]}x{size[2]}"
    benchmark(read_slice)

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_datatype_performance(benchmark, client, dtype):
    """Benchmark performance with different data types"""
    shape = (50, 50)
    data = generate_random_array(shape, dtype)

    box = BoundingBox(bounds=[
        Interval(0, shape[0]),
        Interval(0, shape[1])
    ])
    
    def write_and_read():
        obj_name = f"perf_test_dtype_{dtype.__name__}"
        # Verify data properties before sending
        print(f"\nTest Debug: Original data dtype={data.dtype}, "
              f"itemsize={data.itemsize}, shape={data.shape}, "
              f"nbytes={data.nbytes}")
        
        client.put_object(obj_name, 1, data, box)
        result = client.get_object(obj_name, 1, box)
        
        # Verify data properties after receiving
        print(f"Test Debug: Result data dtype={result.dtype}, "
              f"itemsize={result.itemsize}, shape={result.shape}, "
              f"nbytes={result.nbytes}")
        
        if not np.allclose(data, result):  # Use allclose instead of array_equal for floats
            raise ValueError(f"Data mismatch for {dtype.__name__}")
        return result
    
    benchmark.extra_info.update({
        'dtype': dtype.__name__,
        'itemsize': dtype().itemsize,
        'shape': f"{shape[0]}x{shape[1]}"
    })
    benchmark(write_and_read)

def test_concurrent_access(benchmark, client):
    """Benchmark concurrent read/write operations"""
    shape = (50, 50)
    iterations = 10
    
    def concurrent_ops():
        data = generate_random_array(shape)
        box = BoundingBox(bounds=[
            Interval(0, shape[0]),
            Interval(0, shape[1])
        ])
        
        for i in range(iterations):
            client.put_object(f"perf_test_concurrent", i, data, box)
            _ = client.get_object(f"perf_test_concurrent", i, box)
    
    benchmark.extra_info['iterations'] = iterations
    benchmark(concurrent_ops)
