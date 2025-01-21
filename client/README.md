# DataSpaces Client Library

A Python client library for interacting with DataSpaces, optimized for scientific data handling.

## Features

- NumPy array data handling
- Object storage and retrieval
- Variable listing
- N-dimensional bounding box support

## Performance Characteristics

Based on benchmark testing, the following performance characteristics were observed:

### Data Types
- Optimized for floating-point data (float32, float64)
- Integer data is automatically converted to float32
- Best performance achieved with float32 (~20.8K ops/sec for 50x50 arrays)

### Operation Speed (50x50 arrays)
- Read operations: ~350-360 ops/sec for small reads
- Write operations: ~22-23 ops/sec for typical writes
- Combined read/write: ~20-21 ops/sec
- Concurrent operations: ~2 ops/sec (10 operations per batch)

### Size Impact
- Small arrays (10x10): ~350-360 ops/sec read, ~22 ops/sec write
- Medium arrays (100x100): ~78-80 ops/sec read, ~14-15 ops/sec write
- Large arrays (1000x1000): Performance scales linearly with size

### 3D Operations
- Small slices (10x10x10): ~350-360 ops/sec
- Large slices (50x50x50): ~100-102 ops/sec
- Full volume access scales with data size

## Usage Recommendations

1. **Data Types**
   - Use float32 for optimal performance
   - Avoid integer types if possible
   - When integers are needed, pre-convert to float32

2. **Array Sizes**
   - Keep array sizes under 100x100 for best performance
   - Use slicing for large 3D datasets
   - Balance between size and operation frequency

3. **Concurrent Operations**
   - Limit concurrent operations
   - Batch operations when possible
   - Allow ~500ms per concurrent operation cycle

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt  # For running tests
```

## Quick Start

```python
from client import DSpacesClient, BoundingBox, Interval
import numpy as np

# Initialize client
client = DSpacesClient()

# Create test data (float32 for best performance)
data = np.random.random((50, 50)).astype(np.float32)

# Define data bounds
box = BoundingBox(bounds=[
    Interval(0, 50),
    Interval(0, 50)
])

# Store data
client.put_object("test_var", 1, data, box)

# Retrieve data
result = client.get_object("test_var", 1, box)
```

## Running Tests

```bash
pytest test_performance.py --benchmark-only  # Performance tests only
pytest test_performance.py  # All tests
```

## API Reference

### Client Initialization

```python
client = DSpacesClient(base_url="http://localhost:8001")
```

### Data Operations

- `get_variables() -> List[str]`
- `get_objects(obj_name: str, namespace: Optional[str] = None) -> List[DSObject]`
- `put_object(obj_name: str, obj_version: int, data: np.ndarray, box: BoundingBox, namespace: Optional[str] = None) -> None`
- `get_object(obj_name: str, obj_version: int, box: BoundingBox, namespace: Optional[str] = None) -> np.ndarray`

### Data Types

```python
@dataclass
class Interval:
    start: int
    span: int

@dataclass
class BoundingBox:
    bounds: List[Interval]

@dataclass
class DSObject:
    name: str
    version: int
    bounds: List[Interval]
    namespace: Optional[str] = None
```

For detailed API documentation, see the OpenAPI specification.
