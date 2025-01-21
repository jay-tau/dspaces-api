# DataSpaces Python Client

A Python client library for interacting with the DataSpaces API.

## Features

- NumPy array data handling
- Object storage and retrieval
- Variable listing
- N-dimensional bounding box support

## Installation

```bash
pip install numpy requests
```

## Quick Start

```python
import numpy as np
from dspaces_api.client import DSpacesClient, BoundingBox, Interval

# Initialize client
client = DSpacesClient()

# Create sample data
data = np.array([[1, 2], [3, 4]], dtype=np.float32)
box = BoundingBox(bounds=[
    Interval(0, 2),  # rows
    Interval(0, 2)   # columns
])

# Store data
client.put_object("matrix", 1, data, box)

# Retrieve data
retrieved = client.get_object("matrix", 1, box)
print(retrieved)
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
