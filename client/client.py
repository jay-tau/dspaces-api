import requests
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import warnings

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

class DSpacesError(Exception):
    pass

class DSpacesClient:
    def __init__(self, base_url="http://localhost:8001", debug=False):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.debug = debug
        
        # Only support floating point types
        self.type_map = {
            np.dtype('float32'): 6,
            np.dtype('float64'): 7,
        }
        
        # Server element sizes for floating point types
        self.element_sizes = {
            6: 4,  # float32
            7: 8,  # float64
        }
        
        self.reverse_type_map = {v: k for k, v in self.type_map.items()}

    def put_object(self, obj_name: str, obj_version: int, data: np.ndarray, 
                  box: BoundingBox, namespace: Optional[str] = None) -> None:
        """Store data to DataSpaces per schema"""
        # Convert integer types to float32
        if np.issubdtype(data.dtype, np.integer):
            warnings.warn(f"Integer type {data.dtype} is not supported, converting to float32")
            data = data.astype(np.float32)
        
        if data.dtype not in self.type_map:
            raise DSpacesError(f"Unsupported data type: {data.dtype}. "
                             f"Supported types: {list(self.type_map.keys())}")

        dtype_num = self.type_map[data.dtype]
        params = {
            'element_size': self.element_sizes[dtype_num],
            'element_type': dtype_num,
        }
        
        if self.debug:
            print(f"PUT Debug: dtype={data.dtype}, element_size={params['element_size']}, "
                  f"shape={data.shape}, total_size={data.nbytes}")

        # Per schema: box should be form data, not a file
        form_data = {
            'box': json.dumps({
                'bounds': [
                    {'start': b.start, 'span': b.span} 
                    for b in box.bounds
                ]
            })
        }

        # Data must be sent as binary file
        files = {
            'data': ('data', data.tobytes(), 'application/octet-stream')
        }

        self._make_request(
            'PUT',
            f'/dspaces/obj/{obj_name}/{obj_version}',
            params=params,
            data=form_data,  # Send box as form data
            files=files      # Send data as file
        )

    def get_object(self, obj_name: str, obj_version: int, 
                  box: BoundingBox, namespace: Optional[str] = None) -> np.ndarray:
        """Retrieve a DataSpaces object per schema"""
        params = {}
        if namespace:
            params['namespace'] = namespace

        # Format request body according to schema
        body = {
            'bounds': [
                {'start': b.start, 'span': b.span} 
                for b in box.bounds
            ]
        }

        response = self._make_request('POST', 
                                    f'/dspaces/obj/{obj_name}/{obj_version}',
                                    params=params,
                                    json=body,
                                    return_raw=True)

        # Parse response headers according to schema
        if 'X-DS-Tag' not in response.headers:
            raise DSpacesError("Missing element type in response")
        if 'X-DS-Dims' not in response.headers:
            raise DSpacesError("Missing dimensions in response")

        dtype_num = int(response.headers['X-DS-Tag'])
        if dtype_num not in self.reverse_type_map:
            raise DSpacesError(f"Unsupported element type: {dtype_num}")

        element_size = self.element_sizes[dtype_num]
        dims = [int(d) for d in response.headers['X-DS-Dims'].split(',')]
        expected_size = np.prod(dims) * element_size

        if self.debug:
            print(f"GET Debug: dtype={self.reverse_type_map[dtype_num]}, "
                  f"element_size={element_size}, dims={dims}, "
                  f"content_size={len(response.content)}, "
                  f"expected_size={expected_size}")

        if len(response.content) != expected_size:
            raise DSpacesError(
                f"Data size mismatch: got {len(response.content)} bytes, "
                f"expected {expected_size} bytes for shape {dims} "
                f"and dtype {self.reverse_type_map[dtype_num]}"
            )

        return np.frombuffer(response.content, 
                           dtype=self.reverse_type_map[dtype_num]).reshape(dims)

    def get_variables(self) -> List[str]:
        """Get list of variables per schema"""
        return self._make_request('GET', '/dspaces/var/')

    def get_objects(self, obj_name: str, namespace: Optional[str] = None) -> List[DSObject]:
        """Get objects for variable per schema"""
        params = {'namespace': namespace} if namespace else {}
        response = self._make_request('GET', f'/dspaces/var/{obj_name}', params=params)
        return [DSObject(**obj) for obj in response]

    def delete_object(self, obj_name: str, obj_version: int) -> None:
        """Delete an object from DSpaces"""
        self._make_request('DELETE', f'/dspaces/obj/{obj_name}/{obj_version}')

    def _make_request(self, method: str, endpoint: str, return_raw=False, **kwargs) -> Any:
        """Handle HTTP requests and errors with improved error reporting"""
        try:
            response = self.session.request(method, f"{self.base_url}{endpoint}", **kwargs)
            response.raise_for_status()
            
            if return_raw:
                return response
            return response.json() if response.content else None
            
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'json'):
                try:
                    error_detail = e.response.json()
                    raise DSpacesError(f"API request failed: {error_detail}")
                except json.JSONDecodeError:
                    pass
            raise DSpacesError(f"API request failed: {str(e)}")

def main():
    """Demonstrate DSpaces client with results"""
    client = DSpacesClient()
    
    try:
        # 1. Store a simple 2D array
        data_2d = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], dtype=np.float32)
        
        box_2d = BoundingBox(bounds=[
            Interval(0, 2),
            Interval(0, 2)
        ])
        
        client.put_object("test_matrix", 1, data_2d, box_2d)
        print("Stored array:\n", data_2d)
        
        # 2. List available variables
        variables = client.get_variables()
        print("\nAvailable variables:", variables)
        
        # 3. Store larger 2D array
        data_2d_large = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        
        box_2d_large = BoundingBox(bounds=[
            Interval(0, 3),
            Interval(0, 3)
        ])
        
        client.put_object("matrix2d", 1, data_2d_large, box_2d_large)
        print("\nStored larger array:\n", data_2d_large)
        
        # 4. Store 3D array subset
        data_3d = np.random.rand(4, 4, 4).astype(np.float32)
        print("\nFull 3D array (4x4x4):")
        for z in range(4):
            print(f"\nz = {z}:")
            print(data_3d[z])
            
        subset = data_3d[1:3, 1:3, 1:3]
        print("\nSubset of 3D array (2x2x2) from position (1,1,1):")
        for z in range(2):
            print(f"\nz = {z}:")
            print(subset[z])
            
        box_3d = BoundingBox(bounds=[
            Interval(1, 2),
            Interval(1, 2),
            Interval(1, 2)
        ])
        
        client.put_object("volume", 1, subset, box_3d)
        print("\nStored 3D array subset:\n", subset)
        
        # 5. Retrieve 2D array and compare
        retrieved_2d = client.get_object("matrix2d", 1, box_2d_large)
        print("\nRetrieved 2D array:\n", retrieved_2d)
        print("Arrays match:", np.array_equal(data_2d_large, retrieved_2d))
        
        # 6. Get object metadata
        objects = client.get_objects("matrix2d")
        print("\nObject metadata:")
        for obj in objects:
            print(f"- Name: {obj.name}")
            print(f"  Version: {obj.version}")
            print(f"  Bounds: {obj.bounds}")
        
        return 0
        
    except DSpacesError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return 1

if __name__ == "__main__":
    exit(main())