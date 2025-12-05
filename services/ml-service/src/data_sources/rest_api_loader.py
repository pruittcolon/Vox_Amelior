import pandas as pd
import requests
from typing import List, Dict, Optional, Any, Union

class RestApiLoader:
    """
    Universal REST API Loader.
    Fetches data from a REST endpoint and converts it to a pandas DataFrame.
    """
    
    def __init__(self, 
                 base_url: Optional[str] = None, 
                 headers: Optional[Dict[str, str]] = None,
                 auth: Optional[Any] = None):
        """
        Initialize REST API client.
        :param base_url: Optional base URL for requests.
        :param headers: Default headers to send with every request.
        :param auth: Authentication object (e.g., HTTPBasicAuth) or tuple.
        """
        self.base_url = base_url.rstrip('/') if base_url else ""
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)
        if auth:
            self.session.auth = auth

    def test_connection(self, endpoint: str = "") -> bool:
        """
        Verify connection by making a HEAD or GET request.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if self.base_url else endpoint
        try:
            response = self.session.get(url, timeout=10)
            return response.ok
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def fetch_data(self, 
                   endpoint: str, 
                   params: Optional[Dict] = None, 
                   data_key: Optional[str] = None,
                   method: str = "GET",
                   json_body: Optional[Dict] = None) -> pd.DataFrame:
        """
        Fetch data from an endpoint and return as DataFrame.
        
        :param endpoint: API endpoint (relative to base_url if set, or absolute).
        :param params: Query parameters.
        :param data_key: Key in the JSON response where the list of records is located. 
                         If None, assumes the root response is the list.
                         Supports dot notation for nested keys (e.g. 'data.items').
        :param method: HTTP method (GET, POST, etc.).
        :param json_body: JSON body for POST requests.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if self.base_url else endpoint
        
        try:
            response = self.session.request(method, url, params=params, json=json_body, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract data using data_key if provided
            if data_key:
                keys = data_key.split('.')
                for k in keys:
                    if isinstance(data, dict) and k in data:
                        data = data[k]
                    else:
                        # Key not found or structure mismatch
                        print(f"Warning: Key '{k}' not found in response.")
                        return pd.DataFrame()
            
            # Normalize data
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                print("Warning: Response data is not a list or dict.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise e
