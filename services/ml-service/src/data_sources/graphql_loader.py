import pandas as pd
import requests
from typing import List, Dict, Optional, Any, Union

class GraphQLLoader:
    """
    Universal GraphQL Loader.
    Fetches data from a GraphQL endpoint and converts it to a pandas DataFrame.
    """
    
    def __init__(self, 
                 endpoint: str, 
                 headers: Optional[Dict[str, str]] = None,
                 auth: Optional[Any] = None):
        """
        Initialize GraphQL client.
        :param endpoint: GraphQL API endpoint URL.
        :param headers: Default headers to send with every request.
        :param auth: Authentication object (e.g., HTTPBasicAuth) or tuple.
        """
        self.endpoint = endpoint
        self.session = requests.Session()
        if headers:
            self.session.headers.update(headers)
        if auth:
            self.session.auth = auth

    def test_connection(self) -> bool:
        """
        Verify connection by making a simple introspection query.
        """
        query = """
        query {
            __schema {
                queryType {
                    name
                }
            }
        }
        """
        try:
            response = self.execute_query(query)
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a raw GraphQL query and return the JSON response.
        """
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
            
        response = self.session.post(self.endpoint, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if 'errors' in result:
            raise Exception(f"GraphQL Errors: {result['errors']}")
            
        return result

    def fetch_data(self, 
                   query: str, 
                   variables: Optional[Dict] = None, 
                   data_key: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data using a GraphQL query and return as DataFrame.
        
        :param query: The GraphQL query string.
        :param variables: Optional variables for the query.
        :param data_key: Key in the 'data' object where the list of records is located.
                         Supports dot notation (e.g. 'users', 'organization.members').
        """
        result = self.execute_query(query, variables)
        
        data = result.get('data', {})
        
        # Extract data using data_key if provided
        if data_key:
            keys = data_key.split('.')
            for k in keys:
                if isinstance(data, dict) and k in data:
                    data = data[k]
                else:
                    print(f"Warning: Key '{k}' not found in response data.")
                    return pd.DataFrame()
        
        # Normalize data
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            print("Warning: Response data is not a list or dict.")
            return pd.DataFrame()
