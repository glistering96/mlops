from unittest import TestCase, main

import requests

class TestServer(TestCase):
    def setUp(self):
        self.host = "http://0.0.0.0:20000"
        return super().setUp()
    
    def test_connection(self):
        res = requests.get(self.host + "/api/v1/check_running")
        
        self.assertTrue(
            res.status_code == 200
        )
        
        self.assertTrue(
            res.json()
        )
        

if __name__ == "__main__":
    main()