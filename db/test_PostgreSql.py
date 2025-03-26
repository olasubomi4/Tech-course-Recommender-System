import unittest
import pandas as pd
from db.PostgreSql import PostgreSql

class PostgreSqlTest(unittest.TestCase):
    postgreSql = PostgreSql()
    tableName = "Iphone 16 unit test"
    def setUp(self):
        data = {
            "productId": [2, 1, 3],
            "productName": ["Iphone 16", "Iphone 16 pro", "Iphone 16 pro max"],
            "productColor": ["Black", "White", "Blue"]
        }
        self.sampleDataFrame = pd.DataFrame(data)
    def tearDown(self):
        print(f"Dropping table '{self.tableName}' now")
        self.postgreSql.dropTable(self.tableName)

    def test_insert_products(self):
        result=self.postgreSql.insertProducts(self.sampleDataFrame,self.tableName)
        assert result == True

    def test_retrieve_table_as_data_frame(self):
        self.sampleDataFrame= self.sampleDataFrame[self.sampleDataFrame['productId'] != 2]
        self.sampleDataFrame['id'] = range(1, len(self.sampleDataFrame) + 1)
        self.sampleDataFrame.set_index('id', inplace=True)
        # self.sampleDataFrame= self.sampleDataFrame.reset_index(drop=True)
        self.postgreSql.insertProducts(self.sampleDataFrame,self.tableName)
        result=self.postgreSql.retrieveTableAsDataFrame(self.tableName)
        pd.testing.assert_frame_equal(result,self.sampleDataFrame)

