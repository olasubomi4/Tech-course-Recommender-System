import pandas as pd
import json
class ResponseObject:
    def __init__(self):
        self.__responseMessage=""
        self.__responseStatus=False
        self.__data=None

    def setResponseMessage(self,message):
        self.__responseMessage=message

    def setResponseStatus(self,status):
        self.__responseStatus=status

    def getResponseMessage(self):
        return self.__responseMessage

    def getResponseStatus(self):
        return self.__responseStatus

    def getData(self):
        return self.__data

    def setData(self,data):
        self.__data=data


    def jsonfyResponse(self):
        if self.__data is None:
            self.__data="{}"
        response = {
            "responseMessage": self.getResponseMessage(),
            "responseStatus": self.getResponseStatus(),
            "responseBody":json.loads(self.getData())

        }
        return response