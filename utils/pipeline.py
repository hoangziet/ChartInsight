import bars 
import pies 
import cls 
import pandas as pd 

class ChartPipeline:
    def __init__(self):
        self.id_to_name = {
            0: "bar",
            1: "line",
            2: "pie"
        }
        self.id = None
        self.dataframe = None
    
    def get_chart_type(self, image_path: str) -> str:
        """
        Get the chart type of the image.

        Args:
            image_path (str): Path to the image.

        Returns:
            str: Chart type of the image (bar, line, pie).
        """
        self.id = cls.classify(image_path)
        return self.id_to_name[self.id]
    
    def get_dataframe(self, image_path: str) -> pd.DataFrame:
        """
        Get the dataframe of the data in the image.
        
        Args:
            image_path (str): Path to the image.
        
        Returns:
            pd.DataFrame: Dataframe of the data in the image.
        """
        # get chart type
        self.id = cls.classify(image_path)
        
        # get dataframe
        if self.id == 0:
            self.dataframe = bars.to_dataframe(image_path)
        elif self.id == 2:
            self.dataframe = pies.to_dataframe(image_path)
        return self.dataframe
    