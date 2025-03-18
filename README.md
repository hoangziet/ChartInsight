#  ChartInsight  

A web application that allows users to upload a chart, identify its type, extract values.  

## Features  
- Upload a chart as an image (PNG, JPG)  
- Detect chart types (pie, bar, line)  
- Extract data from charts  

##  Installation  
###  Clone the repository  
```bash
git clone https://github.com/hoangziet/ChartInsight.git
cd ChartInsight
```  
###  Install dependencies  
```bash
pip install -r requirements.txt  # Install necessary libraries
```  
###  Run the application  
```bash
python main.py  
```  

##  Project Structure  
```
/ChartInsight
│── /assets          # Contains dataset and other static resources
│   │── /dataset     # Folder for storing the dataset 
│── /models          # Chart detection models
│── /static          # Static files (CSS, images)
│── /templates       # HTML templates for the web app
│── /utils           # Utility functions
│── main.py          # API for chart recognition
│── README.md        # Documentation
│── requirements.txt # Python dependencies
```  

##  Dataset  

**Dataset used:** [DeepRuleDataset](https://huggingface.co/datasets/niups/DeepRuleDataset/tree/main).  

| Data Type | Chart Type |  
|-----------|------------|  
| 0         | Bar        |  
| 1         | Line       |  
| 2         | Pie        |  

### Instructions for Downloading and Processing the Data  
1. **Download** the dataset from the link above.  
2. **Save** the data inside the `assets/dataset` folder.  
3. **Extract** the downloaded files and move all extracted folders into the `dataset` directory.  
4. **Run** the `reduce_data.ipynb` notebook inside the same folder to process and reduce the dataset size.  

## Technologies Used  
- **Backend:** Flask
- **Image Processing:** OpenCV, PaddleOCR, Matplotlib  
- **Machine Learning:** TensorFlow, Scikit-learn  
- **Data Handling:** Pandas, NumPy  
- **Frontend:** HTML, CSS, JavaScript
- **Development:** Docker (optional), Git/GitHub, Jupyter Notebook

## Roadmap  
- Improve chart recognition accuracy  
- Standardize extracted data  
- Integrate LLM to answer questions about the chart (Coming soon) 

##  License  
This project is licensed under the MIT License.  

##  References  
This work is based on:  
1. Liu, Xiaoyi, Klabjan, Diego, & Bless, Patrick N. (2019). *Data Extraction from Charts via Single Deep Neural Network*. arXiv preprint arXiv:1906.11906. [DOI](https://doi.org/10.48550/arXiv.1906.11906).  
2. [Using Hourglass Networks To Understand Human Poses](https://medium.com/towards-data-science/using-hourglass-networks-to-understand-human-poses-1e40e349fa15)  
3. [deep-vision / Hourglass](https://github.com/ethanyanjiali/deep-vision/tree/master/Hourglass)

   
