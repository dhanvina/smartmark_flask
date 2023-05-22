## Automated Marks Updation System

[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/yourusername/yourrepository/blob/main/LICENSE)

> This project aims to automate the process of updating student marks by leveraging deep learning, machine learning, and computer vision techniques. It significantly reduces the time and effort required for teachers to update marks individually for each student.

### Features

- **Front Sheet Processing**: The system extracts relevant information from the front sheet of the answer script using computer vision techniques. This includes student details, subject codes, and other necessary information.

- **Marks Calculation**: By utilizing machine learning algorithms, the system accurately calculates the marks obtained by the student based on the processed information. It takes into account subject-specific criteria and weightage to determine the final marks.

- **CSV/XLSX File Generation**: The system generates a CSV or XLSX file containing the student details and their corresponding marks. This file can be easily uploaded to the website or any grading system, ensuring quick and seamless updates.

- **Time Efficiency**: With this automated system, teachers can update marks for multiple students in a single shot, significantly reducing the time spent on manual data entry. It eliminates the need for individual mark entry, saving approximately 5-7 minutes per student's paper.

- **Future Enhancement**: The project aims to develop a user-friendly mobile application for teachers, providing them with easy accessibility and usability. This app will further enhance the efficiency of the marks updation process.

### Technologies Used

- Python Flask
- Deep Learning (Neural Networks)
- Machine Learning (Scikit-learn, TensorFlow, or PyTorch)
- Computer Vision (OpenCV)
- CSV/XLSX File Handling

### Getting Started

#### Prerequisites

- Python 
- Flask 
- Install other necessary libraries by running the command: `pip install -r requirements.txt`

#### Installation

1. Clone the repository: `git clone https://github.com/dhanvina/smartmark_flask`
2. Navigate to the project directory: `cd smartmark_flask`
3. Install the required dependencies: `pip install -r requirements.txt`

#### Usage

1. Run the Flask application: `python app.py`
2. Open your web browser and go to `http://localhost:5000` to access the application.
3. Upload the front sheet image of the answer script.
4. The system will process the image, calculate the marks, and generate a CSV/XLSX file.
5. Download the file and update the marks in the website or grading system.

### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or suggestions.

