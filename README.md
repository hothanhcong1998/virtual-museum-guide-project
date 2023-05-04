# Virtual Museum Tour Guide

The Virtual Museum Tour Guide is a project that utilizes machine learning techniques to create a virtual tour guide that can provide visitors with information about museum objects. The project leverages the Louvre Abu Dhabi dataset, which contains videos and information about museum objects, to build a sample virtual tour guide.

## Getting Started

### Installation

1. Clone the repository
```
git clone https://github.com/hothanhcong1998/virtual-museum-guide-project.git
```

2. Install the required packages using pip
```
pip install -r requirements.txt
```
or
```
conda create --name virtual_museum_guide --file requirements.txt
```

### Usage

1. Set OpenAI API key:
Using the API key in the email to replace the API key in the tour_guide_pipeline.py file 

2. Update the path in config.py file

3. Run the system
```
python tour_guide_pipeline.py
```

Note:
- There may be a delay in the system's response, and it may take a few seconds for the text to be generated.
- If you want to stop the system, simply type one of the following words when prompted to ask a question: ['end', 'exit', 'stop', 'goodbye'].}
