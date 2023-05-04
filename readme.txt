1. Create conda environment 
conda create --name virtual_museum_guide --file /Users/cong/Downloads/0_ML709_submit/requirements.txt

2. Set OpenAI API key:
Using the API key in the email to replace the API key in the tour_guide_pipeline.py file 

3. Update the path in config.py file

4. Run the system
python tour_guide_pipeline.py

Note:
- There may be a delay in the system's response, and it may take a few seconds for the text to be generated.
- If you want to stop the system, simply type one of the following words when prompted to ask a question: ['end', 'exit', 'stop', 'goodbye'].}