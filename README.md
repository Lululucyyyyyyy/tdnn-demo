# tdnn-demo

## Set up
The dependencies are listed in requirements.txt. To install, run
```
pip3 install -r requirements.txt
```
To run the demo, make sure you are on the demo branch. To switch to the demo branch, run
```
git checkout demo
```

## How to train the model
After you set the parameters in modeltf.py and modify the model number, run 
```
python3 modeltf.py
```

## How to deploy the model
This will generate a .h5 file with the model number that you specified. To convert this h5 model into a javascript model for model <n>, run 
```
tensorflowjs_converter --input_format=keras --output_format tfjs_layers_model  model<n>.h5 web_model/tfjs_model
```
This will create a folder named tfjs_model/ which can be found in web_model. It contains a json file and a binary file, both of which are needed. Copy the entire folder called tfjs_model/ into web/. Then run the web/ folder on a local or deployed server.

## Data Collection
To collect data, make sure to be on the main branch 
```
git checkout main
```
Once the webpage is loaded, click the "switch" button to switch from "record sample" to "start file write". One in "start file write", click the "start file write" button, and it will take a 1 second recording and download this as "data.txt". 

For the same word, you can download as many as you need. Then, drag and drop them into a folder, for example named samples/. Then, to use mytools:
```
python3 -i mytools.py
```
This will open an interactive shell in python. To name the samples, run
```
name('samples/', 'bah', 0)
```
The first parameter 'samples/' is a folder where all your samples are stored. The second parameter 'bah' is the word you said, and the third parameter 0 is which number to start naming at. For example, if you already recorded 20 samples of 'bah' and have more, you can choose to start at 20 (it is inclusive). Note that this function is 0-indexed.

Then, we need to extract the start times by aligning the dataset. This function aligns it by calculating the sum of the frequency intensities, normalizing so that the values are ranged from 0 to 1, then finding the first value where the frequency sum ratio is greater than a threshold=0.1. The start time is 20ms or 2 frames before that so we can capture the onset. To align the dataset, run
```
align_dataset('samples/', comp=False)
```

Refer to the previous datasets for an example on how data is stored. It should follow this structure:
```
dataset<n>/
    b/
        bah00000-200.txt
        bah00001-250.txt
    d/
        dah00000-200.txt
        dah00001-200.txt
    g/
        gah00000-200.txt
        gah00001-200.txt
```


