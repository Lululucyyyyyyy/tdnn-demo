# tdnn-demo

## Documentation

Model 1 is trained on the entire dataset9 (many words and different vowels)

Model 2 is also trained on the entire dataset9 except the test cases are normalized on a case-to-case basis. 

Model 3 is trained on dataset9 and dataset11 but only on "bah" "dah" "gah" soungs.

To use a model in the demo, run the following line to generate a web_model/tfjs_model folder.

tensorflowjs_converter --input_format=keras --output_format tfjs_layers_model  model2.h5 web_model/tfjs_model

The folder contains a json and a binary file. Drag this entire folder into the web/ directory to use.

