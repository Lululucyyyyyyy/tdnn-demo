var layer1out;
var layer2out;
var flattened;
var linear;


window.onload = function(){ 
 
/**
 * ======================================================
 *          FOR DATA TENSOR DISPLAY
 * ======================================================
 */
function getData(){
    var currDat = localStorage.getItem("currDat");
    var ret = [];
    currDat = currDat.split(",");
    // console.log(currDat.length);
    var numRows = 15;
    var numCols = Math.floor(currDat.length / 15);
    // console.log(numCols);
    for(i=0; i<numCols; i++){
        for(j=0; j<numRows; j++){
            // console.log(i, j, i + numRows * j,currDat[i + numRows * j]);
            ret.push({x: j,
                        y: i,
                        heat: currDat[i + j*numRows]})
        }
    }
    // console.log('return value', ret)
    return ret;
}

anychart.onDocumentReady(function () {
    // create a heatmap
    let chart = anychart.heatMap(getData());

    let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
    // let colorScale = anychart.scales.linearColor('FFFF00', '#FF0000');
    chart.colorScale(colorScale);

    // name the heatmap
    chart.title("16 frequencies over 15 10-ms time frames");
    // set the container for the heatmap
    chart.container("plot-container1");
    // draw the heatmap
    chart.draw();
});

function getNormedData(){
    var dataTensorNormedArr = localStorage.getItem("dataTensorNormed");
    var ret = [];
    dataTensorNormed = dataTensorNormedArr.replace(/[\[\]']+/g, '')
    dataTensorNormed = dataTensorNormed.split(",");
    // console.log(dataTensorNormed);
    var numRows = 15;
    var numCols = Math.floor(dataTensorNormed.length / 15);
    // console.log(numCols);
    for(i=0; i<numCols; i++){
        for(j=0; j<numRows; j++){
            let a = dataTensorNormed[i + j*numRows].replace(/[{()}]/g, '');
            // console.log(i, j, i + numRows * j,a);
            ret.push({x: j,
                        y: i,
                        heat: a});
        }
    }
    // console.log('return value', ret)
    return ret;
}

document.getElementById('normed').onclick = () => {
    console.log('showing normalized data');
    anychart.onDocumentReady(function () {
        // create a heatmap
        let chart = anychart.heatMap(getNormedData());
        let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        // name the heatmap
        chart.title("16 frequencies over 15 10-ms time frames: Normalized Data");
        // set the container for the heatmap
        chart.container("plot-container2");
        // draw the heatmap
        chart.draw();
    });
    document.getElementById('tdnn1').style = "display: flex;"
    document.getElementById('normed').style = "display: none;"
 }

 function getTdnn1(){
    var dataTensorNormed = JSON.parse(localStorage.getItem("dataTensorNormed"));
    dataTensorNormed = tf.tensor(dataTensorNormed);
    // console.log(dataTensorNormed.shape);
    dataTensorNormed = tf.transpose(dataTensorNormed, [0, 2 ,1]);

    var tdnn1 = tf.layers.conv1d({inputShape: [15, 16],
        filters: 8,
        kernelSize: 3,
        dilationRate: 1, // 0 - (-1)
        padding: 'valid', // no padding 
        });
    var dummy = tdnn1.apply(dataTensorNormed);
    // tf.print(dummy);
    tdnn1.setWeights([tdnn1Weight, tdnn1Bias]);
    layer1out = tdnn1.apply(dataTensorNormed);
    // tf.print(layer1out.shape);
    var layer1outArr = tf.reshape(layer1out, [13, 8]);
    layer1outArr = Array.from(layer1outArr.arraySync());
    ret = []
    var numRows = 8;
    var numCols = 13;
    for(i=0; i<numCols; i++){
        for(j=0; j<numRows; j++){
            // console.log(layer1outArr[i][j])
            ret.push({x: j,
                        y: i,
                        heat: layer1outArr[i][j]})
        }
    }
    return ret;
}

function getTdnn1Sigmoid(){
    layer1out = tf.layers.activation({activation: 'sigmoid'}).apply(layer1out);
    var layer1outArr = tf.reshape(layer1out, [13, 8]);
    layer1outArr = Array.from(layer1outArr.arraySync());
    ret = []
    var numRows = 8;
    var numCols = 13;
    for(i=0; i<numCols; i++){
        for(j=0; j<numRows; j++){
            // console.log(layer1outArr[i][j])
            ret.push({x: j,
                        y: i,
                        heat: layer1outArr[i][j]})
        }
    }
    return ret;
}

document.getElementById('tdnn1').onclick = () => {
    anychart.onDocumentReady(function () {
        // create a heatmap
        let chart = anychart.heatMap(getTdnn1());
        let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        // name the heatmap
        chart.title("Layer1");
        // set the container for the heatmap
        chart.container("plot-container3");
        // draw the heatmap
        chart.draw();
    });

    anychart.onDocumentReady(function () {
        // create a heatmap
        let chart = anychart.heatMap(getTdnn1Sigmoid());
        let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        // name the heatmap
        chart.title("Layer1 after Sigmoid");
        // set the container for the heatmap
        chart.container("plot-container4");
        // draw the heatmap
        chart.draw();
    });

    document.getElementById('tdnn2').style = "display: flex;"
    document.getElementById('tdnn1').style = "display: none;"
 }

 function getTdnn2(){
   var tdnn2 = tf.layers.conv1d({inputShape: [13, 8],
        filters: 3,
        kernelSize: 3,
        dilationRate: 2, // 0 - (-2)
        padding: 'valid', // no padding
        // dataFormat: 'channelsFirst'
        // bias term ?
        });
    var dummy = tdnn2.apply(layer1out);
    // tf.print(dummy);
    tdnn2.setWeights([tdnn2Weight, tdnn2Bias]);
    layer2out = tdnn2.apply(layer1out);
    // tf.print(layer1out.shape);
    var layer2outArr = tf.reshape(layer2out, [9, 3]);
    layer2outArr = Array.from(layer2outArr.arraySync());
    ret = []
    var numRows = 3;
    var numCols = 9;
    for(i=0; i<numCols; i++){
        for(j=0; j<numRows; j++){
            // console.log(layer1outArr[i][j])
            ret.push({x: j,
                        y: i,
                        heat: layer2outArr[i][j]})
        }
    }
    return ret;
}

 function getTdnn2Sigmoid(){
    layer2out = tf.layers.activation({activation: 'sigmoid'}).apply(layer2out);
    var layer2outArr = tf.reshape(layer2out, [9, 3]);
    layer2outArr = Array.from(layer2outArr.arraySync());
    ret = []
    var numRows = 3;
    var numCols = 9;
    for(i=0; i<numCols; i++){
        for(j=0; j<numRows; j++){
            // console.log(layer1outArr[i][j])
            ret.push({x: j,
                        y: i,
                        heat: layer2outArr[i][j]})
        }
    }
    return ret;
}

function getFlatten(){
    var flatten_layer = tf.layers.flatten({dataFormat: 'channelsFirst'});
    flattened = flatten_layer.apply(layer2out);
    var flattenedArr = tf.reshape(flattened, [27]);
    flattenedArr = Array.from(flattenedArr.arraySync());
    ret = []
    var numRows = 27;
    var numCols = 0;
    for(j=0; j<numRows; j++){
        // console.log(layer1outArr[i][j])
        ret.push({x: j,
                    y: 1,
                    heat: flattenedArr[j]})
    }
    return ret;
}

document.getElementById('tdnn2').onclick = () => {
    anychart.onDocumentReady(function () {
        // create a heatmap
        let chart = anychart.heatMap(getTdnn2());
        let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        // name the heatmap
        chart.title("Layer2");
        // set the container for the heatmap
        chart.container("plot-container5");
        // draw the heatmap
        chart.draw();
    });

    anychart.onDocumentReady(function () {
        // create a heatmap
        let chart = anychart.heatMap(getTdnn2Sigmoid());
        let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        // name the heatmap
        chart.title("Layer2 after Sigmoid");
        // set the container for the heatmap
        chart.container("plot-container6");
        // draw the heatmap
        chart.draw();
    });

    document.getElementById('flatten').style = "display: flex;"
    document.getElementById('tdnn2').style = "display: none;"
 }

 function getLinear(){
    var linear_layer = tf.layers.dense({inputDim: 27,
        units: 3,
        useBias: true});
    var dummy = linear_layer.apply(flattened);
    linear_layer.setWeights([linearWeight, linearBias]);
    linear = linear_layer.apply(flattened);
    linear = tf.reshape(linear, [3]);
    var linearArr = Array.from(linear.arraySync());
    ret = []
    var numRows = 3;
    var numCols = 0;
    for(j=0; j<numRows; j++){
        // console.log(layer1outArr[i][j])
        ret.push({x: j,
                    y: 1,
                    heat: linearArr[j]})
    }
    return ret;
}

 document.getElementById('flatten').onclick = () => {
    anychart.onDocumentReady(function () {
        // create a heatmap
        let chart = anychart.heatMap(getFlatten());
        let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        // name the heatmap
        chart.title("Flattened");
        // set the container for the heatmap
        chart.container("plot-container7");
        // draw the heatmap
        chart.draw();
    });

    document.getElementById('linear').style = "display: flex;"
    document.getElementById('flatten').style = "display: none;"
 }

 document.getElementById('linear').onclick = () => {
    anychart.onDocumentReady(function () {
        // create a heatmap
        let chart = anychart.heatMap(getLinear());
        let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        // name the heatmap
        chart.title("linear");
        // set the container for the heatmap
        chart.container("plot-container8");
        // draw the heatmap
        chart.draw();
    });

    document.getElementById('final').style = "display: flex;"
    document.getElementById('linear').style = "display: none;"
 }

 document.getElementById('final').onclick = () => {
    var linearArr = Array.from(linear.arraySync());
    var b = linearArr[0];
    var d = linearArr[1];
    var g = linearArr[2];
    var max = 0;
    if(b > d && b > g){
        // class 0 - b
        max = 0;
    } else if (d > b && d > g){
        // class 1 - d
        max = 1;
    } else if (g > b && g > d){
        // class 2 - g
        max = 2
    }
    var nth = ["first", "second", "third"];
    var the_class = ['b', 'd', 'g'];
    var explain_str = "Notice the "+nth[max]+"value, "+String(linearArr[max].toFixed(2)); 
    explain_str += " is the greatest value. This means that it predicts class "+String(max)+", which corresponds to class " + the_class[max] + '.';

    document.getElementById('final-explanation').innerHTML = explain_str;
    document.getElementById('final').style = "display: none;"
 }

};