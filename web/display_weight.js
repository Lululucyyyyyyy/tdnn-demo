


window.onload = function(){ 
 
    /**
     * ======================================================
     *          FOR DATA TENSOR DISPLAY
     * ======================================================
     */
    function getData(arr){
        arr = arr.dataSync();
        var numRows = 15;
        var numCols = Math.floor(arr.length / 15);
        var ret = [];
        for(i=0; i<numCols; i++){
            for(j=0; j<numRows; j++){
                // console.log(i, j, i + numRows * j, arr[i + numRows * j]);
                ret.push({x: i+1,
                            y: j+1,
                            heat: arr[i*numRows + j]})
            }
        }
        // console.log('return value', ret);
        return ret;
    }

    function getDataBias(arr){
        arr = arr.dataSync();
        // var numRows = 15;
        var numCols = arr.length;
        var ret = [];
        for(i=0; i<numCols; i++){
            ret.push({x: i+1, y: 1,
                            heat: arr[i]});
        }
        // console.log('return value', ret);
        return ret;
    }

    function getData2(arr){
        arr = arr.dataSync();
        var numRows = 8;
        var numCols = Math.floor(arr.length / numRows);
        var ret = [];
        for(i=0; i<numCols; i++){
            for(j=0; j<numRows; j++){
                // console.log(i, j, i + numRows * j, arr[i + numRows * j]);
                ret.push({x: i+1,
                            y: j+1,
                            heat: arr[i*numRows + j]})
            }
        }
        // console.log('return value', ret);
        return ret;
    }

    anychart.onDocumentReady(function () {
        var layer1weights = tf.unstack(tdnn1Weight, 2);
        for(let i=1; i<=8; i++){
            var container = "plot-container"+(i+8).toString();
            let chart = anychart.heatMap(getData(layer1weights[i-1]));
            let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
            chart.colorScale(colorScale);
            chart.title("Weight of layer 1 ("+i.toString()+")");
            chart.container(container);
            chart.draw();
        }

        let chart = anychart.heatMap(getDataBias(tdnn1Bias));
        let colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        chart.title("Bias of layer 1");
        chart.container("plot-container17");
        chart.draw();

        tf.print(tdnn2Weight);
        var layer2weights = tf.unstack(tdnn2Weight, 2);
        for(let i=1; i<=3; i++){
            var container = "plot-container"+(i+20).toString();
            let chart = anychart.heatMap(getData2(layer2weights[i-1]));
            chart.colorScale(colorScale);
            chart.title("Weight of layer 2 ("+i.toString()+")");
            chart.container(container);
            chart.draw();
        }

        chart = anychart.heatMap(getDataBias(tdnn2Bias));
        colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        chart.title("Bias of layer 2");
        chart.container("plot-container18");
        chart.draw();

        chart = anychart.heatMap(getDataBias(linearWeight));
        colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        chart.title("Linear Weight");
        chart.container("plot-container24");
        chart.draw();

        chart = anychart.heatMap(getDataBias(linearBias));
        colorScale = anychart.scales.linearColor('#FF6978', '#B1EDE8');
        chart.colorScale(colorScale);
        chart.title("Linear Bias");
        chart.container("plot-container25");
        chart.draw();
        
    });
}