Polymer('g-spectrogram-mini', {
  // Show the controls UI.
  controls: false,
  // Log mode.
  log: true,
  // Show axis labels, and how many ticks.
  labels: false,
  ticks: 5,
  speed: 2,
  // FFT bin size,
  fftsize: 2048,
  oscillator: false,
  color: false,
  going: true,
  writing: false,
  recorded_data: [],
  file_naming_idx: 0,
  file_download: true,
  thresh: 0.3,
  start_time_ms: -1,
  explaining: false,
  dataTensorNormed: tf.zeros([16, 15]),
  data_whole: tf.zeros([16, 1], dtype='float32'),
  frames_since_last_coloured: 0,
  custom_start_time_ms: -1,
  amplitude_over_thresh: false,
  amplitude_thresh: -1500,
  prev_max: 0,
  stopped: false, 

  // current data, 15 frames of 16 frequency bins
  currDat: tf.zeros([16, 15], dtype='float32'),
  currDat2: tf.zeros([16, 1], dtype='float32'),
  sampledFreqs: [126.2,  275.2, 451.1, 
                  658.6, 903.6, 1192.8, 
                  1534.1, 2412.5, 2973.7, 
                  3636.2, 4418.1, 5341, 
                  6430.3, 7716.1, 9233.7],
  sampledIdx: [5, 12, 19, 28, 39, 51, 65, 103, 127, 155, 189, 228, 274, 329, 394],
  sampledIdxBuckets: [0, 8, 15, 33, 45, 58, 84, 115, 141, 172, 208, 251, 201, 362, 500],
  
  attachedCallback: async function() {
    this.tempCanvas = document.createElement('canvas'),
    this.tempCanvas2 = document.createElement('canvas'),
    console.log('Created spectrogram');
    // console.log('cur dat', this.currDat);

    // Require user gesture before creating audio context, etc.
    window.addEventListener('mousedown', () => this.createAudioGraph());
    window.addEventListener('touchstart', () => this.createAudioGraph());
  },

  extractFrequencies: function(){
    this.analyser.getFloatFrequencyData(this.freq2);
    //this.freq2 = this.freq2.map(x => Math.pow(10, x / 10)); // matlab transformation
    const predFrequencies = Array(16).fill(0);
    var currChunk, numElems;
    count = 0
    idx = 0
    sum = 0;
    var sampledIdxTemp = this.sampledIdxBuckets;
    for (i = 0; i < sampledIdxTemp.length - 1; i++){
      currChunk = this.freq2.slice(sampledIdxTemp[i], sampledIdxTemp[i + 1]);
      numElems = sampledIdxTemp[i + 1] - sampledIdxTemp[i];
      predFrequencies[i] = currChunk.reduce((partialSum, a) => partialSum + a, 0) / numElems;
      if (predFrequencies[i] == 0){
        predFrequencies[i] = this.freq2.slice(this.sampledIdx[i]);
        if (predFrequencies[i] == 0){
          predFrequencies = Math.min(predFrequencies);
        }
      }
    }
    // console.log("this.freq2", this.freq2);
    // console.log("pred freq", predFrequencies);
    return predFrequencies;
  },

  extractFrequenciesByte: function(){
    this.analyser.getByteFrequencyData(this.freq);
    const predFrequencies = Array(16).fill(0);
    var currChunk, numElems;
    count = 0
    idx = 0
    sum = 0;
    var sampledIdxTemp = this.sampledIdxBuckets;
    for (i = 0; i < sampledIdxTemp.length - 1; i++){
      currChunk = this.freq.slice(sampledIdxTemp[i], sampledIdxTemp[i + 1]);
      numElems = sampledIdxTemp[i + 1] - sampledIdxTemp[i];
      predFrequencies[i] = currChunk.reduce((partialSum, a) => partialSum + a, 0) / numElems;
      if (predFrequencies[i] == 0){
        predFrequencies[i] = this.freq.slice(this.sampledIdx[i]);
        if (predFrequencies[i] == 0){
          predFrequencies = Math.min(predFrequencies);
        }
      }
    }
    // console.log("this.freq", this.freq);
    // console.log("pred freq", predFrequencies);
    return predFrequencies;
  },

  sumColumns: async function(matrix) {
    const numRows = matrix.length;
    const numCols = matrix[0].length; // Assuming all rows have the same number of columns
    
    const columnSums = new Array(numCols).fill(0);
  
    for (let col = 0; col < numCols; col++) {
      for (let row = 0; row < numRows; row++) {
        columnSums[col] += 10**(matrix[row][col]);
      }
    }

    return columnSums;
  },

  argwhere: async function(array) {
    const indices = [];
    for (let i = 2; i < array.length; i++) {
      if (array[i] > this.thresh) {
        indices.push(i);
      }
    }
    return indices;
  },

  customMax: async function(arguments) {
    if (arguments.length === 0) {
      return undefined; // Return undefined if no arguments are provided
    }
  
    let max = -Infinity; // Start with a very low value
    for (let i = 1; i < arguments.length; i++) {
      if (arguments[i] > max) {
        max = arguments[i];
      }
    }
    return max;
  },

  findMaxFreq: async function(data){
    this.start_time_ms = -1;

    this.sumColumns(data, axis=0).then((col_sums) => {
      this.customMax(col_sums).then((max_col_sum) => {
        var array_2 = Array(col_sums);
        for(var i = 0, length = col_sums.length; i < length; i++){
            array_2[i] = col_sums[i] / max_col_sum;
        }
        console.log(array_2);
        this.argwhere(array_2).then((thresh_indexes) => {
          start_time_ms = thresh_indexes[0]*10 - 20;
          // to capture onset in msec
          console.log(start_time_ms);
          this.start_time_ms = start_time_ms;
        });
      });
    });
  },

  storeData: async function(){
    localStorage.setItem("currDat", self.currDat);
    localStorage.setItem("dataTensorNormed", self.dataTensorNormed);
    console.log('stored');
    // return (self.currDat, self.dataTensorNormed);
  },

  predictModel: async function(data){
    // converts from a canvas data object to a tensor

    this.start_time_ms = -1;
 
    // sum columns
    var matrix = data
    const numRows = matrix.length;
    const numCols = matrix[0].length; // Assuming all rows have the same number of columns
    
    const columnSums = new Array(numCols).fill(0);
  
    for (let col = 0; col < numCols; col++) {
      for (let row = 0; row < numRows; row++) {
        columnSums[col] += 10**(matrix[row][col]);
      }
    }

    // custom max
    var arguments = columnSums
    if (arguments.length === 0) {
      return undefined; // Return undefined if no arguments are provided
    }
    let max = -Infinity; // Start with a very low value
    for (let i = 1; i < arguments.length; i++) {
      if (arguments[i] > max) {
        max = arguments[i];
      }
    }
    
    // normalize
    var array_2 = Array(columnSums);
    for(var i = 0, length = columnSums.length; i < length; i++){
        array_2[i] = columnSums[i] / max;
    }

    // find max
    const thresh_indexes = [];
    for (let i = 2; i < array_2.length; i++) {
      if (array_2[i] > this.thresh) {
        thresh_indexes.push(i);
      }
    }
    
    start_time_ms = thresh_indexes[0]*10 - 20;
    // to capture onset in msec
    this.start_time_ms = start_time_ms;

    // const start_time_ms = await this.findMaxFreq(data);
    start_time_ms = this.start_time_ms;
    var start_frame = start_time_ms / 10;
    var the_dat = this.currDat.slice([0, start_frame], [16, 15]);
    var dataTensor = tf.stack([the_dat]);
    var print = false;

    if (print == true){
      for(var i = 0; i < 15; i ++){
        console.log('currDat',i, this.currDat.dataSync().slice(i*16,(i+1)*16));
      }
    }

    if (print == true){
      for(var i = 0; i < 15; i ++){
        console.log('before transformations',i, dataTensor.dataSync().slice(i*16,(i+1)*16));
      }
    }

    // mean and std transformation
    var subbed = tf.sub(dataTensor, mean);
    var dataTensorNormed = tf.div(subbed, std);
    self.dataTensorNormed = dataTensorNormed;
    var dataTensorNormedTransposed = tf.transpose(dataTensorNormed, [0, 2, 1]);

    if (print == true){
      for(var i = 0; i < 15; i ++){
        console.log('right before model',i, dataTensorNormed.dataSync().slice(i*16,(i+1)*16));
      }
    }

    // document.getElementById('debug-dump').innerHTML = dataTensorNormed;
    // console.log(dataTensorNormed.shape);
    
    // gets model prediction
    var y = model.predict(dataTensorNormedTransposed, {batchSize: 1});
    y = y.dataSync()
    var max_y = Math.max.apply(null, y);;
    var min_y = Math.min.apply(null, y);
    var y_scaled = [0, 0, 0];
    for (i=0; i<3; i++){
      y_scaled[i] = y[i] / 4;//(y[i] - min_y) / (max_y - min_y);
    }
    // console.log(y);
    
    // replaces the text in the result tag by the model prediction
    document.getElementById('pred1').style = "height: "+y_scaled[0] * 20 +"vh";
    document.getElementById('pred2').style = "height: "+y_scaled[1] * 20 +"vh";
    document.getElementById('pred3').style = "height: "+y_scaled[2] * 20 +"vh";
    // document.getElementById('pred4').style = "height: "+y_scaled[3] * 100 +"vh";

    localStorage.setItem("currDat", the_dat.arraySync());
    localStorage.setItem("dataTensorNormedArr", dataTensorNormed.arraySync());
    localStorage.setItem("dataTensorNormed", JSON.stringify(dataTensorNormed.arraySync()));
    // console.log('stored');

    const classes = ["b", "d", "g", "null"];
    var predictedClass = tf.argMax(y).array()
    .then(predictedClass => {
      document.getElementById("predClass").innerHTML = classes[predictedClass];
      // if(predictedClass != 3){
      //   console.log('predicted class', predictedClass);
      //   console.log(y.dataSync());
      //   // dataTensorNormed.array().then(array => console.log(array));
      // }
      }
    )
    .catch(err =>
      console.log(err));

    // setTimeout(() => {
    //   document.getElementById('pred1').style = "height: "+1 +"vh";
    //   document.getElementById('pred2').style = "height: "+1 +"vh";
    //   document.getElementById('pred3').style = "height: "+1 +"vh";
    // }, 1000);
  },

  predictModel_noSegment: async function(){
    // converts from a canvas data object to a tensor
    var start_frame = this.custom_start_time_ms / 10;
    var the_dat = this.currDat.slice([0, start_frame], [16, 15]);
    var dataTensor = tf.stack([the_dat]);
    var print = false;

    if (print == true){
      for(var i = 0; i < 15; i ++){
        console.log('currDat',i, this.currDat.dataSync().slice(i*16,(i+1)*16));
      }
    }

    if (print == true){
      for(var i = 0; i < 15; i ++){
        console.log('before transformations',i, dataTensor.dataSync().slice(i*16,(i+1)*16));
      }
    }

    // mean and std transformation
    var subbed = tf.sub(dataTensor, mean);
    var dataTensorNormed = tf.div(subbed, std);
    self.dataTensorNormed = dataTensorNormed;
    var dataTensorNormedTransposed = tf.transpose(dataTensorNormed, [0, 2, 1]);

    // gets model prediction
    var y = model.predict(dataTensorNormedTransposed, {batchSize: 1});
    y = y.dataSync()
    var y_scaled = [0, 0, 0];
    for (i=0; i<3; i++){
      y_scaled[i] = y[i] / 3;//(y[i] - min_y) / (max_y - min_y);
    }
    console.log(y);
    
    // replaces the text in the result tag by the model prediction
    document.getElementById('pred1').style = "height: "+y_scaled[0] * 20 +"vh";
    document.getElementById('pred2').style = "height: "+y_scaled[1] * 20 +"vh";
    document.getElementById('pred3').style = "height: "+y_scaled[2] * 20 +"vh";

    localStorage.setItem("currDat", the_dat.arraySync());
    localStorage.setItem("dataTensorNormedArr", dataTensorNormed.arraySync());
    localStorage.setItem("dataTensorNormed", JSON.stringify(dataTensorNormed.arraySync()));
    // console.log('stored');

    const classes = ["b", "d", "g", "null"];
    var predictedClass = tf.argMax(y).array()
    .then(predictedClass => {
      document.getElementById("predClass").innerHTML = classes[predictedClass];
      // if(predictedClass != 3){
      //   console.log('predicted class', predictedClass);
      //   console.log(y.dataSync());
      //   // dataTensorNormed.array().then(array => console.log(array));
      // }
      }
    )
    .catch(err =>
      console.log(err));

    // setTimeout(() => {
    //   document.getElementById('pred1').style = "height: "+1 +"vh";
    //   document.getElementById('pred2').style = "height: "+1 +"vh";
    //   document.getElementById('pred3').style = "height: "+1 +"vh";
    // }, 1000);
  },

  doneTimer: function() {
    console.log("1s pause");
    this.writing = false;
    this.color = false;
    console.log('after timeout')
    console.log(this.writing, this.color);
    console.log(this.currDat);
    var link = document.createElement('a');
    var data_pre = this.currDat.arraySync();
    console.log('currDat arraysync', currDat.arraySync());
    var str = "";
    for (row in data_pre) {
      str += data_pre[row].toString();
      str += '\n';
    }
    var data = new Blob([str], {type: 'text/plain'});
    console.log(data);
    textFile = window.URL.createObjectURL(data);
    console.log('File written successfully to', textFile);
    link.href = textFile;
    link.download = "data.txt";
    link.click();
    document.getElementById('file-write-btn').innerHTML = "Start File Write";
  },

  createAudioGraph: async function() {
    if (this.audioContext) {
      return;
    }
    // Get input from the microphone.
    this.audioContext = new AudioContext({sampleRate: 22050});
    try {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      this.ctx = this.$.canvas.getContext('2d');
      this.onStream(stream);


      // predict when amplitude is greater than threshold

      // var date2 = Date.now();
      // setInterval(() => {
      //   console.log(this.amplitude_over_thresh);
        // if(this.amplitude_over_thresh){
        //   var data_pre = this.currDat2.arraySync();
        //   this.predictModel(data_pre);
        //   this.color = true;
        //   // tf.print(this.currDat2);
        //   // var date = Date.now();
        //   // console.log(date - date2);
        //   // date2 = date;
        //   // console.log(this.currDat2.shape);
        //   var longest_frames = 100;
        //   if(this.currDat2.shape[1] > longest_frames){
        //     this.currDat2 = tf.slice(this.currDat2, [0, this.currDat2.shape[1] - longest_frames], [16, longest_frames]);
        //   }
        // } else {
        //   this.color = false;
        // }
      // }, 200);
    } catch (e) {
      this.onStreamError(e);
    }
  },

  render: function() {
    var n = Date.now();
    // console.log("time diff:", n - this.now);
    this.now = n;
    this.width = window.innerWidth;
    this.height = window.innerHeight;

    var didResize = false;
    // Ensure dimensions are accurate.
    if (this.$.canvas.width != this.width) {
      this.$.canvas.width = this.width;
      this.$.labels.width = this.width;
      didResize = true;
    }
    if (this.$.canvas.height != this.height) {
      this.$.canvas.height = this.height;
      this.$.labels.height = this.height;
      didResize = true;
    }

    // stop button
    // document.getElementById('mini-spectrogram').onclick = () => {
    //   if (this.audioContext.state == "running"){
    //     console.log('mini clicked, stopping now');
    //     this.audioContext.suspend().then( () => {
    //       this.going = false;
    //     });
    //   } else if (this.audioContext.state == "suspended"){
    //     console.log('mini clicked, starting again');
    //     this.audioContext.resume().then(() => {
    //       this.going = true;
    //     });
    //   }
    // }

    document.getElementById('switch-act-btn').onclick = () => {
      if (this.file_download){
        this.file_download = false;
        document.getElementById('file-write-btn').style = "display: none;";
        document.getElementById('predict-btn').style = "display: block;";
      } else {
        this.file_download = true;
        document.getElementById('file-write-btn').style = "display: block;";
        document.getElementById('predict-btn').style = "display: none;";
      }
    }

    document.getElementById('start-stop-btn').onclick = () => {
      if (this.stopped){
        this.stopped = false;
        document.getElementById('start-stop-btn').innerHTML = "Stop Spectrogram";
      } else {
        this.stopped = true;
        document.getElementById('start-stop-btn').innerHTML = "Start Spectrogram";
        this.predictModel(this.data_whole.arraySync());
        this.custom_start_time_ms = this.start_time_ms;
      }
    }

    document.getElementById('spec-left').onclick = () => {
      console.log('left clicked');
      this.custom_start_time_ms -= 10;
    }

    document.getElementById('spec-right').onclick = () => {
      console.log('right clicked');
      this.custom_start_time_ms += 10;
    }

    document.getElementById('spec-pred').onclick = () => {
      console.log('predicting!!');
      this.predictModel_noSegment();
    }
    // console.log(this.stopped);

    if(this.file_download){
      document.getElementById('file-write-btn').onclick = () => {
        this.currDat = tf.zeros([16, 1], dtype='float32');
        this.writing = true;
        this.color = true;
        document.getElementById('file-write-btn').innerHTML = "Writing...";
        
        setTimeout(() => {
          this.writing = false;
          this.color = false;
          var link = document.createElement('a');
          var data_pre = this.currDat.arraySync();
          var str = "";
          for (row in data_pre) {
            str += data_pre[row].toString();
            str += '\n';
          }
          var data = new Blob([str], {type: 'text/plain'});
          // console.log(data);
          textFile = window.URL.createObjectURL(data);
          console.log('File written successfully to', textFile);
          link.href = textFile;
          link.download = "data.txt";
          link.click();
          document.getElementById('file-write-btn').innerHTML = "Start File Write";
        }, 1000);

      }

      if(this.writing){
        // data
        var currCol = this.extractFrequencies();
        currCol = tf.transpose(tf.tensor([currCol]));
        var currDat = tf.concat([this.currDat, currCol], 1);
        this.currDat = currDat;
        // console.log('376', this.currDat);
        // this.predictModel();
      }
    } else {
      // predicting
      var currCol = this.extractFrequencies();
      currCol = tf.transpose(tf.tensor([currCol]));
      var currDat = tf.concat([this.currDat, currCol], 1);
      this.currDat = currDat;
      if (this.writing == false && this.stopped == false){
        this.frames_since_last_coloured ++;
      } else if (this.writing == true && this.stopped == false){
        var data_whole = tf.concat([this.data_whole, currCol], 1);
        this.data_whole = data_whole;
      }
      // console.log(this.frames_since_last_coloured, this.data_whole.shape[1]);
      document.getElementById('predict-btn').onclick = () => {
        if (this.writing == false){
          this.currDat = tf.zeros([16, 1], dtype='float32');
          this.writing = true;
          this.color = true;
          this.frames_since_last_coloured = 0;
          document.getElementById('predict-btn').innerHTML = "Stop and Predict";
          this.data_whole = tf.zeros([16, 1], dtype='float32');
        } else {
          this.writing = false;
          this.color = false;
          var data_pre = currDat.arraySync();
          this.predictModel(data_pre);
          document.getElementById('file-write-btn').innerHTML = "Record Sample";
        }
      }
    }


    // this.renderTimeDomain();
    if (this.going){
      this.renderFreqDomain();
    }

    if (this.labels && didResize) {
      this.renderAxesLabels();
    }

    setTimeout(() => {
      requestAnimationFrame(this.render.bind(this));
      var currCol = this.extractFrequencies();
      currCol = tf.transpose(tf.tensor([currCol]));
      var currDat = tf.concat([this.currDat, currCol], 1);
      this.currDat = currDat;

      var currDat2 = tf.concat([this.currDat2, currCol], 1);
      // find max
      currCol = currCol.arraySync();
      let amp = 0;
      for (let i = 2; i < currCol.length; i++) {
        if(parseFloat(-currCol[i]) == Infinity){
          amp = 0;
          break;
        }
        amp += parseFloat(currCol[i]);
      }
      // if (amp > -1000){
      //   console.log(amp);
      //   this.prev_max = amp;
      // }
      if (amp > this.amplitude_thresh) {
        this.amplitude_over_thresh = true;
      } else {
        this.amplitude_over_thresh = false;
      }
      
      this.currDat2 = currDat2;
    }, 10);
    
    var now = new Date();
    if (this.lastRenderTime_) {
      this.instantaneousFPS = now - this.lastRenderTime_;
    }
    this.lastRenderTime_ = now;
  },

  renderTimeDomain: function() {
    var times = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteTimeDomainData(times);

    for (var i = 0; i < times.length; i++) {
      var value = times[i];
      var percent = value / 256;
      var barHeight = this.height * percent;
      var offset = this.height - barHeight - 1;
      var barWidth = this.width/times.length;
      this.ctx.fillStyle = 'black';
      this.ctx.fillRect(i * barWidth, offset, 1, 1);
    }
  },

  renderFreqDomain: function() {
    this.analyser.getByteFrequencyData(this.freq);

    // Check if we're getting lots of zeros.
    if (this.freq[0] === 0) {
      //console.warn(`Looks like zeros...`);
    }

    var ctx = this.ctx;
    // Copy the current canvas onto the temp canvas.

    // not stopped case: keep plotting
    if (this.stopped == false){
      this.tempCanvas.width = this.width;
      this.tempCanvas.height = this.height;
      var tempCtx = this.tempCanvas.getContext('2d');
      var tempCtx2 = this.tempCanvas2.getContext('2d');
      tempCtx.drawImage(this.$.canvas, 0, 0, this.width, this.height);
      tempCtx2.drawImage(this.$.canvas, 0, 0, this.width, this.height);

      // Iterate over the frequencies.
      var freq16 = this.extractFrequenciesByte();
      for (var i = 0; i < 16; i++) {
        var value;
        // Draw each pixel with the specific color.
        if (this.log) {
          logIndex = this.logScale(i, 16);
          value = freq16[logIndex];
        } else {
          value = freq16[i];
        }

        // console.log("16 process value: ", value);
        ctx.fillStyle = (this.color ? this.getFullColor(value) : this.getGrayColor(value));

        var percent = i / 16;
        var y = Math.round(percent * this.height);

        // draw the line at the right side of the canvas
        ctx.fillRect(this.width - this.speed, this.height - y,
                    this.speed, this.height / 16);
      }

      // Translate the canvas.
      ctx.translate(-this.speed, 0);
      // Draw the copied image.
      ctx.drawImage(this.tempCanvas, 0, 0, this.width, this.height,
                    0, 0, this.width, this.height);

      // Reset the transformation matrix.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    } else {
      this.tempCanvas2.width = this.width;
      this.tempCanvas2.height = this.height;
      var tempCtx2 = this.tempCanvas2.getContext('2d');
      tempCtx2.drawImage(this.tempCanvas, 0, 0, this.width, this.height);

      // draw start time line
      tempCtx2.fillStyle = 'rgb(0, 0, 255)';
      // console.log(this.width, this.start_time_ms/10, this.speed, this.height);
      var horiz = this.data_whole.shape[1];
      var horiz_shift = (horiz + this.frames_since_last_coloured) * this.speed 
      tempCtx2.fillRect(this.width - horiz_shift, 0, 5, this.height);
      var horiz_shift_start = horiz_shift - this.custom_start_time_ms / 10 * this.speed;
      tempCtx2.fillStyle = 'rgb(0, 255, 255)';
      tempCtx2.fillRect(this.width - horiz_shift_start, 0, 5, this.height);
      
      // console.log(this.start_time_ms, horiz_shift_start);

      // Translate the canvas.
      // ctx.translate(-this.speed, 0);
      // Draw the copied image.
      ctx.drawImage(this.tempCanvas2, 0, 0, this.width, this.height,
                    0, 0, this.width, this.height);

      // Reset the transformation matrix.
      // ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
  },

  /**
   * Given an index and the total number of entries, return the
   * log-scaled value.
   */
  logScale: function(index, total, opt_base) {
    var base = opt_base || 2;
    var logmax = this.logBase(total + 1, base);
    var exp = logmax * index / total;
    return Math.round(Math.pow(base, exp) - 1);
  },

  undoLogScale: function(val, total, opt_base){
    var base = opt_base || 2;
    var exp = this.logBase(val, base);
    var logmax = this.logBase(total + 1, base);
    var index = exp * total / logmax;
    return index;
  },

  logBase: function(val, base) {
    return Math.log(val) / Math.log(base);
  },

  renderAxesLabels: function() {
    if (!this.audioContext) {
      return;
    }
    var canvas = this.$.labels;
    canvas.width = this.width;
    canvas.height = this.height;
    var ctx = canvas.getContext('2d');
    var startFreq = 440;
    var nyquist = this.audioContext.sampleRate/2;
    var endFreq = nyquist - startFreq;
    var step = (endFreq - startFreq) / this.ticks;
    var yLabelOffset = 5;
    // Render the vertical frequency axis.
    for (var i = 0; i <= this.ticks; i++) {
      var freq = startFreq + (step * i);
      // Get the y coordinate from the current label.
      var index = this.freqToIndex(freq);
      var percent = index / this.getFFTBinCount();
      var y = (1-percent) * this.height;
      var x = this.width - 60;
      // Get the value for the current y coordinate.
      var label;
      if (this.log) {
        // Handle a logarithmic scale.
        var logIndex = this.logScale(index, this.getFFTBinCount());
        // Never show 0 Hz.
        freq = Math.max(1, this.indexToFreq(logIndex));
      }
      var label = this.formatFreq(freq);
      var units = this.formatUnits(freq);
      ctx.font = '16px Inconsolata';
      // Draw the value.
      ctx.textAlign = 'right';
      ctx.fillText(label, x, y + yLabelOffset);
      // Draw the units.
      ctx.textAlign = 'left';
      ctx.fillText(units, x + 10, y + yLabelOffset);
      // Draw a tick mark.
      ctx.fillRect(x + 40, y, 30, 2);
    }
  },

  clearAxesLabels: function() {
    var canvas = this.$.labels;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, this.width, this.height);
  },

  formatFreq: function(freq) {
    return (freq >= 1000 ? (freq/1000).toFixed(1) : Math.round(freq));
  },

  formatUnits: function(freq) {
    return (freq >= 1000 ? 'KHz' : 'Hz');
  },

  indexToFreq: function(index) {
    var nyquist = this.audioContext.sampleRate/2;
    return nyquist/this.getFFTBinCount() * index;
  },

  freqToIndex: function(frequency) {
    var nyquist = this.audioContext.sampleRate/2;
    return Math.round(frequency/nyquist * this.getFFTBinCount());
  },

  getFFTBinCount: function() {
    return this.fftsize / 2;
  },

  onStream: function(stream) {
    var input = this.audioContext.createMediaStreamSource(stream);
    var analyser = this.audioContext.createAnalyser();
    analyser.smoothingTimeConstant = 0;
    analyser.fftSize = this.fftsize;

    // Connect graph.
    input.connect(analyser);

    this.analyser = analyser;
    this.freq = new Uint8Array(this.analyser.frequencyBinCount);
    this.freq2 = new Float32Array(this.analyser.frequencyBinCount);

    // Setup a timer to visualize some stuff.
    this.render();
  },

  onStreamError: function(e) {
    console.error(e);
  },

  getGrayColor: function(value) {
    return 'rgb(V, V, V)'.replace(/V/g, 255 - value);
  },

  getFullColor: function(value) {
    var fromH = 62;
    var toH = 0;
    var percent = value / 255;
    var delta = percent * (toH - fromH);
    var hue = fromH + delta;
    return 'hsl(H, 100%, 50%)'.replace(/H/g, hue);
  },
  
  logChanged: function() {
    if (this.labels) {
      this.renderAxesLabels();
    }
  },

  ticksChanged: function() {
    if (this.labels) {
      this.renderAxesLabels();
    }
  },

  labelsChanged: function() {
    if (this.labels) {
      this.renderAxesLabels();
    } else {
      this.clearAxesLabels();
    }
  }
});
