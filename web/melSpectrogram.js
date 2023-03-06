// var wavesurfer;

// Recording.
var mic, recorder, soundFile;
var isRecording = false;

var SR = 44100; // sampling rate

document.getElementById('start-rec').addEventListener('click', () => {
   console.log('start button pressed');
   
   var wavesurfer = WaveSurfer.spectrogram.create({
      wavesurfer: wavesurfer,
      container: "#wave-spectrogram",
      labels: true,
      height: 256,
   });

   wavesurfer.load('../dataset8/b/bear1.wav');
   // setup();
   const colormap = require('colormap');
   const colors = colormap({
       colormap: 'hot',
       nshades: 256,
       format: 'float'
   });
   const fs = require('fs');
   fs.writeFile('hot-colormap.json', JSON.stringify(colors));
});


// function setup(){
//    createCanvas(500, 500);
//    // select('#ws-waveform').mouseClicked(startRecording);
//    document.getElementById('ws-waveform').addEventListener('click', () => {
//       console.log('wv button pressed')
//       startRecording();
//    });

//    // Setup mic
//    mic = new p5.AudioIn();
//    mic.start();
//    // create a sound recorder and set input
//    recorder = new p5.SoundRecorder();
//    recorder.setInput(mic);

//    // Store audio data
//    // soundFile = new p5.SoundFile();
//    // soundFile = loadSound("https://dl.dropbox.com/s/rhgr1e6201xyh9q/Akai%20XR10-XR10Kick08.wav?raw=1", onLoaded);
// }

// function onLoaded(){
//    processAudio();
// }

// function processAudio(){
//    // show waveform on wavesurfer component
//    wavesurfer.loadDecodedBuffer(soundFile.buffer);

//    // Spectrogram
//    fill(255);
//    text('Spectrogram', 10, 30);
//    spectrogram = createSpectrogram(soundFile.buffer);
//    console.log("SPECTROGRAM", spectrogram);
//    image(spectrogram, 10, 40, 600, 200);

//    // Mel Spectrogram
//    text('Mel Spectrogram', 10, 270);
//    melspectrogram = createMelSpectrogram(soundFile.buffer);
//    image(melspectrogram, 10, 280, 600, 200);
// }


// function draw(){
//    // react to mic input volume
//    if (mic.enabled){
//       var level = mic.getLevel();  
//       var waveform = select("#ws-waveform");
//       waveform.style('background', 
//                      'rgb(' + int(level* 255) +', 0, 0)');
//    }
// }

// function startRecording(){
//    if (mic.enabled === false) {
//       select("#ws-waveform-text").html('Allow mic input please!');
//       return;
//    }

//    if (!isRecording){   
//       // Start recording...
//       recorder.record(soundFile, 2.0, onRecStop);

//       select("#ws-waveform-text").html('Recording...');
//       var waveform = select("#ws-waveform");
//       waveform.style('border-color', 'red');
//    } 
// }

// function onRecStop(){
//    var waveform = select("#ws-waveform");
//    waveform.style('border-color', 'white');
//    select("#ws-waveform-text").html('');　// done!
//    isRecording = false;

//    processAudio();
// }


// function createSpectrogram(buffer){
//    const fftSize = 1024;                        // fft window size
//    const nOverlap = fftSize / 2;                // overlap size
//    const channelOne = buffer.getChannelData(0);  // use only the first channel
//    const bufferLength = buffer.length;
//    const sampleRate = buffer.sampleRate;
//    const spectrogram = [];

//    // Create a fft object. Here we use default "Hanning" window function
//    const fft = new FFT(fftSize, sampleRate); 
   
//    // Segment 
//    let currentOffset = 0;
//    while (currentOffset + fftSize < channelOne.length) {
//       const segment = channelOne.slice(currentOffset, currentOffset + fftSize); 
//       fft.forward(segment);  // generate spectrum for this segment
//       let spectrum = fft.spectrum;
      
//       const array = new Uint8Array(fftSize / 2); // 0 - 254
//       for (let j = 0; j < fftSize / 2; j++) { // use the first half of the fft result <- Nyquist theorem
//          array[j] = Math.max(-255, Math.log10(spectrum[j]) * 45);
//       }
//       spectrogram.push(array);
//       currentOffset += fftSize - nOverlap;
//    }
   
//    // Create P5 Image
//    var specW = spectrogram.length;
//    var specH = fftSize / 2;
//    var img = createImage(specW, specH);
//    img.loadPixels();
//    for (var i = 0; i < img.width; i++) {
//       for (var j = 0; j < img.height; j++) {
//          img.set(i, j, color(spectrogram[i][img.height - j - 1]));
//       }
//    }
//    img.updatePixels();
//    return img;
// }

// function createMelSpectrogram(buffer){
//    const fftSize = 1024;                        // fft window size
//    const nOverlap = fftSize / 2;                // overlap size
//    const channelOne = buffer.getChannelData(0);  // use only the first channel
//    const bufferLength = buffer.length;
//    const sampleRate = buffer.sampleRate;
//    const spectrogram = [];

//    const melCount = 129;

//    // Create a fft object. Here we use default "Hanning" window function
//    const fft = new FFT(fftSize, sampleRate); 

//    // Mel Filterbanks
//    var melFilterbanks = createMelFilterbank(fftSize/2, melCount, 
//                                             lowHz=50, highHz=20000, sr=sampleRate); 

//    // Segment 
//    let currentOffset = 0;
//    let maxValue = 0.0;
//    while (currentOffset + fftSize < channelOne.length) {
//       const segment = channelOne.slice(currentOffset, currentOffset + fftSize); 
//       fft.forward(segment);  // generate spectrum for this segment
//       let spectrum = fft.spectrum;

//       const melspec = applyFilterbank(spectrum, melFilterbanks);

//       const array = new Uint8Array(melCount); // 0 - 254
//       for (let j = 0; j < melCount; j++) {
//          array[j] = Math.max(-255, Math.log10(melspec[j]) * 45);         
//       }

//       spectrogram.push(array);
//       currentOffset += fftSize - nOverlap;
//    }

//    // Create P5 Image
//    var specW = spectrogram.length;
//    var specH = melCount;
//    var img = createImage(specW, specH);
//    img.loadPixels();
//    for (var i = 0; i < img.width; i++) {
//       for (var j = 0; j < img.height; j++) {
//          img.set(i, j, color(spectrogram[i][img.height - j - 1]));
//       }
//    }
//    img.updatePixels();
//    return img;
// }


// /////////////////////

// function sum(array) {
//    return array.reduce(function(a, b) { return a + b; });
// }

// // // Use a lower minimum value for energy.
// // const MIN_VAL = -10;
// // function logGtZero(val) {
// //    // Ensure that the log argument is nonnegative.
// //    const offset = Math.exp(MIN_VAL);
// //    return Math.log(val + offset);
// // }

// function applyFilterbank(fftEnergies, filterbank){
//    if (fftEnergies.length != filterbank[0].length) {
//       console.error(`Each entry in filterbank should have dimensions matching
// FFT. |FFT| = ${fftEnergies.length}, |filterbank[0]| = ${filterbank[0].length}.`);
//       return;
//    }

//    // Apply each filter to the whole FFT signal to get one value.
//    let out = new Float32Array(filterbank.length);
//    for (let i = 0; i < filterbank.length; i++) {
//       // To calculate filterbank energies we multiply each filterbank with the
//       // power spectrum.
//       const win = applyWindow(fftEnergies, filterbank[i]);
//       // Then add up the coefficents, and average
//       out[i] = sum(win) / fftEnergies.length;
//       // out[i] = logGtZero(sum(win));
//    }
//    return out;
// }

// function  applyWindow(buffer, win) {
//    if (buffer.length != win.length) {
//       console.error(`Buffer length ${buffer.length} != window length
// ${win.length}.`);
//       return;
//    }

//    let out = new Float32Array(buffer.length);
//    for (let i = 0; i < buffer.length; i++) {
//       out[i] = win[i] * buffer[i];
//    }
//    return out;
// }

// function hzToMel(hz) {
//    return 1125 * Math.log(1 + hz/700);
// }

// function melToHz(mel) {
//    return 700 * (Math.exp(mel/1125) - 1);
// }

// function freqToBin(freq, fftSize, sr=SR) {
//    return Math.floor((fftSize+1) * freq / (sr/2));
// }

// function linearSpace(start, end, count) {
//    const delta = (end - start) / (count + 1);
//    let out = [];
//    for (let i = 0; i < count; i++) {
//       out[i] = start + delta * i;
//    }
//    return out;
// }

// function triangleWindow(length, startIndex, peakIndex, endIndex) {
//    const win = new Float32Array(length);
//    const deltaUp = 1.0 / (peakIndex - startIndex);
//    for (let i = startIndex; i < peakIndex; i++) {
//       // Linear ramp up between start and peak index (values from 0 to 1).
//       win[i] = (i - startIndex) * deltaUp;
//    }
//    const deltaDown = 1.0 / (endIndex - peakIndex);
//    for (let i = peakIndex; i < endIndex; i++) {
//       // Linear ramp down between peak and end index (values from 1 to 0).
//       win[i] = 1 - (i - peakIndex) * deltaDown;
//    }
//    return win;
// }

// function createMelFilterbank(fftSize, melCount=128, lowHz=50, highHz=8000, sr=SR) {
//    const lowMel = hzToMel(lowHz);
//    const highMel = hzToMel(highHz);

//    // Construct linearly spaced array of melCount intervals, between lowMel and
//    // highMel.
//    const mels = linearSpace(lowMel, highMel, melCount + 2);
//    // Convert from mels to hz.
//    const hzs = mels.map(mel => melToHz(mel));
//    // Go from hz to the corresponding bin in the FFT.
//    const bins = hzs.map(hz => freqToBin(hz, fftSize, sr));

//    // Now that we have the start and end frequencies, create each triangular
//    // window (each value in [0, 1]) that we will apply to an FFT later. These
//    // are mostly sparse, except for the values of the triangle
//    const length = bins.length - 2;
//    const filters = [];
//    for (let i = 0; i < length; i++) {
//       // Now generate the triangles themselves.
//       filters[i] = triangleWindow(fftSize, bins[i], bins[i+1], bins[i+2]);
//    }

//    return filters;
// }
