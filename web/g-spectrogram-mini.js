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
  file_download: false,
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

    self.mean = tf.tensor([
      [-9.921779295944855903e+01, -9.504090236890817778e+01, -8.973951395087387084e+01, -8.813558030370704444e+01, -8.830461164898035520e+01, -8.856550378896328368e+01, -8.888497280347849028e+01, -8.919695596459642672e+01, -8.920588392858408611e+01, -8.911086566631610140e+01, -8.906545378265602153e+01, -8.904514218622436772e+01, -8.914320796529163715e+01, -8.955490017075669584e+01, -8.991511852336380173e+01],
      [-9.075273659405756632e+01, -8.522021021517683437e+01, -7.967333662146580764e+01, -7.790902755360818333e+01, -7.802522402404873958e+01, -7.837438814249024688e+01, -7.870837898420491285e+01, -7.894293338570781771e+01, -7.885325426563987605e+01, -7.910064233854650695e+01, -7.926168331594702465e+01, -7.943693163225370313e+01, -7.991612965621173714e+01, -8.026557460613278749e+01, -8.086238828418217395e+01],
      [-8.412980505689999688e+01, -6.861726509503944271e+01, -6.160314973518360659e+01, -6.338417906408213298e+01, -6.599926225194391805e+01, -6.798213181959354756e+01, -6.950867571228647535e+01, -7.063717526240688471e+01, -7.091154763452892951e+01, -7.082058905724690590e+01, -7.097939856408815729e+01, -7.131390446744566702e+01, -7.197601356561713715e+01, -7.263725258615436076e+01, -7.344943545691680242e+01],
      [-8.202547463138840556e+01, -6.726231416943110730e+01, -5.978254379139583108e+01, -6.155418878231408542e+01, -6.529390954625277743e+01, -6.734430237812989617e+01, -6.786520803753280973e+01, -6.780727952312142293e+01, -6.755405051753206180e+01, -6.708668832543627047e+01, -6.696780552746422188e+01, -6.702466487469278889e+01, -6.764758838935929930e+01, -6.841865509653298716e+01, -6.933185566873093819e+01],
      [-8.036808947381848611e+01, -6.402168821492631423e+01, -5.607803683564694097e+01, -5.880026675308737083e+01, -6.428195743533107986e+01, -6.832104446233962847e+01, -7.112860491659887430e+01, -7.360626133778963265e+01, -7.552907670218989722e+01, -7.663933951899690555e+01, -7.728362021148637950e+01, -7.783980912982292466e+01, -7.859137361523721665e+01, -7.903635458537905834e+01, -7.949150355633872778e+01],
      [-8.586243808009973577e+01, -7.212140651088321874e+01, -6.461985399553495313e+01, -6.482037816926599305e+01, -6.638747799102384306e+01, -6.699520312615160833e+01, -6.709673100597453299e+01, -6.718971215346381598e+01, -6.749734921171634028e+01, -6.796173379729205521e+01, -6.862869039571509688e+01, -6.960827156698063334e+01, -7.095992656093204687e+01, -7.243919977705779445e+01, -7.385276274549597986e+01],
      [-9.252242135655551181e+01, -8.193151522720846458e+01, -7.508731863079984237e+01, -7.333493310515869723e+01, -7.292675932398381633e+01, -7.279851943903287292e+01, -7.268202527825475556e+01, -7.286110524480677952e+01, -7.331530379141709375e+01, -7.410264474456299411e+01, -7.514087227986063056e+01, -7.653645959198041737e+01, -7.808035902291838681e+01, -7.972187457479135730e+01, -8.124538866771149515e+01],
      [-9.631140203157599444e+01, -8.653365892318925034e+01, -8.058126815552289202e+01, -7.914875455172557395e+01, -7.901523831482376181e+01, -7.861658513770913714e+01, -7.815832889374178194e+01, -7.778380431353785696e+01, -7.776296050316709341e+01, -7.808189126888103715e+01, -7.881516162888924271e+01, -7.998609873933612846e+01, -8.121702075177594793e+01, -8.286320603014941355e+01, -8.452704968611629965e+01],
      [-9.898570150699255521e+01, -8.902490052775478091e+01, -8.325490185276690625e+01, -8.206026553415594549e+01, -8.217173439878510521e+01, -8.206777378851859339e+01, -8.179553940749480034e+01, -8.158620012723483228e+01, -8.159113950687846284e+01, -8.194551992762417569e+01, -8.271591833194904098e+01, -8.379262108657459862e+01, -8.527125165078393820e+01, -8.693087593045393646e+01, -8.870130569391568542e+01],
      [-9.731649517286325590e+01, -8.746131300130326736e+01, -8.236929510639782848e+01, -8.149582732467762014e+01, -8.243119718443672639e+01, -8.343588395250206702e+01, -8.419934852721901564e+01, -8.490606780806542986e+01, -8.560293743537714306e+01, -8.626958993765023820e+01, -8.705041088663442395e+01, -8.810765681938093508e+01, -8.930487206158686320e+01, -9.074024863166975763e+01, -9.222308047312610313e+01],
      [-9.826397040689631979e+01, -8.894941416326558681e+01, -8.401177968549797015e+01, -8.338197573591213541e+01, -8.433952176518987187e+01, -8.555004024644028959e+01, -8.656682861283832153e+01, -8.752681908240684550e+01, -8.855826101316941390e+01, -8.962059532575233334e+01, -9.101209611159104895e+01, -9.245796829245779236e+01, -9.387190102077533993e+01, -9.542288815923284062e+01, -9.662643739896866180e+01],
      [-9.758806770114317430e+01, -8.834228202808751007e+01, -8.355351584535551979e+01, -8.284117575480733819e+01, -8.438225965679816909e+01, -8.553290120472236424e+01, -8.683410727579811805e+01, -8.788557754649478682e+01, -8.858780119830881006e+01, -8.931116247523159757e+01, -9.036820358807878506e+01, -9.214134288837671249e+01, -9.370903188985040799e+01, -9.552278440057452258e+01, -9.680801415962473300e+01],
      [-9.765616042935796770e+01, -8.873693786753970869e+01, -8.364011382502982883e+01, -8.329731536222992361e+01, -8.454096060809618507e+01, -8.573168055585576042e+01, -8.697410164931342536e+01, -8.778545989215113821e+01, -8.855834913322991042e+01, -8.963302993220756321e+01, -9.094887374412857639e+01, -9.237594458880376180e+01, -9.429281322772686735e+01, -9.624319595868425381e+01, -9.695342280660554479e+01],
      [-9.815158038423092535e+01, -8.933801628854693888e+01, -8.410716047688389096e+01, -8.340456175561912744e+01, -8.445557337679261423e+01, -8.543481285305604445e+01, -8.658629289801477569e+01, -8.749824951349044966e+01, -8.807287346849594201e+01, -8.953565821765296562e+01, -9.124960318289923578e+01, -9.246335969988597014e+01, -9.420965479835543022e+01, -9.559263646827554339e+01, -9.662376084348460381e+01],
      [-9.828051817053807326e+01, -8.904106305680184619e+01, -8.426504379294607361e+01, -8.347032146703003264e+01, -8.486687533152985452e+01, -8.609704697806880347e+01, -8.698791094200012708e+01, -8.745210689660945036e+01, -8.841204860559915346e+01, -8.972669369458114375e+01, -9.148092522918744862e+01, -9.288640740647890937e+01, -9.457704844149519374e+01, -9.601775870579247396e+01, -9.699619168640047917e+01],
      [-9.812028448052261353e+01, -8.886189397360659825e+01, -8.393420566841203367e+01, -8.361577577964661145e+01, -8.467248749975196631e+01, -8.585780364370138784e+01, -8.655090891778556283e+01, -8.710252166659461182e+01, -8.824202473589228646e+01, -8.932932631959070591e+01, -9.117467536427977848e+01, -9.287102187423816702e+01, -9.428779325775899167e+01, -9.571418749472918819e+01, -9.688136771524592916e+01]
      ]);
    self.std = tf.tensor([
      [7.286668624295761454e+00, 6.188322730874405764e+00, 5.261313414160342816e+00, 4.813804650930848084e+00, 4.846254177390325601e+00, 4.928120517923108679e+00, 5.110628872465769135e+00, 4.972703492092898081e+00, 5.104121176992391540e+00, 5.047996496430074309e+00, 5.049501916797716206e+00, 4.972665288182120058e+00, 5.137493630015012691e+00, 5.247234809997560312e+00, 5.731357695858214640e+00],
      [6.222726552219799423e+00, 6.355200195538980523e+00, 5.592861929227797901e+00, 4.965898420667410385e+00, 4.593947561127387225e+00, 4.618380428753004807e+00, 4.559983249040277187e+00, 4.647741335784619565e+00, 4.734766900657366051e+00, 4.709230721526005858e+00, 4.746862685045726060e+00, 4.817993885350151828e+00, 4.895244911128163956e+00, 5.104074127275255890e+00, 5.425696186536110410e+00],
      [1.149276954358194303e+01, 6.950008622509002087e+00, 3.789078716475736019e+00, 3.822899114287794831e+00, 3.825160927584695170e+00, 3.925480714904737489e+00, 4.062455440517886096e+00, 4.033021711769053219e+00, 4.303809246110916753e+00, 4.725029214791585552e+00, 4.976262344758877632e+00, 5.399389045252785735e+00, 5.616748615195561456e+00, 6.092293093496623513e+00, 6.674583280583751943e+00],
      [1.322292410839044585e+01, 7.493369872956865407e+00, 5.069112290793458264e+00, 5.375087603161272654e+00, 5.068803979686340000e+00, 4.845116805284191308e+00, 4.965912247567180415e+00, 5.227710146093220267e+00, 5.808021060382837497e+00, 6.145419799341554246e+00, 6.178541241400212769e+00, 6.208369330504670991e+00, 6.411527603499252770e+00, 6.865318843728352327e+00, 7.238730581419275723e+00],
      [1.399008021310209138e+01, 8.412427176272634810e+00, 6.316221081138832183e+00, 6.901079025981870174e+00, 6.671288150424958374e+00, 6.338879584539180101e+00, 6.310612002899705253e+00, 6.158847140121669561e+00, 5.936086000325309087e+00, 5.974853076961812448e+00, 6.191452165526021290e+00, 6.302446620380608877e+00, 6.713104888818222094e+00, 7.064212864325383201e+00, 7.266301046848555600e+00],
      [1.296535757482220497e+01, 1.015793838554453998e+01, 1.012086172905775427e+01, 9.904754207193757765e+00, 9.474583963547313914e+00, 9.503765775264145788e+00, 9.538079755101183110e+00, 9.682263797336805311e+00, 9.673029888569622869e+00, 9.720033605957080880e+00, 9.481597138503030209e+00, 9.252176689582645608e+00, 9.341815542032877673e+00, 9.492645775113203399e+00, 9.657698964216805848e+00],
      [1.163938877283649376e+01, 9.464643295204977491e+00, 1.006711246545269134e+01, 1.044292278560415355e+01, 1.033453286019951634e+01, 1.029746898153433854e+01, 1.024878478089102174e+01, 1.011987141385173672e+01, 9.998504271606449834e+00, 9.974580314522679458e+00, 9.821220613669742860e+00, 9.975826079756643594e+00, 1.022619662886315339e+01, 1.068024265113605686e+01, 1.105146386100367728e+01],
      [1.198646429637759958e+01, 9.961727265084961758e+00, 1.058679147449905678e+01, 1.071968932131252217e+01, 1.048717428816468455e+01, 1.030385517460335976e+01, 1.028099643148885356e+01, 1.034847494820489189e+01, 1.039201385587929671e+01, 1.061475104074391851e+01, 1.071362084354512589e+01, 1.099939899780413022e+01, 1.144249443389244014e+01, 1.172876333131877757e+01, 1.212402573229997849e+01],
      [1.320144205301844309e+01, 1.062429762279809076e+01, 1.011432786428976627e+01, 1.011316636261457980e+01, 1.000830360630071425e+01, 1.016318481843081045e+01, 1.054280680053829045e+01, 1.077583728416845155e+01, 1.102297115676767447e+01, 1.098475840239842327e+01, 1.070187625607728776e+01, 1.068059993922534900e+01, 1.079036248398540820e+01, 1.119742616111521194e+01, 1.152225840218803476e+01],
      [1.357671492964433391e+01, 9.923407894323696965e+00, 9.110633108982900907e+00, 9.675232948553105672e+00, 1.014480103754114637e+01, 1.030755228571678650e+01, 1.015674997291133153e+01, 9.955398041479911697e+00, 9.858222650289091504e+00, 9.865516606920182952e+00, 1.018910249849788663e+01, 1.075440217074224414e+01, 1.135169024805441929e+01, 1.211311021263566801e+01, 1.257115260221323538e+01],
      [1.420218166208538513e+01, 1.045310557144564889e+01, 9.254446194736109632e+00, 9.507813199441947916e+00, 9.547684205552284809e+00, 9.268452961009465696e+00, 9.058048852136838747e+00, 8.925607953894054702e+00, 8.817929853875188684e+00, 8.813773960142384567e+00, 9.566979700998496483e+00, 1.043922529607286442e+01, 1.115633967972823903e+01, 1.215055452075874420e+01, 1.253071518904788562e+01],
      [1.542721471051009097e+01, 1.222315519311286636e+01, 1.182985658445341670e+01, 1.261286586019639877e+01, 1.287341006487998740e+01, 1.182640100595713051e+01, 1.169678711126175052e+01, 1.187794200826878743e+01, 1.170951803983771633e+01, 1.157661892753010235e+01, 1.217588985729190298e+01, 1.277056879468613282e+01, 1.298778619037931925e+01, 1.403150206344887785e+01, 1.430156703513895877e+01],
      [1.520177876315661081e+01, 1.255396516768548842e+01, 1.148898759698637662e+01, 1.245639595738716388e+01, 1.283571415368146695e+01, 1.178837519712128135e+01, 1.153503128140190626e+01, 1.141098226556180961e+01, 1.169609743228173748e+01, 1.195600635491226882e+01, 1.246845854051796998e+01, 1.265944586240405023e+01, 1.294696019483553684e+01, 1.405022313104263532e+01, 1.398377439219162000e+01],
      [1.548937150996751377e+01, 1.271517357591173969e+01, 1.139073171822060360e+01, 1.209676925353820565e+01, 1.252277569997614037e+01, 1.181903891669183615e+01, 1.200069087394632916e+01, 1.197172569472403403e+01, 1.154204084715996004e+01, 1.190555285081449632e+01, 1.244053481983413434e+01, 1.250465220922589715e+01, 1.273835247313514429e+01, 1.345108167825863887e+01, 1.377640273085810740e+01],
      [1.560750248845418930e+01, 1.182129760985778866e+01, 1.114520254304939151e+01, 1.188283245879660299e+01, 1.234741832871065448e+01, 1.240363140302228118e+01, 1.246344646967970249e+01, 1.186704556855528558e+01, 1.182471179039792730e+01, 1.197523078119053608e+01, 1.229366514916924125e+01, 1.255157673146142372e+01, 1.289233096505268783e+01, 1.349687490194341954e+01, 1.378171974757912110e+01],
      [1.550477521004400749e+01, 1.194260282173696375e+01, 1.120092687631478690e+01, 1.211157118062989468e+01, 1.239156596716742698e+01, 1.274711324083283159e+01, 1.255944679643404349e+01, 1.230489997620647635e+01, 1.233601799052070724e+01, 1.187370991828170297e+01, 1.201355521816218008e+01, 1.282539098259325883e+01, 1.316209747244972483e+01, 1.378991698968482638e+01, 1.394202175638105423e+01]
      ]);

    console.log('loaded mean and std');
    // tf.print(mean)
    // tf.print(std)

    self.model = await tf.loadLayersModel('tfjs_model/model.json');

    console.log('loaded model')
    // console.log(model);
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
    var start_time_ms = -1;
    if(this.custom_start_time_ms == -1){
      start_time_ms = this.start_time_ms;
    } else {
      start_time_ms = this.custom_start_time_ms;
    }
    localStorage.setItem("currDat", this.currDat.arraySync());
    localStorage.setItem("dataWhole", this.data_whole.arraySync());
    console.log('dataWhole shape', this.data_whole.shape);
    localStorage.setItem("dataTensorNormedArr", dataTensorNormed.arraySync());
    localStorage.setItem("dataTensorNormed", JSON.stringify(dataTensorNormed.arraySync()));
    localStorage.setItem("starttime", start_time_ms);
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
    var subbed = tf.sub(dataTensor, self.mean);
    var dataTensorNormed = tf.div(subbed, self.std);
    // self.dataTensorNormed = dataTensorNormed;
    var dataTensorNormedTransposed = tf.transpose(dataTensorNormed, [0, 2, 1]);

    // document.getElementById('debug-dump').innerHTML = dataTensorNormed;
    // console.log(dataTensorNormed.shape);
    
    // // gets model prediction
    // console.log('model read from g-spectrogram-mini', model)
    var y = model.predict(dataTensorNormedTransposed, {batchSize: 1});
    
    // var y = self.model.predict(dataTensorNormedTransposed);
    y = y.dataSync()
    console.log(y);
    var max_y = Math.max.apply(null, y);
    var min_y = Math.min.apply(null, y);
    var y_scaled = [0, 0, 0];
    for (i=0; i<3; i++){
      y_scaled[i] = (y[i] - min_y) / (max_y - min_y);
    }
    
    // replaces the text in the result tag by the model prediction
    document.getElementById('pred1').style = "height: "+y_scaled[0] * 30 +"vh";
    document.getElementById('pred2').style = "height: "+y_scaled[1] * 30 +"vh";
    document.getElementById('pred3').style = "height: "+y_scaled[2] * 30 +"vh";
    document.getElementById('pred1_text').innerHTML = y[0].toLocaleString(
      undefined, { minimumFractionDigits: 2 , maximumFractionDigits :2});
      document.getElementById('pred2_text').innerHTML = y[1].toLocaleString(
        undefined, { minimumFractionDigits: 2 , maximumFractionDigits :2});
        document.getElementById('pred3_text').innerHTML = y[2].toLocaleString(
          undefined, { minimumFractionDigits: 2 , maximumFractionDigits :2});

    // localStorage.setItem("currDat", the_dat.arraySync());
    // localStorage.setItem("dataTensorNormedArr", dataTensorNormed.arraySync());
    // localStorage.setItem("dataTensorNormed", JSON.stringify(dataTensorNormed.arraySync()));
    // console.log('stored');

    const classes = ["b", "d", "g", "null"];
    var predictedClass = tf.argMax(y).array()
    .then(predictedClass => {
      document.getElementById("predClass").innerHTML = classes[predictedClass];
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

    // start_frame = 34;
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

    // tf.print(dataTensor);

    // mean and std transformation

    var subbed = tf.sub(dataTensor, self.mean);
    var dataTensorNormed = tf.div(subbed, self.std);
    self.dataTensorNormed = dataTensorNormed;
    var dataTensorNormedTransposed = tf.transpose(dataTensorNormed, [0, 2, 1]);

    // gets model prediction
    var y = self.model.predict(dataTensorNormedTransposed);
    y = y.dataSync()
    var y_scaled = [0, 0, 0];
    var max_y = Math.max.apply(null, y);
    var min_y = Math.min.apply(null, y);
    for (i=0; i<3; i++){
      y_scaled[i] = (y[i] - min_y) / (max_y - min_y);
    }
    // console.log(y_scaled);
    
    // replaces the text in the result tag by the model prediction
    document.getElementById('pred1').style = "height: "+y_scaled[0] * 30 +"vh";
    document.getElementById('pred2').style = "height: "+y_scaled[1] * 30 +"vh";
    document.getElementById('pred3').style = "height: "+y_scaled[2] * 30 +"vh";
    document.getElementById('pred1_text').innerHTML = y[0].toLocaleString(
      undefined, { minimumFractionDigits: 2 , maximumFractionDigits :2});
      document.getElementById('pred2_text').innerHTML = y[1].toLocaleString(
        undefined, { minimumFractionDigits: 2 , maximumFractionDigits :2});
        document.getElementById('pred3_text').innerHTML = y[2].toLocaleString(
          undefined, { minimumFractionDigits: 2 , maximumFractionDigits :2});

    localStorage.setItem("currDat", the_dat.arraySync());
    // console.log('currDat dimensions', the_dat.shape);
    localStorage.setItem("dataTensorNormedArr", dataTensorNormed.arraySync());
    localStorage.setItem("dataTensorNormed", JSON.stringify(dataTensorNormed.arraySync()));
    // console.log('stored');

    const classes = ["b", "d", "g", "null"];
    var predictedClass = tf.argMax(y).array()
    .then(predictedClass => {
      document.getElementById("predClass").innerHTML = classes[predictedClass];
      }
    )
    .catch(err =>
      console.log(err));
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
        document.getElementById('start-stop-btn').innerHTML = "Pause";
      } else {
        this.stopped = true;
        document.getElementById('start-stop-btn').innerHTML = "Resume";
        // console.log(data_whole.shape);
        // data_whole shape: 16 times length
        this.predictModel(this.data_whole.arraySync());
        this.custom_start_time_ms = this.start_time_ms;

        // console.log('the current selection')
        // tf.print(this.data_whole.arraySync());
      }
    }

    document.getElementById('store-in-browser').onclick = () => {
      console.log('storing data');
      this.storeData();
      // console.log(localStorage.getItem("dataTensorNormed"));
      console.log(localStorage.getItem("starttime"));
    }

    document.getElementById('spec-left').onclick = () => {
      console.log('left clicked');
      this.custom_start_time_ms -= 10;
      this.predictModel_noSegment();
    }

    document.getElementById('spec-right').onclick = () => {
      console.log('right clicked');
      this.custom_start_time_ms += 10;
      this.predictModel_noSegment();
    }

    document.getElementById('spec-pred').onclick = () => {
      console.log('predicting!!');
      this.predictModel_noSegment();
    }

    document.getElementById('download').onclick = () => {
      console.log('downloading selected segment');
      this.currDat = tf.zeros([16, 1], dtype='float32');
      var link = document.createElement('a');
      var data_pre = this.data_whole.arraySync();
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
      file_name = this.custom_start_time_ms.toString() + "data.txt"
      link.download = file_name;
      link.click();
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
          // var data_pre = currDat.arraySync();
          var data_pre = data_whole.arraySync();
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
    // this.analyser.getByteFrequencyData(this.freq);
    this.analyser.getFloatFrequencyData(this.freq2);

    // Check if we're getting lots of zeros.
    if (this.freq[0] === 0) {
      //console.warn(`Looks like zeros...`);
    }

    // console.log('bytefrequency data', this.freq)

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
        var y = Math.round(percent * this.height + 80);

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

      var horiz_shift_start1 = horiz_shift - (this.custom_start_time_ms / 10 + 15) * this.speed;
      tempCtx2.fillRect(this.width - horiz_shift_start1, 0, 5, this.height);
      
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
