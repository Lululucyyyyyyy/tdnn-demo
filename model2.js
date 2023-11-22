// model_param 15

const DEBUG = true; 
/** ===================================================== 
 *            PARAMETERS  
* ===================================================== 
*/


var tdnn1Weight = [[1.812771260738372803e-01,3.991544246673583984e-01,1.410541683435440063e-01,-4.683352112770080566e-01,1.933487355709075928e-01,1.079184114933013916e-01,7.376736402511596680e-01,1.495528072118759155e-01,-4.889904856681823730e-01,-7.367792725563049316e-02,-3.485316038131713867e-01,-6.977038383483886719e-01,-3.102774918079376221e-01,8.021134138107299805e-02,7.661584019660949707e-02,5.246437788009643555e-01,4.072675406932830811e-01,2.798868715763092041e-01,1.697847992181777954e-01,3.858925104141235352e-01,7.181980013847351074e-01,-3.204737901687622070e-01,-2.732799351215362549e-01,-3.341503143310546875e-01,-1.334160089492797852e+00,-6.850094199180603027e-01,6.693439930677413940e-02,-1.938715577125549316e-01,6.928023099899291992e-01,1.033642172813415527e+00,1.757519841194152832e-01,1.048007011413574219e-01,-4.013004302978515625e-01,-4.025991559028625488e-01,-7.964355945587158203e-01,1.960715465247631073e-02,-1.504157204180955887e-02,2.115354835987091064e-01,2.098854333162307739e-01,6.732114553451538086e-01,2.102609425783157349e-01,-3.749601840972900391e-01,-1.757828146219253540e-01,-3.944992125034332275e-01,-3.310253620147705078e-01,1.252710521221160889e-01,3.298304080963134766e-01,3.485022485256195068e-01], 
[-9.417685214430093765e-05,5.626087635755538940e-02,-4.186416417360305786e-02,7.414809465408325195e-01,-5.540154576301574707e-01,-9.887300431728363037e-02,-5.506240129470825195e-01,-4.347817599773406982e-02,3.756023943424224854e-01,-3.437191247940063477e-01,-1.864396035671234131e-01,-1.967630237340927124e-01,-3.559022545814514160e-01,-3.058075904846191406e-01,-5.453938841819763184e-01,-7.065693140029907227e-01,-3.984584510326385498e-01,-6.116695404052734375e-01,4.586413502693176270e-01,-3.383498638868331909e-03,-5.157477259635925293e-01,1.383599936962127686e-01,-6.295224428176879883e-01,-5.884341597557067871e-01,1.793388247489929199e+00,7.970205545425415039e-01,-3.175047338008880615e-01,6.849387288093566895e-01,-3.626346290111541748e-01,-1.085987329483032227e+00,-1.477793883532285690e-02,-3.843515515327453613e-01,-3.904053568840026855e-01,2.991178631782531738e-01,8.998126983642578125e-01,-1.549401581287384033e-01,6.155755519866943359e-01,2.535133361816406250e-01,-1.361194998025894165e-01,-4.723791480064392090e-01,1.536693274974822998e-01,4.694431647658348083e-02,-1.615124754607677460e-02,-3.476702570915222168e-01,1.956289559602737427e-01,-1.582563519477844238e-01,-3.200124502182006836e-01,2.107347100973129272e-01], 
[-1.364187151193618774e-01,-8.790610730648040771e-02,1.783297508955001831e-01,-4.488100111484527588e-02,6.629506498575210571e-02,3.617174327373504639e-01,-5.145129561424255371e-01,3.014904856681823730e-01,-8.007989078760147095e-02,-2.301797866821289062e-01,-1.637241542339324951e-01,-2.129669487476348877e-01,-2.859885990619659424e-01,-3.593474030494689941e-01,-5.744909048080444336e-01,-6.159896850585937500e-01,-9.689952135086059570e-01,-6.925728321075439453e-01,1.572076380252838135e-01,-2.177841514348983765e-01,-3.072472214698791504e-01,3.476523458957672119e-01,3.763901814818382263e-02,7.580404728651046753e-02,8.109049201011657715e-01,4.394566714763641357e-01,3.198165595531463623e-01,3.315130174160003662e-01,-2.760989367961883545e-01,-1.519863456487655640e-01,8.189451098442077637e-01,8.738092184066772461e-01,9.193699955940246582e-01,2.454329431056976318e-01,6.122266873717308044e-02,-5.278438925743103027e-01,1.858159601688385010e-01,-2.624011337757110596e-01,-1.196144297719001770e-01,1.999155282974243164e-01,1.499535739421844482e-01,5.954797863960266113e-01,3.215548098087310791e-01,4.378263279795646667e-02,1.954666525125503540e-01,2.490288168191909790e-01,-2.380674891173839569e-02,-1.080467179417610168e-01], 
[-3.453865647315979004e-01,-1.059761047363281250e-01,-4.692391753196716309e-01,-4.369699060916900635e-01,-1.681911759078502655e-02,-1.615533381700515747e-01,-1.469005644321441650e-01,-1.333518624305725098e-01,1.208607405424118042e-01,8.596385270357131958e-02,-2.547604441642761230e-01,-4.087654054164886475e-01,2.983983457088470459e-01,1.522354185581207275e-01,1.852110475301742554e-01,4.491671174764633179e-02,2.325332611799240112e-01,-9.346301853656768799e-02,-2.273132503032684326e-01,-3.597356379032135010e-01,-4.908682107925415039e-01,8.355309069156646729e-02,2.056937664747238159e-02,-5.687685310840606689e-02,-1.201161265373229980e+00,-3.125847578048706055e-01,3.286487758159637451e-01,-3.532351553440093994e-01,3.932851254940032959e-01,8.281950354576110840e-01,-1.294722259044647217e-01,-1.163855865597724915e-01,-4.802327156066894531e-01,5.635440349578857422e-02,1.184429526329040527e-01,-1.134344339370727539e-01,-1.215117573738098145e-01,7.774278521537780762e-02,5.011335611343383789e-01,-5.341957509517669678e-02,4.101458191871643066e-01,1.393298208713531494e-01,2.553727328777313232e-01,2.460131943225860596e-01,4.885313659906387329e-02,3.540842831134796143e-01,1.484221369028091431e-01,-1.171707287430763245e-01], 
[-3.459519445896148682e-01,-2.137374579906463623e-01,4.873414337635040283e-01,8.302395939826965332e-01,-5.091637000441551208e-02,-6.295188516378402710e-02,-5.218213200569152832e-01,-9.181624650955200195e-01,-8.511172980070114136e-02,-5.777450799942016602e-01,-4.882035553455352783e-01,3.267807364463806152e-01,8.102556318044662476e-02,-4.281166568398475647e-02,-1.886780187487602234e-02,1.227272272109985352e+00,9.577680826187133789e-01,7.979171872138977051e-01,-1.214678734540939331e-01,4.762062430381774902e-01,6.313931941986083984e-01,9.554079920053482056e-02,7.072538137435913086e-01,8.093696832656860352e-01,-6.607840061187744141e-01,-3.072120547294616699e-01,2.662767767906188965e-01,3.917893767356872559e-02,2.252827137708663940e-01,2.826895713806152344e-01,-7.952219247817993164e-01,-4.434951543807983398e-01,-1.501554250717163086e-01,2.745675668120384216e-02,-1.151973158121109009e-01,3.433490991592407227e-01,-5.512009263038635254e-01,-4.190637469291687012e-01,-1.445578783750534058e-01,9.125787019729614258e-02,-1.109101530164480209e-02,2.800869941711425781e-01,1.983575075864791870e-01,2.256314642727375031e-02,-1.407668739557266235e-01,-7.361121475696563721e-02,-3.618170917034149170e-01,-3.124438822269439697e-01], 
[1.922176033258438110e-01,-1.261912584304809570e-01,4.040918350219726562e-01,4.025632739067077637e-01,-1.486271768808364868e-01,-7.350756973028182983e-02,-1.522179394960403442e-01,-2.727510631084442139e-01,-4.475723803043365479e-01,8.663473278284072876e-02,2.508518099784851074e-01,2.060984224081039429e-01,1.506023854017257690e-02,-2.680881023406982422e-01,-6.163657903671264648e-01,4.971690475940704346e-01,2.064018249511718750e-01,6.018650531768798828e-01,6.702221930027008057e-02,1.340472251176834106e-01,4.085620045661926270e-01,9.866712987422943115e-02,2.532522939145565033e-02,3.841453194618225098e-01,8.998554944992065430e-01,-9.794536232948303223e-02,-5.807905793190002441e-01,2.427065819501876831e-01,-4.803620576858520508e-01,-1.060010552406311035e+00,1.022140309214591980e-01,3.153875470161437988e-01,4.585806727409362793e-01,-4.341426119208335876e-02,7.745593786239624023e-02,-3.216684758663177490e-01,-9.341292828321456909e-02,-1.789900213479995728e-01,-1.553950607776641846e-01,-2.470640540122985840e-01,-2.969751954078674316e-01,2.434488534927368164e-01,-3.994981348514556885e-01,-8.439631760120391846e-02,2.039301693439483643e-01,-5.943564325571060181e-02,-1.456884890794754028e-01,2.840546332299709320e-02], 
[7.272978778928518295e-03,8.692963980138301849e-03,-5.727803334593772888e-02,1.985026150941848755e-01,-2.364600300788879395e-01,-3.582246899604797363e-01,-2.179019153118133545e-01,4.296927154064178467e-01,-3.194957971572875977e-02,-3.205410242080688477e-01,-2.405509352684020996e-02,-2.030661553144454956e-01,-1.596434116363525391e-01,1.190623939037322998e-01,3.396236300468444824e-01,-3.875084817409515381e-01,-4.358614087104797363e-01,-4.044428467750549316e-01,1.038493156433105469e+00,3.811030089855194092e-01,7.409285753965377808e-02,-3.832121193408966064e-01,-7.872769236564636230e-01,-4.602785706520080566e-01,5.224564671516418457e-01,5.982733964920043945e-01,8.414568901062011719e-01,1.152210384607315063e-01,6.105245649814605713e-02,4.387020468711853027e-01,4.657056033611297607e-01,-2.377872765064239502e-01,-8.525408506393432617e-01,7.272409796714782715e-01,5.580896511673927307e-02,-2.853922545909881592e-01,6.303297281265258789e-01,-4.766767621040344238e-01,-6.735102534294128418e-01,2.126588523387908936e-01,-1.028687357902526855e-01,-3.392000496387481689e-01,2.498059719800949097e-01,1.745581179857254028e-01,3.220563828945159912e-01,3.865771889686584473e-01,-6.209917366504669189e-02,4.741534218192100525e-02], 
[1.618360579013824463e-01,9.163953661918640137e-01,4.240742027759552002e-01,-3.083733916282653809e-01,1.888215839862823486e-01,-4.949066340923309326e-01,6.664124727249145508e-01,3.301397562026977539e-01,2.881737649440765381e-01,8.488897681236267090e-01,3.716635406017303467e-01,2.926147878170013428e-01,-4.058343470096588135e-01,-4.790077507495880127e-01,-3.855452239513397217e-01,6.511480212211608887e-01,3.449667096138000488e-01,-2.666735053062438965e-01,-3.689637407660484314e-02,3.925099223852157593e-02,-4.207477271556854248e-01,-2.155617438256740570e-02,8.966187387704849243e-02,-1.171492561697959900e-01,-8.043417334556579590e-01,-3.203189373016357422e-01,-3.663089573383331299e-01,1.525911390781402588e-01,4.950171113014221191e-01,3.768462240695953369e-01,-7.535721659660339355e-01,-7.971494197845458984e-01,-7.899171113967895508e-01,-9.383701533079147339e-02,-6.492786854505538940e-02,3.061419725418090820e-01,-1.068482547998428345e-01,1.379740238189697266e-01,-1.086805313825607300e-01,-2.572957873344421387e-01,1.058437488973140717e-02,-7.166638374328613281e-01,2.450869828462600708e-01,3.895846009254455566e-01,-1.893006414175033569e-01,1.026583611965179443e-01,4.621953070163726807e-01,4.910500347614288330e-01] ];

var tdnn1Bias = [5.362448096275329590e-01, 
-7.560657262802124023e-01, 
6.926202774047851562e-02, 
5.252105593681335449e-01, 
-1.155071258544921875e+00, 
7.222492694854736328e-01, 
1.003656610846519470e-01, 
1.187776684761047363e+00];

var tdnn2Weight = [[-1.335004329681396484e+00,-3.724779784679412842e-01,-1.679126381874084473e+00,7.451456189155578613e-01,-7.813916802406311035e-01,-4.233115613460540771e-01,2.098637372255325317e-01,9.374514222145080566e-01,1.506308436393737793e+00,-3.626474738121032715e-01,-1.514259725809097290e-01,4.422380328178405762e-01,-1.731797754764556885e-01,-7.271988689899444580e-02,3.479409217834472656e-01,5.102519989013671875e-01,-2.961184978485107422e-01,1.366954147815704346e-01,1.007673859596252441e+00,2.976249456405639648e-01,7.475045919418334961e-01,2.105935811996459961e-01,-5.583910942077636719e-01,-1.016762971878051758e+00], 
[-2.028425693511962891e+00,1.148884415626525879e+00,7.755841016769409180e-01,3.214877843856811523e+00,-8.416063785552978516e-01,-1.814857721328735352e+00,1.510740399360656738e+00,1.432322144508361816e+00,3.727772533893585205e-01,-1.934915423393249512e+00,-4.430717527866363525e-01,-1.309582114219665527e+00,-1.802784800529479980e+00,2.786726504564285278e-02,7.205623388290405273e-01,2.101324319839477539e+00,1.021720767021179199e+00,1.017131328582763672e+00,4.168103337287902832e-01,-1.033692359924316406e+00,-4.686785638332366943e-01,-2.249528646469116211e+00,-1.356131553649902344e+00,-1.215999245643615723e+00], 
[-3.266112506389617920e-01,-1.192969322204589844e+00,4.956808388233184814e-01,-1.807245969772338867e+00,2.732292935252189636e-02,5.021244883537292480e-01,-7.315289974212646484e-01,1.285398483276367188e+00,6.767660975456237793e-01,1.108877584338188171e-01,2.798820436000823975e-01,7.615875601768493652e-01,3.504985809326171875e+00,6.392170786857604980e-01,-1.266905546188354492e+00,-2.151490747928619385e-01,1.171389669179916382e-01,-6.162536144256591797e-01,-2.789105892181396484e+00,-3.500140309333801270e-01,-8.906414508819580078e-01,3.414770960807800293e-01,2.972684204578399658e-01,1.493515223264694214e-01] ];

var tdnn2Bias = [5.037331581115722656e-01, 
5.090649724006652832e-01, 
-1.856959164142608643e-01];

var linearWeight = [[-1.275321364402770996e+00,-4.543158113956451416e-01,3.914128541946411133e-01,5.705081224441528320e-01,2.209817320108413696e-01,-3.969017267227172852e-01,-9.858759641647338867e-01,-1.136175751686096191e+00,-1.894137382507324219e+00,-2.691997289657592773e+00,-1.358188271522521973e+00,-1.686157584190368652e-01,3.143717944622039795e-01,-6.356549859046936035e-01,-8.238990306854248047e-01,-1.295418143272399902e+00,-5.842572450637817383e-01,-6.720532774925231934e-01,4.231986045837402344e+00,3.462161064147949219e+00,1.694118857383728027e+00,7.500541806221008301e-01,2.983818054199218750e-01,2.143852859735488892e-01,4.505692794919013977e-02,-4.713188111782073975e-01,-9.198500514030456543e-01], 
[-8.496687561273574829e-02,1.881149262189865112e-01,-1.177444756031036377e-01,1.318774223327636719e-01,6.903445720672607422e-01,1.243825674057006836e+00,1.214125990867614746e+00,1.695776104927062988e+00,2.118555307388305664e+00,-3.085791826248168945e+00,-1.141215682029724121e+00,-1.220597147941589355e+00,-4.660932421684265137e-01,1.449039280414581299e-01,4.009313583374023438e-01,5.768314599990844727e-01,-8.639951944351196289e-01,-5.542716979980468750e-01,-2.193373918533325195e+00,-2.377320528030395508e+00,-9.505733847618103027e-01,-2.308136522769927979e-01,-2.855276130139827728e-02,-1.937471777200698853e-01,4.755280613899230957e-01,1.153797388076782227e+00,9.530591964721679688e-01], 
[1.346620559692382812e+00,1.635186523199081421e-01,-6.337150931358337402e-01,-8.352040052413940430e-01,-6.488962173461914062e-01,-3.833971321582794189e-01,-4.799719750881195068e-01,-3.569128811359405518e-01,-4.100141227245330811e-01,5.919455051422119141e+00,2.940081596374511719e+00,1.031345009803771973e+00,1.490654796361923218e-01,2.877749800682067871e-01,2.698507308959960938e-01,5.754169225692749023e-01,1.367343544960021973e+00,1.256675362586975098e+00,-1.907542943954467773e+00,-1.138034701347351074e+00,-5.459690690040588379e-01,-4.889220297336578369e-01,-5.166100859642028809e-01,-1.903004758059978485e-02,-2.252324968576431274e-01,-7.363407611846923828e-01,-1.005339622497558594e-01] ];

var linearBias = [7.878746986389160156e-01, 
5.369362831115722656e-01, 
-1.389516949653625488e+00];


var mean = tf.tensor([[-1.002117233276367188e+02,-9.577408599853515625e+01,-9.068260955810546875e+01,-8.908010101318359375e+01,-8.883013153076171875e+01,-8.932281494140625000e+01,-9.016002655029296875e+01,-9.055487060546875000e+01,-9.070959472656250000e+01,-9.072010803222656250e+01,-9.088055419921875000e+01,-9.074184417724609375e+01,-9.079324340820312500e+01,-9.090682983398437500e+01,-9.094776153564453125e+01], 
[-9.118019866943359375e+01,-8.533576965332031250e+01,-8.066574859619140625e+01,-7.937021636962890625e+01,-7.982843780517578125e+01,-8.063751220703125000e+01,-8.098416900634765625e+01,-8.120027160644531250e+01,-8.101397705078125000e+01,-8.080242156982421875e+01,-8.099962615966796875e+01,-8.102444458007812500e+01,-8.101643371582031250e+01,-8.123394012451171875e+01,-8.158362579345703125e+01], 
[-7.908394622802734375e+01,-6.795561981201171875e+01,-6.299384307861328125e+01,-6.415457916259765625e+01,-6.698778533935546875e+01,-6.903987121582031250e+01,-7.090476226806640625e+01,-7.205741882324218750e+01,-7.236093902587890625e+01,-7.263012695312500000e+01,-7.282294464111328125e+01,-7.315522766113281250e+01,-7.335762786865234375e+01,-7.344966125488281250e+01,-7.365165710449218750e+01], 
[-7.723698425292968750e+01,-6.609275054931640625e+01,-6.033831405639648438e+01,-6.125852584838867188e+01,-6.495683288574218750e+01,-6.807265472412109375e+01,-6.998278808593750000e+01,-7.079728698730468750e+01,-7.103822326660156250e+01,-7.083998870849609375e+01,-7.051218414306640625e+01,-6.968090057373046875e+01,-6.903481292724609375e+01,-6.869385528564453125e+01,-6.874176788330078125e+01], 
[-7.412657165527343750e+01,-6.228937149047851562e+01,-5.660129928588867188e+01,-5.796651840209960938e+01,-6.337821197509765625e+01,-6.802429962158203125e+01,-7.007588958740234375e+01,-7.264845275878906250e+01,-7.526815032958984375e+01,-7.713230895996093750e+01,-7.818622589111328125e+01,-7.839830780029296875e+01,-7.877249145507812500e+01,-7.847310638427734375e+01,-7.832982635498046875e+01], 
[-8.387055969238281250e+01,-7.378907012939453125e+01,-6.789600372314453125e+01,-6.708167266845703125e+01,-6.776391601562500000e+01,-6.850004577636718750e+01,-6.909053039550781250e+01,-6.977168273925781250e+01,-7.027288055419921875e+01,-7.040834808349609375e+01,-7.011553955078125000e+01,-6.955569458007812500e+01,-6.934315490722656250e+01,-6.950107574462890625e+01,-6.963580322265625000e+01], 
[-9.155134582519531250e+01,-8.371862792968750000e+01,-7.816220092773437500e+01,-7.634665679931640625e+01,-7.591387939453125000e+01,-7.560898590087890625e+01,-7.545040893554687500e+01,-7.523364257812500000e+01,-7.526804351806640625e+01,-7.517501831054687500e+01,-7.499105072021484375e+01,-7.477107238769531250e+01,-7.487026977539062500e+01,-7.501760864257812500e+01,-7.523283386230468750e+01], 
[-9.680385589599609375e+01,-8.962319946289062500e+01,-8.452355957031250000e+01,-8.263411712646484375e+01,-8.217227172851562500e+01,-8.184000396728515625e+01,-8.135627746582031250e+01,-8.097808074951171875e+01,-8.064376068115234375e+01,-8.011406707763671875e+01,-7.963831329345703125e+01,-7.897914123535156250e+01,-7.823196411132812500e+01,-7.798129272460937500e+01,-7.784097290039062500e+01], 
[-9.823347473144531250e+01,-9.039065551757812500e+01,-8.504531097412109375e+01,-8.340519714355468750e+01,-8.330726623535156250e+01,-8.358108520507812500e+01,-8.383286285400390625e+01,-8.394358062744140625e+01,-8.390792083740234375e+01,-8.363030242919921875e+01,-8.322698974609375000e+01,-8.278472900390625000e+01,-8.263745880126953125e+01,-8.279560852050781250e+01,-8.325686645507812500e+01], 
[-9.404048156738281250e+01,-8.648565673828125000e+01,-8.192173767089843750e+01,-8.076251983642578125e+01,-8.140911102294921875e+01,-8.252915954589843750e+01,-8.333788299560546875e+01,-8.397171020507812500e+01,-8.424188995361328125e+01,-8.445464324951171875e+01,-8.419017028808593750e+01,-8.378512573242187500e+01,-8.363720703125000000e+01,-8.400854492187500000e+01,-8.481444549560546875e+01], 
[-9.245171356201171875e+01,-8.478999328613281250e+01,-8.019131469726562500e+01,-7.909461975097656250e+01,-7.977750396728515625e+01,-8.125591278076171875e+01,-8.258496856689453125e+01,-8.377391815185546875e+01,-8.458017730712890625e+01,-8.513094329833984375e+01,-8.536252593994140625e+01,-8.543948364257812500e+01,-8.571141052246093750e+01,-8.615319061279296875e+01,-8.681996917724609375e+01], 
[-9.252978515625000000e+01,-8.485858154296875000e+01,-8.018564605712890625e+01,-7.924929809570312500e+01,-8.072441864013671875e+01,-8.251803588867187500e+01,-8.433859252929687500e+01,-8.509761810302734375e+01,-8.482872009277343750e+01,-8.465825653076171875e+01,-8.461141967773437500e+01,-8.516030883789062500e+01,-8.610494232177734375e+01,-8.689176940917968750e+01,-8.781414031982421875e+01], 
[-9.245498657226562500e+01,-8.486826324462890625e+01,-8.033454132080078125e+01,-7.970590209960937500e+01,-8.071962738037109375e+01,-8.282256317138671875e+01,-8.387088775634765625e+01,-8.478997802734375000e+01,-8.482357788085937500e+01,-8.460531616210937500e+01,-8.493441009521484375e+01,-8.538180541992187500e+01,-8.646597290039062500e+01,-8.739923095703125000e+01,-8.800405120849609375e+01], 
[-9.306197357177734375e+01,-8.555018615722656250e+01,-8.090498352050781250e+01,-8.007194519042968750e+01,-8.160449981689453125e+01,-8.278910827636718750e+01,-8.373555755615234375e+01,-8.454074096679687500e+01,-8.435779571533203125e+01,-8.501428985595703125e+01,-8.582287597656250000e+01,-8.587295532226562500e+01,-8.687106323242187500e+01,-8.766886138916015625e+01,-8.779148864746093750e+01], 
[-9.329986572265625000e+01,-8.578216552734375000e+01,-8.153522491455078125e+01,-8.062363433837890625e+01,-8.178515625000000000e+01,-8.318257904052734375e+01,-8.383527374267578125e+01,-8.423461914062500000e+01,-8.457284545898437500e+01,-8.512177276611328125e+01,-8.624547576904296875e+01,-8.656912231445312500e+01,-8.722628784179687500e+01,-8.797763824462890625e+01,-8.855437469482421875e+01], 
[-9.282476806640625000e+01,-8.543283081054687500e+01,-8.145740509033203125e+01,-8.092426300048828125e+01,-8.189454650878906250e+01,-8.302781677246093750e+01,-8.351535034179687500e+01,-8.381501770019531250e+01,-8.401467895507812500e+01,-8.493599700927734375e+01,-8.615591430664062500e+01,-8.692542266845703125e+01,-8.746574401855468750e+01,-8.753370666503906250e+01,-8.828668975830078125e+01] ]);

var std = tf.tensor([[6.719078540802001953e+00,5.749404907226562500e+00,4.898271083831787109e+00,4.398175239562988281e+00,4.545714855194091797e+00,4.250361919403076172e+00,4.622580051422119141e+00,4.593819141387939453e+00,4.510543346405029297e+00,4.801941394805908203e+00,4.934604167938232422e+00,4.654274940490722656e+00,4.764934539794921875e+00,4.584099292755126953e+00,4.782719612121582031e+00], 
[6.342644691467285156e+00,5.974368572235107422e+00,4.976337909698486328e+00,4.246608734130859375e+00,3.895013570785522461e+00,3.661607980728149414e+00,3.548603057861328125e+00,3.768376350402832031e+00,3.952059030532836914e+00,4.035831451416015625e+00,4.192992687225341797e+00,3.970843791961669922e+00,4.175644397735595703e+00,4.045134544372558594e+00,4.143473148345947266e+00], 
[8.881951332092285156e+00,5.157588005065917969e+00,3.656180143356323242e+00,3.858328580856323242e+00,3.638562202453613281e+00,3.682990550994873047e+00,3.623205661773681641e+00,3.452733516693115234e+00,3.375993013381958008e+00,3.671075344085693359e+00,3.928206205368041992e+00,4.224139690399169922e+00,4.436187267303466797e+00,4.420345306396484375e+00,4.553473949432373047e+00], 
[1.026656818389892578e+01,6.485726356506347656e+00,4.820881366729736328e+00,5.149184703826904297e+00,5.053880214691162109e+00,4.875271797180175781e+00,4.541214942932128906e+00,4.059286594390869141e+00,4.046182155609130859e+00,4.104789257049560547e+00,4.324111938476562500e+00,4.594259738922119141e+00,4.775348663330078125e+00,5.026633739471435547e+00,5.076302528381347656e+00], 
[1.027587223052978516e+01,6.892772674560546875e+00,6.261926174163818359e+00,6.660936832427978516e+00,6.768378257751464844e+00,5.874133586883544922e+00,5.715199470520019531e+00,5.764828205108642578e+00,5.707117080688476562e+00,5.624756813049316406e+00,5.580380439758300781e+00,5.567800521850585938e+00,5.829201698303222656e+00,5.972345352172851562e+00,5.712526321411132812e+00], 
[1.023552799224853516e+01,8.873279571533203125e+00,8.686912536621093750e+00,8.665233612060546875e+00,8.663433074951171875e+00,8.631060600280761719e+00,8.726878166198730469e+00,8.895898818969726562e+00,9.076897621154785156e+00,9.549194335937500000e+00,9.425239562988281250e+00,9.217574119567871094e+00,9.178993225097656250e+00,9.062804222106933594e+00,9.036879539489746094e+00], 
[9.536827087402343750e+00,9.005516052246093750e+00,9.786251068115234375e+00,9.975633621215820312e+00,9.854873657226562500e+00,9.689513206481933594e+00,9.709946632385253906e+00,9.855302810668945312e+00,9.990579605102539062e+00,1.021886825561523438e+01,9.957658767700195312e+00,9.766823768615722656e+00,9.817954063415527344e+00,1.007981967926025391e+01,1.027569293975830078e+01], 
[1.003920841217041016e+01,1.051316261291503906e+01,1.123549461364746094e+01,1.136150932312011719e+01,1.096458721160888672e+01,1.055384445190429688e+01,1.057018280029296875e+01,1.061071109771728516e+01,1.106089878082275391e+01,1.168393325805664062e+01,1.186556053161621094e+01,1.231734657287597656e+01,1.273614788055419922e+01,1.282299613952636719e+01,1.289178752899169922e+01], 
[1.214770317077636719e+01,1.163953304290771484e+01,1.166370391845703125e+01,1.150164508819580078e+01,1.127138233184814453e+01,1.107936859130859375e+01,1.114136028289794922e+01,1.129504585266113281e+01,1.148153686523437500e+01,1.164847278594970703e+01,1.154589080810546875e+01,1.156838035583496094e+01,1.147459125518798828e+01,1.177632713317871094e+01,1.188506031036376953e+01], 
[1.157553195953369141e+01,9.312150001525878906e+00,8.497574806213378906e+00,8.526160240173339844e+00,8.945981025695800781e+00,9.213360786437988281e+00,9.701611518859863281e+00,1.020994281768798828e+01,1.048670959472656250e+01,1.075112056732177734e+01,1.102964591979980469e+01,1.109640312194824219e+01,1.115660285949707031e+01,1.147415924072265625e+01,1.160012340545654297e+01], 
[1.211823081970214844e+01,8.708394050598144531e+00,7.605426788330078125e+00,7.649981498718261719e+00,7.861310005187988281e+00,7.918128490447998047e+00,7.911915779113769531e+00,8.072831153869628906e+00,7.997637271881103516e+00,7.867955684661865234e+00,7.822262763977050781e+00,7.707031726837158203e+00,7.316214084625244141e+00,7.373615264892578125e+00,7.268844127655029297e+00], 
[1.377928733825683594e+01,1.137340545654296875e+01,1.093579387664794922e+01,1.141959667205810547e+01,1.227519130706787109e+01,1.177817726135253906e+01,1.216489791870117188e+01,1.214098548889160156e+01,1.171675586700439453e+01,1.177224445343017578e+01,1.111558437347412109e+01,1.085806179046630859e+01,1.094001674652099609e+01,1.091232776641845703e+01,1.054715156555175781e+01], 
[1.381102657318115234e+01,1.122176837921142578e+01,1.017683982849121094e+01,1.097564601898193359e+01,1.162026500701904297e+01,1.183752155303955078e+01,1.097929859161376953e+01,1.157479572296142578e+01,1.188644790649414062e+01,1.164066219329833984e+01,1.111252307891845703e+01,1.089731979370117188e+01,1.026377487182617188e+01,1.068065452575683594e+01,1.040765380859375000e+01], 
[1.413106250762939453e+01,1.146474552154541016e+01,9.791912078857421875e+00,1.024342918395996094e+01,1.145886230468750000e+01,1.150324058532714844e+01,1.173241615295410156e+01,1.165956974029541016e+01,1.165893268585205078e+01,1.177848720550537109e+01,1.143704414367675781e+01,1.031453037261962891e+01,9.571035385131835938e+00,1.035306549072265625e+01,9.981118202209472656e+00], 
[1.403078842163085938e+01,1.099930000305175781e+01,9.737999916076660156e+00,1.026885318756103516e+01,1.089288711547851562e+01,1.209579372406005859e+01,1.223530578613281250e+01,1.165033435821533203e+01,1.171885204315185547e+01,1.174365234375000000e+01,1.169549083709716797e+01,1.079526519775390625e+01,1.025603389739990234e+01,1.007524776458740234e+01,1.020457363128662109e+01], 
[1.374530220031738281e+01,1.060841846466064453e+01,9.919692039489746094e+00,1.057711696624755859e+01,1.120086002349853516e+01,1.226619720458984375e+01,1.205610179901123047e+01,1.221717739105224609e+01,1.158854961395263672e+01,1.144622135162353516e+01,1.138123989105224609e+01,1.108569431304931641e+01,1.042210388183593750e+01,1.036419677734375000e+01,9.880658149719238281e+00] ]);


// mean = tf.transpose(mean, [1, 0]);
// std = tf.transpose(std, [1, 0]);
/** =====================================================
 *            TDNN1 Processing Params
 * =====================================================
 */

tdnn1Weight = tf.tensor(tdnn1Weight);
tdnn1Weight = tf.reshape(tdnn1Weight, [8, 16, 3]);
// tf.print(tdnn1Weight);
tdnn1Weight = tf.transpose(tdnn1Weight, [2, 1, 0]);
tdnn1Bias = tf.tensor(tdnn1Bias);


if(DEBUG){
  console.log('===== TDNN1 LAYER =====');
  console.log("weight", tdnn1Weight.shape, "should be (8, 16, 3)")
  console.log("bias", tdnn1Bias.shape, "should be (8,");
}


/** =====================================================
 *            TDNN2 Processing Params
 * =====================================================
 */
tdnn2Weight = tf.tensor(tdnn2Weight);
tdnn2Weight = tf.reshape(tdnn2Weight, [3, 8, 3]);
// tf.print(tdnn2Weight);
tdnn2Weight = tf.transpose(tdnn2Weight, [2, 1, 0]);
tdnn2Bias = tf.tensor(tdnn2Bias);

if(DEBUG){
  console.log('===== TDNN2 LAYER =====');
  console.log("weight", tdnn2Weight.shape, "should be (3, 8, 3)")
  console.log("bias", tdnn2Bias.shape, "should be (3,");
}

/** =====================================================
 *            LINEAR Processing Params
 * =====================================================
 */
linearWeight = tf.tensor(linearWeight);
linearWeight = tf.transpose(linearWeight, [1, 0]);
linearBias = tf.tensor(linearBias);

if(DEBUG){
  console.log('===== LINEAR LAYER =====');
  console.log("weight", linearWeight.shape, "should be (3, 27)")
  console.log("bias", linearBias.shape, "should be (3,");
}
  
/** =====================================================
 *           MODEL DEFINITION
 * =====================================================
 */


const model = tf.sequential();
  
// Adding tdnn1
const context1 = [-1, 0, 1];
model.add(tf.layers.conv1d({inputShape: [15, 16],
                            filters: 8,
                            kernelSize: 3,
                            dilationRate: 1, // 0 - (-1)
                            padding: 'valid', // no padding
                            // dataFormat: 'channelsFirst'
                            // bias term ?
                            }));
// console.log(tdnn1Weight.shape, tdnn1Bias.shape);

model.layers[0].setWeights([tdnn1Weight, tdnn1Bias]);

// Adding Sigmoid 1
model.add(tf.layers.activation({activation: 'sigmoid'}));

// Adding tdnn2
const context2 = [-2, 0, 2];
model.add(tf.layers.conv1d({inputShape: [13, 8],
                            filters: 3,
                            kernelSize: 3,
                            dilationRate: 2, // 0 - (-2)
                            padding: 'valid', // no padding
                            // dataFormat: 'channelsFirst'
                            // bias term ?
                            }));
model.layers[2].setWeights([tdnn2Weight, tdnn2Bias]);

// Adding Sigmoid 2
model.add(tf.layers.activation({activation: 'sigmoid'}));

// Adding Flatten
model.add(tf.layers.flatten({dataFormat: 'channelsFirst'}));

// Adding Linear
model.add(tf.layers.dense({inputDim: 27,
                          units: 3,
                          useBias: true}));
model.layers[5].setWeights([linearWeight, linearBias]);
  

/** =====================================================
 *           COMPILING THE MODEL
 * =====================================================
 */
model.compile({loss: 'categoricalCrossentropy', optimizer: 'sgd'});


/** =====================================================
 *           TESTING THE MODEL PLS REMOVE LATER
 * =====================================================
 */

// var subbed = tf.sub(dataTensor, mean);
//     var dataTensorNormed = tf.div(subbed, std);
//     dataTensorNormed = dataTensorNormed.expandDims(0);

// var y = model.predict(dataTensorNormed, {batchSize: 1});
// console.log(y.dataSync);

// console.log('something is working at least');
// var files;
// var samples = [];
// var samples_tensors = [];
// var num_files;
// document.getElementById("filepicker").addEventListener("change", function(event) {
//   files = event.target.files;
//   console.log('something changed');
//   console.log(files);
// }, false);

// document.getElementById('submit').onclick = () => {
//     console.log('clicked');
//     var file_list = files;
//     let promises = [];
//     for (let file of file_list) {
//         let filePromise = new Promise(resolve => {
//             let reader = new FileReader();
//             reader.readAsText(file);
//             reader.onload = () => resolve(reader.result);
//         });
//         promises.push(filePromise);
//     }

//     Promise.all(promises).then(fileContents => {
//         // fileContents will be an array containing
//         // the contents of the files, perform the
//         // character replacements and other transformations
//         // here as needed
//         console.log(fileContents.length);
//         num_files = fileContents.length;
//         for (let i = 0; i < fileContents.length; i++) {
//             var file = fileContents[i];
//             const rows = file.split("\n");
//             for (let j = 0; j < rows.length - 1; j++) {
//                 rows[j] = rows[j].split(",");
//                 for (let k = 0; k < rows[j].length; k ++){
//                     // console.log(rows[j][k])
//                     rows[j][k] = Number(rows[j][k]);
//                 }
//             }
//             samples.push(rows);
//             // console.log('1', samples);
//         }

//         for (let j = 0; j < samples.length; j ++){
//             let data = samples[j];
//             // console.log(data);
     
//             // sum columns
//             var matrix = data
//             const numRows = matrix.length;
//             const numCols = matrix[0].length; // Assuming all rows have the same number of columns
        
//             const columnSums = new Array(numCols).fill(0);
        
//             // console.log(numCols, numRows);
//             for (let col = 0; col < numCols; col++) {
//               for (let row = 0; row < numRows; row++) {
//                 if(matrix[row][col] == undefined){
//                     continue;
//                 }
//                 columnSums[col] += 10**(matrix[row][col]);
//               }
//             }
//             // console.log(columnSums);
//             // custom max
//             if (columnSums.length === 0) {
//               return undefined; // Return undefined if no columnSums are provided
//             }
//             let max = -Infinity; // Start with a very low value
//             for (let i = 1; i < columnSums.length; i++) {
//               if (columnSums[i] > max) {
//                 max = columnSums[i];
//               }
//             }
//             // console.log(max);
//             // normalize
//             var array_2 = Array(columnSums);
//             for(var i = 0, length = columnSums.length; i < length; i++){
//                 array_2[i] = columnSums[i] / max;
//             }
//             // console.log(array_2);
//             // find max
//             const thresh_indexes = [];
//             for (let i = 2; i < array_2.length; i++) {
//               if (array_2[i] > 0.3) {
//                 thresh_indexes.push(i);
//               }
//             }
        
//             let start_time_ms = thresh_indexes[0]*10 - 20;
//             // console.log('start time', start_time_ms);
    
//             // to capture onset in msec
    
//             var start_frame = start_time_ms / 10;
//             var currDat = tf.tensor(data);
//             var the_dat = currDat.slice([0, start_frame -1 ], [16, 15]);
//             var dataTensor = the_dat; // no transpose
//             samples_tensors.push(dataTensor);
            
//         }
//         let dataTensors = tf.stack(samples_tensors);
//         tf.print(dataTensors);
//         console.log('sample_tensors', samples_tensors.shape);
//         console.log('dataTensors shape, mean shape, std shape', dataTensors.shape, mean.shape, std.shape);
//         var subbed = tf.sub(dataTensors, mean);
//         var dataTensorNormed = tf.div(subbed, std);

//         var dataTensorNormedTransposed = tf.transpose(dataTensorNormed, [0, 2, 1]);
//         tf.print(dataTensorNormedTransposed);
//         var y = model.predict(dataTensorNormedTransposed, {batchSize: 1}); // predicting the model

//         var y_arr  = Array.from(y.dataSync());
//         var the_str = "";
//         console.log(typeof(y.dataSync));
//         for(let i=0; i < y_arr.length; i += 3) {
//             the_str = the_str + y_arr[i]+ ","+ y_arr[i+1]+ ","+ y_arr[i + 2]+ "\n";
//             console.log( y_arr[i]+ ","+ y_arr[i+1]+ ","+ y_arr[i + 2]+ "\n")
//         }
//         console.log(the_str);
//         document.getElementById('class34').innerHTML = the_str;
//         var normed_y = tf.add(tf.div(tf.sub(y, y.mean()), 20), 0.5);
//         // tf.image.resize(normed_y, [100, 100]);
//         tf.print(normed_y);
//         tf.browser.toPixels(normed_y, document.getElementsByTagName("canvas")[0]);
//     });
// }

// document.getElementById('single').addEventListener('click', function(){
//   console.log('lmfoa');
//   var arr = Array(16).fill().map(() => Array(15).fill(0));
//   arr[10][9] = 1
//   console.log(arr);
//   var single_sample = tf.tensor(arr);
//   single_sample = tf.transpose(single_sample, [1, 0]);
//   console.log(single_sample.shape);
//   var single_sample_set = tf.expandDims(single_sample, 0)
//   console.log('single sample set shape', single_sample_set.shape);
//   var y = model.predict(single_sample_set, {batchSize: 1});
//   console.log('predicted values (y)');
//   tf.print(y);
//   document.getElementById('class34').innerHTML = y;
// });