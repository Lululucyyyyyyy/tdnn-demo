const DEBUG = true; 
/** ===================================================== 
 *            PARAMETERS  
* ===================================================== 
*/

var tdnn1Weight = [[1.615776866674423218e-01,7.173244357109069824e-01,-3.600513339042663574e-01,-4.317452013492584229e-01,3.240724205970764160e-01,3.481120467185974121e-01,1.264496922492980957e+00,-4.993483126163482666e-01,-1.716446876525878906e+00,4.578845798969268799e-01,1.375126242637634277e-01,4.878286719322204590e-01,-5.463902950286865234e-01,3.269463777542114258e-01,1.437870383262634277e+00,-3.207933008670806885e-01,-7.860606908798217773e-01,-3.932978212833404541e-01,1.208361148834228516e+00,2.353659570217132568e-01,-1.731502294540405273e+00,1.235476493835449219e+00,9.039855003356933594e-02,-1.526147127151489258e+00,3.403080701828002930e-01,1.592718511819839478e-01,-2.563360691070556641e+00,-2.206249952316284180e+00,-7.624908089637756348e-01,1.445505380630493164e+00,-9.339705109596252441e-01,7.275081276893615723e-01,1.503642678260803223e+00,6.353123486042022705e-02,8.030033111572265625e-01,1.321001052856445312e-01,4.116766154766082764e-01,-1.856408268213272095e-01,-6.734496951103210449e-01,-5.280317068099975586e-01,2.325821816921234131e-01,-2.927263975143432617e-01,-4.784611463546752930e-01,-8.270282149314880371e-01,-2.972904443740844727e-01,-1.007838129997253418e+00,6.109450757503509521e-02,5.200375318527221680e-01], 
[-3.452513832598924637e-03,-1.336529970169067383e+00,1.308408498764038086e+00,-3.600727319717407227e-01,-1.175992935895919800e-02,5.393897294998168945e-01,2.284594327211380005e-01,2.087873965501785278e-02,-1.792060136795043945e-01,1.588798642158508301e+00,-6.548770666122436523e-01,-3.042885959148406982e-01,-1.409039139747619629e+00,7.898665666580200195e-01,-5.386318564414978027e-01,1.924874544143676758e+00,1.953338645398616791e-02,-2.282663106918334961e+00,1.266022771596908569e-01,7.581895589828491211e-01,-3.125071525573730469e-01,1.693768948316574097e-01,9.127893447875976562e-01,2.597072720527648926e-01,-3.332494735717773438e+00,5.335280895233154297e-01,1.814848899841308594e+00,-2.814588308334350586e+00,2.285535097122192383e+00,3.076463699340820312e+00,-9.333670735359191895e-01,-2.458352297544479370e-01,6.821522116661071777e-01,1.597627997398376465e+00,-4.668914154171943665e-02,-1.084846377372741699e+00,8.561133742332458496e-01,-5.740663409233093262e-01,-1.318393647670745850e-01,3.352861702442169189e-01,-4.147407412528991699e-01,6.692518591880798340e-01,8.568079471588134766e-01,7.292349338531494141e-01,8.745934963226318359e-01,1.184138059616088867e+00,-6.561943292617797852e-01,-6.916561126708984375e-01], 
[-9.521145373582839966e-02,-3.556398907676339149e-03,1.609508246183395386e-01,2.984281256794929504e-02,-2.202465385198593140e-01,1.361910402774810791e-01,3.772200345993041992e-01,-8.964907526969909668e-01,-7.566453814506530762e-01,1.897730082273483276e-01,1.050844714045524597e-01,4.187459051609039307e-01,4.666791260242462158e-01,1.352828383445739746e+00,-1.116645216941833496e+00,1.446906805038452148e+00,1.975263476371765137e+00,3.778667449951171875e-01,9.931989312171936035e-01,2.135787248611450195e+00,-6.897564530372619629e-01,-2.619797468185424805e+00,4.272846281528472900e-01,-1.666705250740051270e+00,-4.296140372753143311e-01,1.042730093002319336e+00,-4.769817292690277100e-01,2.500162124633789062e-01,-3.316564485430717468e-02,-5.073943138122558594e-01,-1.062359571456909180e+00,-1.575413048267364502e-01,1.595913767814636230e+00,-1.887445151805877686e-01,4.813833534717559814e-01,-2.668620049953460693e-01,1.545606702566146851e-01,-1.183445081114768982e-01,6.691344976425170898e-01,-3.985083699226379395e-01,-9.626450948417186737e-03,-5.791527405381202698e-02,-1.748253941535949707e+00,4.977345764636993408e-01,1.406136900186538696e-01,-5.424109697341918945e-01,5.862668156623840332e-01,4.138270020484924316e-01], 
[-4.003115892410278320e-01,-4.915787652134895325e-02,4.963667094707489014e-01,1.228542685508728027e+00,5.958993434906005859e-01,-1.038601636886596680e+00,1.377211689949035645e+00,1.317198425531387329e-01,-1.696777462959289551e+00,3.302029967308044434e-01,2.122111916542053223e-01,-9.592766165733337402e-01,1.158550605177879333e-01,-1.370083987712860107e-01,8.381529450416564941e-01,7.704634070396423340e-01,4.387546181678771973e-01,1.982323169708251953e+00,-9.867287240922451019e-03,-1.541227340698242188e+00,4.897983670234680176e-01,-2.086551904678344727e+00,9.413799643516540527e-02,6.608991622924804688e-01,-1.114586472511291504e+00,3.201376795768737793e-01,6.620292663574218750e-01,1.921589732170104980e+00,3.258171677589416504e-01,8.978007435798645020e-01,-2.343396186828613281e+00,-1.781494498252868652e+00,-1.351436972618103027e-01,-2.623797357082366943e-01,6.619822978973388672e-02,-5.733544230461120605e-01,-1.265271455049514771e-01,-1.048604771494865417e-01,5.792835354804992676e-01,-7.128936648368835449e-01,1.378682374954223633e+00,-6.105946302413940430e-01,1.088364839553833008e+00,-1.324118971824645996e+00,-1.623441576957702637e-01,-8.682973682880401611e-02,-1.606066077947616577e-01,-1.905658282339572906e-02], 
[1.615658164024353027e+00,-1.153308153152465820e+00,5.100411549210548401e-02,4.145072698593139648e-01,9.433984160423278809e-01,-1.478326469659805298e-01,7.164877653121948242e-01,-7.471098303794860840e-01,-1.114450931549072266e+00,1.337672829627990723e+00,-6.986818909645080566e-01,1.271316885948181152e+00,7.793805003166198730e-01,3.348873257637023926e-01,1.484649538993835449e+00,-1.057517647743225098e+00,-2.664629518985748291e-01,-9.116870164871215820e-01,2.144865274429321289e+00,-5.753396153450012207e-01,-1.923464179039001465e+00,1.785091638565063477e+00,-1.761712282896041870e-01,8.973836302757263184e-01,-1.430236250162124634e-01,-9.725254774093627930e-01,1.143494844436645508e+00,-1.342817306518554688e+00,-1.426638841629028320e+00,1.225027918815612793e+00,-1.395657539367675781e+00,-1.842909306287765503e-01,1.695090413093566895e+00,-1.026954174041748047e+00,2.356003671884536743e-01,1.310371279716491699e+00,-3.690620958805084229e-01,-3.661884963512420654e-01,-9.056421518325805664e-01,-6.749981641769409180e-01,4.917913079261779785e-01,1.116621270775794983e-01,-1.130558013916015625e+00,-6.039890050888061523e-01,1.364171266555786133e+00,2.915799319744110107e-01,-2.689051926136016846e-01,-1.139767646789550781e+00], 
[-5.971841514110565186e-02,-4.761638343334197998e-01,4.356950521469116211e-01,-2.002421468496322632e-01,-1.332062184810638428e-01,1.159576654434204102e+00,-1.284611463546752930e+00,1.678481221199035645e+00,-2.288321703672409058e-01,2.979312837123870850e-01,-5.772209167480468750e-01,1.523624062538146973e+00,-1.004847049713134766e+00,9.531534314155578613e-01,2.758104801177978516e-01,-1.129093050956726074e+00,8.181349039077758789e-01,3.663451597094535828e-02,1.743597388267517090e+00,-7.807526737451553345e-02,-5.065284371376037598e-01,4.749117195606231689e-01,-1.509342074394226074e+00,-1.287008523941040039e+00,1.966358661651611328e+00,5.462512373924255371e-01,-1.406587511301040649e-01,2.689583301544189453e-01,-1.774290055036544800e-01,-1.296455502510070801e+00,9.343785047531127930e-01,-1.138463973999023438e+00,4.878230392932891846e-01,9.053863286972045898e-01,-1.116675138473510742e+00,1.618416905403137207e+00,2.328500598669052124e-01,-7.623634338378906250e-01,-1.254516988992691040e-01,6.275811791419982910e-01,-6.836158633232116699e-01,-1.288200736045837402e+00,-8.081474155187606812e-02,-4.982246086001396179e-02,-2.493182867765426636e-01,2.336936816573143005e-02,-2.374809980392456055e-01,-5.362555384635925293e-01], 
[-2.020278573036193848e-01,-6.568687409162521362e-02,-5.507173538208007812e-01,-5.333126783370971680e-01,4.117356240749359131e-01,-4.234439730644226074e-01,7.236804813146591187e-02,8.475040197372436523e-01,-9.587838649749755859e-01,1.514374762773513794e-01,1.136253595352172852e+00,2.505342662334442139e-01,9.687843918800354004e-01,-3.425743877887725830e-01,2.180043756961822510e-01,-2.221308350563049316e-01,8.183828592300415039e-01,1.677451014518737793e+00,-1.196573972702026367e+00,-5.375201106071472168e-01,-3.216617405414581299e-01,2.635321378707885742e+00,-1.126178726553916931e-01,1.255911350250244141e+00,2.354023933410644531e+00,-2.432079017162322998e-01,-5.276182293891906738e-01,-2.241705179214477539e+00,-2.059963941574096680e+00,-8.604991436004638672e-01,-2.340532541275024414e+00,-7.787506580352783203e-01,-9.806937575340270996e-01,-2.273060381412506104e-01,4.213893786072731018e-02,1.599299013614654541e-01,9.135885834693908691e-01,4.985212087631225586e-01,-1.764091998338699341e-01,-6.061559915542602539e-02,6.290938258171081543e-01,5.936161279678344727e-01,-2.253904193639755249e-01,-1.014562100172042847e-01,-6.163067221641540527e-01,8.915224075317382812e-01,-2.008010745048522949e-01,3.287578523159027100e-01], 
[8.783959746360778809e-01,-1.297753691673278809e+00,3.054320439696311951e-02,7.292236089706420898e-01,4.427608251571655273e-01,-1.076445698738098145e+00,-7.148210406303405762e-01,-1.229741692543029785e+00,4.926630556583404541e-01,8.072228431701660156e-01,8.114566206932067871e-01,-2.129449844360351562e+00,-1.627279639244079590e+00,-2.572104930877685547e-01,3.989839553833007812e-02,-1.228928327560424805e+00,1.074609756469726562e+00,3.118561804294586182e-01,1.715852975845336914e+00,1.245982289314270020e+00,8.606585264205932617e-01,6.371878385543823242e-01,-3.403256833553314209e-01,-1.424303412437438965e+00,4.960089325904846191e-01,6.095127463340759277e-01,-7.817386388778686523e-01,9.388458728790283203e-01,-7.818238735198974609e-01,1.229679957032203674e-02,-3.640756607055664062e-01,-1.116449713706970215e+00,2.288946866989135742e+00,-3.291352987289428711e-01,8.774346709251403809e-01,-1.334406971931457520e+00,-2.979497015476226807e-01,-3.157010078430175781e-01,5.269924998283386230e-01,3.730039000511169434e-01,7.249338030815124512e-01,1.226357936859130859e+00,-2.079573571681976318e-01,-2.090241014957427979e-01,-2.858183979988098145e-01,9.884177893400192261e-02,-1.969695240259170532e-01,-1.951646357774734497e-01] ];

var tdnn1Bias = [3.471738100051879883e-01, 
-2.152900934219360352e+00, 
-4.984462261199951172e+00, 
-1.693641662597656250e+00, 
-2.693674087524414062e+00, 
-1.592693328857421875e+00, 
-2.226783037185668945e+00, 
1.703680276870727539e+00];

var tdnn2Weight = [[-2.518510103225708008e+00,-1.269004106521606445e+00,-1.167806684970855713e-01,-1.489641427993774414e+00,-1.420427680015563965e+00,-3.349974870681762695e+00,2.439129590988159180e+00,1.024893879890441895e+00,2.882792651653289795e-01,-3.333858251571655273e+00,-1.480436801910400391e+00,1.614195823669433594e+00,3.468437194824218750e+00,2.306246995925903320e+00,9.902876019477844238e-01,-1.065966010093688965e+00,-6.285744905471801758e-01,-6.177406311035156250e-01,4.445912316441535950e-02,-1.643864393234252930e+00,-2.417411804199218750e+00,1.040309071540832520e+00,-8.770182728767395020e-02,1.745612621307373047e+00], 
[1.940568923950195312e+00,-1.239314317703247070e+00,-3.366604089736938477e+00,-2.307534933090209961e+00,6.595717668533325195e-01,1.692628502845764160e+00,-6.560643672943115234e+00,-4.485341548919677734e+00,-1.156704664230346680e+00,-3.251230239868164062e+00,6.559439748525619507e-02,2.007281780242919922e+00,-2.764745712280273438e+00,-1.771603822708129883e+00,1.028169512748718262e+00,3.288066387176513672e+00,2.977946519851684570e+00,1.475748419761657715e+00,-4.241210460662841797e+00,1.205540895462036133e+00,1.451887965202331543e+00,4.149601459503173828e+00,2.312139272689819336e+00,1.711303949356079102e+00], 
[2.447964906692504883e+00,1.387485861778259277e+00,1.902276754379272461e+00,4.142662525177001953e+00,1.628898978233337402e-01,7.513161301612854004e-01,3.268991112709045410e-01,1.119639515876770020e+00,-5.802182555198669434e-01,3.942100048065185547e+00,3.285771846771240234e+00,1.234983801841735840e+00,1.288206338882446289e+00,-2.241858720779418945e+00,-7.950381040573120117e-01,-3.425814151763916016e+00,1.409118056297302246e+00,1.542488932609558105e-01,1.337586164474487305e+00,-5.267814993858337402e-01,6.840823292732238770e-01,-4.622075557708740234e+00,-4.883073568344116211e-01,6.890448927879333496e-01] ];

var tdnn2Bias = [3.478255748748779297e+00, 
-1.725674420595169067e-01, 
-1.151237130165100098e+00];

var linearWeight = [[1.074077248573303223e+00,1.392075181007385254e+00,6.201183795928955078e-01,3.511539697647094727e-01,8.132041692733764648e-01,1.455018818378448486e-01,-4.315594434738159180e-01,1.898505724966526031e-02,-1.159036636352539062e+00,-7.952469348907470703e+00,-4.975874423980712891e+00,-1.979544043540954590e+00,-5.364618897438049316e-01,-5.045258998870849609e-01,2.258702218532562256e-01,7.087447047233581543e-01,5.463308095932006836e-01,-1.529615044593811035e+00,5.161601066589355469e+00,3.021363973617553711e+00,1.213116168975830078e+00,-5.129529833793640137e-01,-5.966424345970153809e-01,2.872000336647033691e-01,1.059065461158752441e+00,7.294355630874633789e-01,1.975052654743194580e-01], 
[-2.407886505126953125e+00,-3.074234724044799805e+00,-2.810477256774902344e+00,-2.466505765914916992e+00,-2.528295755386352539e+00,-2.996613502502441406e+00,-2.599432706832885742e+00,-1.974507093429565430e+00,-3.053683280944824219e+00,6.499930381774902344e+00,5.578430652618408203e+00,1.673927903175354004e+00,-2.943126857280731201e-01,4.108847379684448242e-01,-4.167074263095855713e-01,-4.942923188209533691e-01,-7.886945009231567383e-01,-1.282857418060302734e+00,3.299722909927368164e+00,4.194572925567626953e+00,3.387089490890502930e+00,1.126910448074340820e+00,-6.184095889329910278e-02,-1.159652948379516602e+00,-1.714333057403564453e+00,-1.835815787315368652e+00,-2.081422328948974609e+00], 
[1.902660727500915527e+00,2.162961006164550781e+00,1.690208792686462402e+00,1.197375178337097168e+00,2.086381196975708008e+00,2.414008140563964844e+00,3.180685520172119141e+00,1.917525887489318848e+00,2.655402660369873047e+00,4.337633609771728516e+00,2.501040697097778320e+00,-7.814369797706604004e-01,-2.639186382293701172e-01,-3.859965801239013672e-01,-3.858404606580734253e-02,4.153030812740325928e-01,-1.529323756694793701e-01,5.243257284164428711e-01,-5.695884704589843750e+00,-5.660903930664062500e+00,-4.387213706970214844e+00,-1.625827908515930176e+00,-8.942835927009582520e-01,4.885446131229400635e-01,8.465791940689086914e-01,1.167461872100830078e+00,1.000181511044502258e-01] ];

var linearBias = [-9.676914662122726440e-02, 
-3.758729934692382812e+00, 
-3.513656854629516602e+00];


var mean = tf.tensor([[-9.834073638916015625e+01,-9.467967224121093750e+01,-9.097612762451171875e+01,-8.872144317626953125e+01,-8.832453918457031250e+01,-8.857080841064453125e+01,-8.886937713623046875e+01,-8.908764648437500000e+01,-8.916129302978515625e+01,-8.910598754882812500e+01,-8.906036376953125000e+01,-8.907570648193359375e+01,-8.924309539794921875e+01,-8.953280639648437500e+01,-8.995555877685546875e+01], 
[-8.968573760986328125e+01,-8.522927856445312500e+01,-8.094073486328125000e+01,-7.853623962402343750e+01,-7.809577941894531250e+01,-7.835391235351562500e+01,-7.865912628173828125e+01,-7.881750488281250000e+01,-7.894991302490234375e+01,-7.905796813964843750e+01,-7.925639343261718750e+01,-7.952619934082031250e+01,-7.985858917236328125e+01,-8.033209991455078125e+01,-8.089719390869140625e+01], 
[-8.290941619873046875e+01,-7.147406005859375000e+01,-6.453691864013671875e+01,-6.364815902709960938e+01,-6.577024078369140625e+01,-6.781539154052734375e+01,-6.936362457275390625e+01,-7.034153747558593750e+01,-7.078047180175781250e+01,-7.089164733886718750e+01,-7.102323913574218750e+01,-7.141117858886718750e+01,-7.196652221679687500e+01,-7.268180084228515625e+01,-7.349151611328125000e+01], 
[-8.232730102539062500e+01,-6.971695709228515625e+01,-6.287386703491210938e+01,-6.220500946044921875e+01,-6.472646331787109375e+01,-6.683102416992187500e+01,-6.766710662841796875e+01,-6.773543548583984375e+01,-6.747277069091796875e+01,-6.719174194335937500e+01,-6.701634979248046875e+01,-6.720811462402343750e+01,-6.769382476806640625e+01,-6.846230316162109375e+01,-6.933826446533203125e+01], 
[-8.050564575195312500e+01,-6.685235595703125000e+01,-5.963843154907226562e+01,-5.970509338378906250e+01,-6.378102111816406250e+01,-6.789672851562500000e+01,-7.101147460937500000e+01,-7.341233825683593750e+01,-7.524754333496093750e+01,-7.647462463378906250e+01,-7.724894714355468750e+01,-7.790497589111328125e+01,-7.849483489990234375e+01,-7.904644775390625000e+01,-7.951602172851562500e+01], 
[-8.604211425781250000e+01,-7.421808624267578125e+01,-6.718395996093750000e+01,-6.526863098144531250e+01,-6.606819915771484375e+01,-6.683048248291015625e+01,-6.710073089599609375e+01,-6.727001190185546875e+01,-6.756170654296875000e+01,-6.804146575927734375e+01,-6.874726867675781250e+01,-6.975218963623046875e+01,-7.102619934082031250e+01,-7.244647216796875000e+01,-7.394275665283203125e+01], 
[-9.261211395263671875e+01,-8.320113372802734375e+01,-7.678759002685546875e+01,-7.377975463867187500e+01,-7.301753234863281250e+01,-7.279985809326171875e+01,-7.277914428710937500e+01,-7.295330810546875000e+01,-7.342945098876953125e+01,-7.418997955322265625e+01,-7.526549530029296875e+01,-7.659633636474609375e+01,-7.812998199462890625e+01,-7.970515441894531250e+01,-8.131430053710937500e+01], 
[-9.650337219238281250e+01,-8.782748413085937500e+01,-8.209489440917968750e+01,-7.958860015869140625e+01,-7.893780517578125000e+01,-7.860870361328125000e+01,-7.819715881347656250e+01,-7.791481781005859375e+01,-7.789022064208984375e+01,-7.823751068115234375e+01,-7.898456573486328125e+01,-8.003705596923828125e+01,-8.139150238037109375e+01,-8.290792846679687500e+01,-8.459101104736328125e+01], 
[-9.952897644042968750e+01,-9.045652770996093750e+01,-8.480198669433593750e+01,-8.251036071777343750e+01,-8.211484527587890625e+01,-8.202400207519531250e+01,-8.182521820068359375e+01,-8.166130065917968750e+01,-8.171184539794921875e+01,-8.208481597900390625e+01,-8.281997680664062500e+01,-8.393077087402343750e+01,-8.534098815917968750e+01,-8.698021697998046875e+01,-8.874081420898437500e+01], 
[-9.826299285888671875e+01,-8.906965637207031250e+01,-8.377667999267578125e+01,-8.208675384521484375e+01,-8.244112396240234375e+01,-8.334397125244140625e+01,-8.417045593261718750e+01,-8.489224243164062500e+01,-8.558216094970703125e+01,-8.629850006103515625e+01,-8.713790130615234375e+01,-8.815467834472656250e+01,-8.938905334472656250e+01,-9.076322174072265625e+01,-9.223538970947265625e+01], 
[-9.921736145019531250e+01,-9.041437530517578125e+01,-8.543949127197265625e+01,-8.389564514160156250e+01,-8.440865325927734375e+01,-8.547147369384765625e+01,-8.653605651855468750e+01,-8.753901672363281250e+01,-8.855783081054687500e+01,-8.972490692138671875e+01,-9.103137969970703125e+01,-9.245464324951171875e+01,-9.392869567871093750e+01,-9.532179260253906250e+01,-9.663571929931640625e+01], 
[-9.870212554931640625e+01,-8.982271575927734375e+01,-8.489520263671875000e+01,-8.357271575927734375e+01,-8.423397064208984375e+01,-8.556067657470703125e+01,-8.672514343261718750e+01,-8.774649810791015625e+01,-8.857476043701171875e+01,-8.941772460937500000e+01,-9.060350799560546875e+01,-9.208855438232421875e+01,-9.381151580810546875e+01,-9.536106872558593750e+01,-9.674058532714843750e+01], 
[-9.884315490722656250e+01,-9.001406860351562500e+01,-8.521851348876953125e+01,-8.381631469726562500e+01,-8.450514221191406250e+01,-8.572808074951171875e+01,-8.680886840820312500e+01,-8.774739837646484375e+01,-8.864445495605468750e+01,-8.970964050292968750e+01,-9.098725891113281250e+01,-9.254854583740234375e+01,-9.432002258300781250e+01,-9.584693145751953125e+01,-9.704862976074218750e+01], 
[-9.923785400390625000e+01,-9.053671264648437500e+01,-8.561060333251953125e+01,-8.398265838623046875e+01,-8.442037963867187500e+01,-8.547395324707031250e+01,-8.648566436767578125e+01,-8.736285400390625000e+01,-8.835590362548828125e+01,-8.961416625976562500e+01,-9.108869171142578125e+01,-9.264690399169921875e+01,-9.408502197265625000e+01,-9.548601531982421875e+01,-9.678428649902343750e+01], 
[-9.920845031738281250e+01,-9.053803253173828125e+01,-8.559152221679687500e+01,-8.419754791259765625e+01,-8.480209350585937500e+01,-8.596495819091796875e+01,-8.682846832275390625e+01,-8.760768890380859375e+01,-8.852790069580078125e+01,-8.986950683593750000e+01,-9.137473297119140625e+01,-9.299758148193359375e+01,-9.450713348388671875e+01,-9.587976837158203125e+01,-9.711996459960937500e+01], 
[-9.914311981201171875e+01,-9.031877136230468750e+01,-8.546905517578125000e+01,-8.406415557861328125e+01,-8.470528411865234375e+01,-8.566583251953125000e+01,-8.648978424072265625e+01,-8.729183197021484375e+01,-8.822299194335937500e+01,-8.958621978759765625e+01,-9.113597106933593750e+01,-9.279380798339843750e+01,-9.429965209960937500e+01,-9.563889312744140625e+01,-9.687388610839843750e+01] ]);

var std = tf.tensor([[7.479398250579833984e+00,7.401870727539062500e+00,6.203837394714355469e+00,5.033008575439453125e+00,4.868764877319335938e+00,4.963184356689453125e+00,5.008862018585205078e+00,5.070852756500244141e+00,5.045858383178710938e+00,5.051112174987792969e+00,5.022284507751464844e+00,5.057617664337158203e+00,5.129976749420166016e+00,5.394038677215576172e+00,5.629917621612548828e+00], 
[7.074994564056396484e+00,7.570242881774902344e+00,6.472349643707275391e+00,5.137835025787353516e+00,4.738092422485351562e+00,4.596708774566650391e+00,4.611170768737792969e+00,4.642896652221679688e+00,4.694878578186035156e+00,4.734127044677734375e+00,4.763488769531250000e+00,4.829124927520751953e+00,4.951845169067382812e+00,5.157651901245117188e+00,5.408147335052490234e+00], 
[1.463958930969238281e+01,1.240227413177490234e+01,5.892759799957275391e+00,4.217080116271972656e+00,4.286021709442138672e+00,4.188833713531494141e+00,4.150907516479492188e+00,4.179685115814208984e+00,4.367692470550537109e+00,4.678973197937011719e+00,5.045841217041015625e+00,5.356560230255126953e+00,5.741289615631103516e+00,6.181211471557617188e+00,6.700946807861328125e+00], 
[1.720596885681152344e+01,1.308765602111816406e+01,6.872563362121582031e+00,5.668050289154052734e+00,5.644539833068847656e+00,5.090422630310058594e+00,5.027577877044677734e+00,5.353775024414062500e+00,5.753953933715820312e+00,6.057823657989501953e+00,6.184817314147949219e+00,6.283990859985351562e+00,6.534621238708496094e+00,6.891439914703369141e+00,7.262851715087890625e+00], 
[1.844884109497070312e+01,1.430970001220703125e+01,7.988345623016357422e+00,7.465167522430419922e+00,7.705735683441162109e+00,7.033782005310058594e+00,6.640152454376220703e+00,6.404170036315917969e+00,6.159350872039794922e+00,6.084422588348388672e+00,6.184617996215820312e+00,6.437457084655761719e+00,6.725802898406982422e+00,7.032291889190673828e+00,7.218702316284179688e+00], 
[1.649908447265625000e+01,1.423079395294189453e+01,1.066641139984130859e+01,9.883018493652343750e+00,9.687964439392089844e+00,9.523882865905761719e+00,9.588342666625976562e+00,9.646582603454589844e+00,9.710326194763183594e+00,9.649827957153320312e+00,9.522344589233398438e+00,9.415298461914062500e+00,9.438427925109863281e+00,9.571247100830078125e+00,9.758803367614746094e+00], 
[1.379905891418457031e+01,1.267264461517333984e+01,1.068278789520263672e+01,1.034205722808837891e+01,1.037771320343017578e+01,1.031117630004882812e+01,1.023954582214355469e+01,1.014323043823242188e+01,1.006101131439208984e+01,9.976151466369628906e+00,9.989934921264648438e+00,1.009439659118652344e+01,1.039041423797607422e+01,1.074236583709716797e+01,1.115349292755126953e+01], 
[1.376746273040771484e+01,1.267968559265136719e+01,1.092276287078857422e+01,1.063735103607177734e+01,1.052228927612304688e+01,1.037837886810302734e+01,1.033164501190185547e+01,1.035567283630371094e+01,1.046720981597900391e+01,1.059615707397460938e+01,1.081445503234863281e+01,1.110164356231689453e+01,1.145198917388916016e+01,1.184148025512695312e+01,1.216407966613769531e+01], 
[1.512942028045654297e+01,1.311781978607177734e+01,1.073406982421875000e+01,1.010389900207519531e+01,1.010584735870361328e+01,1.025309848785400391e+01,1.051229953765869141e+01,1.079814434051513672e+01,1.094585609436035156e+01,1.093083572387695312e+01,1.083273506164550781e+01,1.079072475433349609e+01,1.098095321655273438e+01,1.127469253540039062e+01,1.160867977142333984e+01], 
[1.572132396697998047e+01,1.267202472686767578e+01,9.939941406250000000e+00,9.672226905822753906e+00,1.008779239654541016e+01,1.024006462097167969e+01,1.017066192626953125e+01,1.001999664306640625e+01,9.921348571777343750e+00,1.000352096557617188e+01,1.031886577606201172e+01,1.083086776733398438e+01,1.148753547668457031e+01,1.209942245483398438e+01,1.257775020599365234e+01], 
[1.604116249084472656e+01,1.294763660430908203e+01,1.007960414886474609e+01,9.456366539001464844e+00,9.493108749389648438e+00,9.346612930297851562e+00,9.130207061767578125e+00,8.981265068054199219e+00,8.904871940612792969e+00,9.141846656799316406e+00,9.713722229003906250e+00,1.048760890960693359e+01,1.135029315948486328e+01,1.202777671813964844e+01,1.255563354492187500e+01], 
[1.721290588378906250e+01,1.450263118743896484e+01,1.248037624359130859e+01,1.247123241424560547e+01,1.250364017486572266e+01,1.218308353424072266e+01,1.183467483520507812e+01,1.178417396545410156e+01,1.174357223510742188e+01,1.185961246490478516e+01,1.224759483337402344e+01,1.273691368103027344e+01,1.336123371124267578e+01,1.385931491851806641e+01,1.418946647644042969e+01], 
[1.712116813659667969e+01,1.441304206848144531e+01,1.244399833679199219e+01,1.229644966125488281e+01,1.241188526153564453e+01,1.210734176635742188e+01,1.160920715332031250e+01,1.156396961212158203e+01,1.172550487518310547e+01,1.209602451324462891e+01,1.242723274230957031e+01,1.278109359741210938e+01,1.334296226501464844e+01,1.373294925689697266e+01,1.407487010955810547e+01], 
[1.710954856872558594e+01,1.453566265106201172e+01,1.237934684753417969e+01,1.203310585021972656e+01,1.218884563446044922e+01,1.215546512603759766e+01,1.196045780181884766e+01,1.185975551605224609e+01,1.184969139099121094e+01,1.204739093780517578e+01,1.236100482940673828e+01,1.263835144042968750e+01,1.297956371307373047e+01,1.338030338287353516e+01,1.390806865692138672e+01], 
[1.704825210571289062e+01,1.426736259460449219e+01,1.189234924316406250e+01,1.183017730712890625e+01,1.227486038208007812e+01,1.244192695617675781e+01,1.226996803283691406e+01,1.208356761932373047e+01,1.193753528594970703e+01,1.210484600067138672e+01,1.235878658294677734e+01,1.266006469726562500e+01,1.306141185760498047e+01,1.344664859771728516e+01,1.383849334716796875e+01], 
[1.723992729187011719e+01,1.430276298522949219e+01,1.201265716552734375e+01,1.193276596069335938e+01,1.247105216979980469e+01,1.259671306610107422e+01,1.256118011474609375e+01,1.242243576049804688e+01,1.220871925354003906e+01,1.215003299713134766e+01,1.234600353240966797e+01,1.275378799438476562e+01,1.332350444793701172e+01,1.369323158264160156e+01,1.399093723297119141e+01] ]);


tdnn1Weight = tf.tensor(tdnn1Weight);
tdnn1Weight = tf.reshape(tdnn1Weight, [8, 16, 3]);
// tf.console.log(tdnn1Weight);
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
// tf.console.log(tdnn2Weight);
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


// console.log('javascript step by step breakdown');
// var sample = 
// [[ 8.7812e-01,  1.2407e+00,  1.9310e+00,  2.3613e+00,  2.4478e+00,
//   2.5102e+00,  2.7010e+00,  2.9250e+00,  3.3115e+00,  2.8883e+00,
//   3.0813e+00,  2.1545e+00,  1.3816e+00,  1.5416e+00,  1.8604e+00],
// [ 1.8652e+00,  1.4726e+00,  2.6435e+00,  3.7489e+00,  3.7657e+00,
//   3.2418e+00,  2.4165e+00,  1.0889e+00,  1.4067e+00,  1.8899e+00,
//   2.3693e+00,  2.9142e+00,  3.2461e+00,  2.8083e+00,  3.1665e+00],
// [ 1.3574e+00,  1.7353e+00,  3.8234e+00,  5.0406e+00,  4.2522e+00,
//   3.3547e+00,  3.5532e+00,  4.0928e+00,  3.9618e+00,  3.6157e+00,
//   2.9181e+00,  2.9209e+00,  2.7056e+00,  2.8000e+00,  2.3998e+00],
// [ 4.0066e-01,  6.1020e-01,  1.9244e+00,  2.9866e+00,  3.0672e+00,
//   1.3023e+00,  9.1902e-01,  1.2536e+00,  1.4667e+00,  1.4401e+00,
//   1.3121e+00,  9.2619e-01,  9.2601e-01,  9.6007e-01,  9.7398e-01],
// [-9.0212e-02,  3.7420e-02,  4.9942e-01,  7.1241e-01,  5.6281e-01,
//  -2.7889e-01, -1.4440e-01, -6.6381e-02,  6.0634e-01,  7.3103e-01,
//   8.0484e-01,  2.6849e-01,  6.0238e-01,  8.3613e-01,  9.6668e-01],
// [-6.1845e-01, -7.7900e-01, -7.4408e-01, -7.7804e-01, -1.0413e+00,
//  -1.4268e+00, -1.5361e+00, -1.5700e+00, -1.5159e+00, -1.3418e+00,
//  -1.3639e+00, -1.2808e+00, -1.0468e+00, -9.5382e-01, -9.8102e-01],
// [-8.6749e-01, -8.2288e-01, -5.1317e-01, -4.6100e-01, -6.9465e-01,
//  -1.0605e+00, -1.2896e+00, -1.4374e+00, -1.8615e+00, -1.9797e+00,
//  -2.1927e+00, -1.8470e+00, -1.4646e+00, -1.3648e+00, -1.1374e+00],
// [-2.9438e-02, -4.6516e-01, -8.5577e-01, -8.8318e-01, -1.0025e+00,
//  -1.1590e+00, -1.3115e+00, -1.4651e+00, -1.7459e+00, -1.7367e+00,
//  -1.8046e+00, -1.6919e+00, -1.3035e+00, -1.2533e+00, -1.3831e+00],
// [ 3.2388e-01, -1.4720e-01, -4.1796e-01, -4.4554e-01, -4.4375e-01,
//  -9.9012e-01, -1.0712e+00, -1.3131e+00, -1.5597e+00, -1.7018e+00,
//  -1.8082e+00, -2.1604e+00, -2.1171e+00, -2.0593e+00, -1.9726e+00],
// [ 8.3993e-01,  6.2816e-01,  5.5966e-02, -2.3964e-01, -4.9239e-01,
//  -8.6283e-01, -1.1368e+00, -1.2375e+00, -1.9451e+00, -2.0071e+00,
//  -2.0477e+00, -2.1091e+00, -1.8438e+00, -1.7548e+00, -1.6571e+00],
// [ 6.0948e-01,  4.9543e-01,  3.2334e-01,  2.3464e-02, -1.2787e-01,
//  -5.5318e-01, -1.0055e+00, -1.2005e+00, -1.9419e+00, -1.9556e+00,
//  -1.9278e+00, -1.4373e+00, -1.2745e+00, -1.1026e+00, -1.2511e+00],
// [-6.2640e-02,  1.9750e-02,  2.3122e-01,  2.4234e-01,  2.0163e-01,
//  -1.0186e+00, -1.3089e+00, -2.2547e+00, -2.0092e+00, -2.0985e+00,
//  -1.6536e+00, -1.3844e+00, -2.2603e+00, -1.0283e+00, -9.6722e-01],
// [-4.7499e-01, -2.3094e-01, -2.5168e-02, -3.7515e-01, -2.3262e-01,
//  -4.4852e-01, -9.6714e-01, -1.2189e+00, -1.5344e+00, -1.3715e+00,
//  -1.0982e+00, -1.3101e+00, -1.7709e+00, -1.0274e+00, -8.5002e-01],
// [-5.6167e-01, -3.7874e-01, -3.0856e-02, -2.2589e-01, -2.2172e-01,
//  -4.6192e-01, -7.4103e-01, -1.0906e+00, -1.8598e+00, -1.4889e+00,
//  -1.1250e+00, -1.2155e+00, -1.4877e+00, -1.0687e+00, -1.0387e+00],
// [ 2.7706e-01,  3.8372e-02,  9.8518e-02,  1.0265e-01, -4.6239e-02,
//  -1.9408e+00, -8.4140e-01, -9.2782e-01, -1.5681e+00, -1.6586e+00,
//  -1.6603e+00, -1.1394e+00, -2.1488e+00, -9.1470e-01, -1.0487e+00],
// [ 3.4053e-01,  8.4809e-02, -3.5706e-01, -8.4459e-02, -1.5111e-04,
//  -1.1552e+00, -1.3269e+00, -9.2315e-01, -1.3716e+00, -1.5203e+00,
//  -1.4949e+00, -7.5268e-01, -6.6642e-01, -6.7303e-01, -8.7690e-01]];
// sample = tf.expandDims(sample, 0)
// sample_transposed = tf.transpose(sample, [0, 2, 1])

// var tdnn1out = model.layers[0].apply(sample_transposed)
// var sigmoid1out = model.layers[1].apply(tdnn1out)
// var tdnn2out = model.layers[2].apply(sigmoid1out)
// var sigmoid2out = model.layers[3].apply(tdnn2out)
// var flattened = model.layers[4].apply(sigmoid2out)
// var densed = model.layers[5].apply(flattened)

// console.log('=========== debugging output results, layer by layer ===========')
// console.log()
// console.log('input shape', sample_transposed.shape)
// tf.print(sample_transposed);
// console.log()
// console.log('tdnn1out shape', tdnn1out.shape)
// tf.print(tdnn1out);
// console.log()
// console.log('sigmoid1out shape', sigmoid1out.shape)
// tf.print(sigmoid1out);
// console.log()
// console.log('tdnn2out shape', tdnn2out.shape)
// tf.print(tdnn2out);
// console.log()
// console.log('sigmoid2out shape', sigmoid2out.shape)
// tf.print(sigmoid2out);
// console.log()
// console.log()
// console.log('flattened shape', flattened.shape)
// tf.print(flattened);
// console.log()
// console.log()
// console.log('densed shape', densed.shape)
// tf.print(densed)
// console.log()
// console.log('model actual output:')
// console.log(model.predict(sample_transposed).arraySync())