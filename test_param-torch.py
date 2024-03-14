import torch
import torch.nn as nn
from tdnn import TDNN as TDNNLayer
# from samples_from_web.sample1 import sample1
# from samples_from_web.sample2 import sample2
# from samples_from_web.sample3 import sample3

# sample = sample1
logged = True
normed = True


class TDNNv1(nn.Module):
    '''
    TDNN Model from Paper, consisting of the following layers:
    - tdnn 1: 16 in channels, 8 out channels, 15 samples, window of 3
    - sigmoid after tdnn
    - tdnn 2: 8 in channels, 3 out channels, 13 samples, window of 5
    - sigmoid after tdnn
    - flatten: 9 frequencies, 3 out channels, flattens to (27, ) array
    - linear: 27 inputs, 4 outputs
    '''

    def __init__(self):
        super(TDNNv1, self).__init__()

        self.tdnn1 = TDNNLayer(16, 8, [-1,0,1])
        self.sigmoid1 = nn.Sigmoid()
        self.tdnn2 = TDNNLayer(8, 3, [-2,0,2])
        self.sigmoid2 = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(27, 3)
        self.network = nn.Sequential(
            self.tdnn1,
            self.sigmoid1,
            self.tdnn2,
            self.sigmoid2,
            self.flatten,
            self.linear,
        )

    def forward(self, x):
        out = self.network(x)
        return out

model = TDNNv1()

tdnn1Weight = [[1.615776866674423218e-01,7.173244357109069824e-01,-3.600513339042663574e-01,-4.317452013492584229e-01,3.240724205970764160e-01,3.481120467185974121e-01,1.264496922492980957e+00,-4.993483126163482666e-01,-1.716446876525878906e+00,4.578845798969268799e-01,1.375126242637634277e-01,4.878286719322204590e-01,-5.463902950286865234e-01,3.269463777542114258e-01,1.437870383262634277e+00,-3.207933008670806885e-01,-7.860606908798217773e-01,-3.932978212833404541e-01,1.208361148834228516e+00,2.353659570217132568e-01,-1.731502294540405273e+00,1.235476493835449219e+00,9.039855003356933594e-02,-1.526147127151489258e+00,3.403080701828002930e-01,1.592718511819839478e-01,-2.563360691070556641e+00,-2.206249952316284180e+00,-7.624908089637756348e-01,1.445505380630493164e+00,-9.339705109596252441e-01,7.275081276893615723e-01,1.503642678260803223e+00,6.353123486042022705e-02,8.030033111572265625e-01,1.321001052856445312e-01,4.116766154766082764e-01,-1.856408268213272095e-01,-6.734496951103210449e-01,-5.280317068099975586e-01,2.325821816921234131e-01,-2.927263975143432617e-01,-4.784611463546752930e-01,-8.270282149314880371e-01,-2.972904443740844727e-01,-1.007838129997253418e+00,6.109450757503509521e-02,5.200375318527221680e-01], 
[-3.452513832598924637e-03,-1.336529970169067383e+00,1.308408498764038086e+00,-3.600727319717407227e-01,-1.175992935895919800e-02,5.393897294998168945e-01,2.284594327211380005e-01,2.087873965501785278e-02,-1.792060136795043945e-01,1.588798642158508301e+00,-6.548770666122436523e-01,-3.042885959148406982e-01,-1.409039139747619629e+00,7.898665666580200195e-01,-5.386318564414978027e-01,1.924874544143676758e+00,1.953338645398616791e-02,-2.282663106918334961e+00,1.266022771596908569e-01,7.581895589828491211e-01,-3.125071525573730469e-01,1.693768948316574097e-01,9.127893447875976562e-01,2.597072720527648926e-01,-3.332494735717773438e+00,5.335280895233154297e-01,1.814848899841308594e+00,-2.814588308334350586e+00,2.285535097122192383e+00,3.076463699340820312e+00,-9.333670735359191895e-01,-2.458352297544479370e-01,6.821522116661071777e-01,1.597627997398376465e+00,-4.668914154171943665e-02,-1.084846377372741699e+00,8.561133742332458496e-01,-5.740663409233093262e-01,-1.318393647670745850e-01,3.352861702442169189e-01,-4.147407412528991699e-01,6.692518591880798340e-01,8.568079471588134766e-01,7.292349338531494141e-01,8.745934963226318359e-01,1.184138059616088867e+00,-6.561943292617797852e-01,-6.916561126708984375e-01], 
[-9.521145373582839966e-02,-3.556398907676339149e-03,1.609508246183395386e-01,2.984281256794929504e-02,-2.202465385198593140e-01,1.361910402774810791e-01,3.772200345993041992e-01,-8.964907526969909668e-01,-7.566453814506530762e-01,1.897730082273483276e-01,1.050844714045524597e-01,4.187459051609039307e-01,4.666791260242462158e-01,1.352828383445739746e+00,-1.116645216941833496e+00,1.446906805038452148e+00,1.975263476371765137e+00,3.778667449951171875e-01,9.931989312171936035e-01,2.135787248611450195e+00,-6.897564530372619629e-01,-2.619797468185424805e+00,4.272846281528472900e-01,-1.666705250740051270e+00,-4.296140372753143311e-01,1.042730093002319336e+00,-4.769817292690277100e-01,2.500162124633789062e-01,-3.316564485430717468e-02,-5.073943138122558594e-01,-1.062359571456909180e+00,-1.575413048267364502e-01,1.595913767814636230e+00,-1.887445151805877686e-01,4.813833534717559814e-01,-2.668620049953460693e-01,1.545606702566146851e-01,-1.183445081114768982e-01,6.691344976425170898e-01,-3.985083699226379395e-01,-9.626450948417186737e-03,-5.791527405381202698e-02,-1.748253941535949707e+00,4.977345764636993408e-01,1.406136900186538696e-01,-5.424109697341918945e-01,5.862668156623840332e-01,4.138270020484924316e-01], 
[-4.003115892410278320e-01,-4.915787652134895325e-02,4.963667094707489014e-01,1.228542685508728027e+00,5.958993434906005859e-01,-1.038601636886596680e+00,1.377211689949035645e+00,1.317198425531387329e-01,-1.696777462959289551e+00,3.302029967308044434e-01,2.122111916542053223e-01,-9.592766165733337402e-01,1.158550605177879333e-01,-1.370083987712860107e-01,8.381529450416564941e-01,7.704634070396423340e-01,4.387546181678771973e-01,1.982323169708251953e+00,-9.867287240922451019e-03,-1.541227340698242188e+00,4.897983670234680176e-01,-2.086551904678344727e+00,9.413799643516540527e-02,6.608991622924804688e-01,-1.114586472511291504e+00,3.201376795768737793e-01,6.620292663574218750e-01,1.921589732170104980e+00,3.258171677589416504e-01,8.978007435798645020e-01,-2.343396186828613281e+00,-1.781494498252868652e+00,-1.351436972618103027e-01,-2.623797357082366943e-01,6.619822978973388672e-02,-5.733544230461120605e-01,-1.265271455049514771e-01,-1.048604771494865417e-01,5.792835354804992676e-01,-7.128936648368835449e-01,1.378682374954223633e+00,-6.105946302413940430e-01,1.088364839553833008e+00,-1.324118971824645996e+00,-1.623441576957702637e-01,-8.682973682880401611e-02,-1.606066077947616577e-01,-1.905658282339572906e-02], 
[1.615658164024353027e+00,-1.153308153152465820e+00,5.100411549210548401e-02,4.145072698593139648e-01,9.433984160423278809e-01,-1.478326469659805298e-01,7.164877653121948242e-01,-7.471098303794860840e-01,-1.114450931549072266e+00,1.337672829627990723e+00,-6.986818909645080566e-01,1.271316885948181152e+00,7.793805003166198730e-01,3.348873257637023926e-01,1.484649538993835449e+00,-1.057517647743225098e+00,-2.664629518985748291e-01,-9.116870164871215820e-01,2.144865274429321289e+00,-5.753396153450012207e-01,-1.923464179039001465e+00,1.785091638565063477e+00,-1.761712282896041870e-01,8.973836302757263184e-01,-1.430236250162124634e-01,-9.725254774093627930e-01,1.143494844436645508e+00,-1.342817306518554688e+00,-1.426638841629028320e+00,1.225027918815612793e+00,-1.395657539367675781e+00,-1.842909306287765503e-01,1.695090413093566895e+00,-1.026954174041748047e+00,2.356003671884536743e-01,1.310371279716491699e+00,-3.690620958805084229e-01,-3.661884963512420654e-01,-9.056421518325805664e-01,-6.749981641769409180e-01,4.917913079261779785e-01,1.116621270775794983e-01,-1.130558013916015625e+00,-6.039890050888061523e-01,1.364171266555786133e+00,2.915799319744110107e-01,-2.689051926136016846e-01,-1.139767646789550781e+00], 
[-5.971841514110565186e-02,-4.761638343334197998e-01,4.356950521469116211e-01,-2.002421468496322632e-01,-1.332062184810638428e-01,1.159576654434204102e+00,-1.284611463546752930e+00,1.678481221199035645e+00,-2.288321703672409058e-01,2.979312837123870850e-01,-5.772209167480468750e-01,1.523624062538146973e+00,-1.004847049713134766e+00,9.531534314155578613e-01,2.758104801177978516e-01,-1.129093050956726074e+00,8.181349039077758789e-01,3.663451597094535828e-02,1.743597388267517090e+00,-7.807526737451553345e-02,-5.065284371376037598e-01,4.749117195606231689e-01,-1.509342074394226074e+00,-1.287008523941040039e+00,1.966358661651611328e+00,5.462512373924255371e-01,-1.406587511301040649e-01,2.689583301544189453e-01,-1.774290055036544800e-01,-1.296455502510070801e+00,9.343785047531127930e-01,-1.138463973999023438e+00,4.878230392932891846e-01,9.053863286972045898e-01,-1.116675138473510742e+00,1.618416905403137207e+00,2.328500598669052124e-01,-7.623634338378906250e-01,-1.254516988992691040e-01,6.275811791419982910e-01,-6.836158633232116699e-01,-1.288200736045837402e+00,-8.081474155187606812e-02,-4.982246086001396179e-02,-2.493182867765426636e-01,2.336936816573143005e-02,-2.374809980392456055e-01,-5.362555384635925293e-01], 
[-2.020278573036193848e-01,-6.568687409162521362e-02,-5.507173538208007812e-01,-5.333126783370971680e-01,4.117356240749359131e-01,-4.234439730644226074e-01,7.236804813146591187e-02,8.475040197372436523e-01,-9.587838649749755859e-01,1.514374762773513794e-01,1.136253595352172852e+00,2.505342662334442139e-01,9.687843918800354004e-01,-3.425743877887725830e-01,2.180043756961822510e-01,-2.221308350563049316e-01,8.183828592300415039e-01,1.677451014518737793e+00,-1.196573972702026367e+00,-5.375201106071472168e-01,-3.216617405414581299e-01,2.635321378707885742e+00,-1.126178726553916931e-01,1.255911350250244141e+00,2.354023933410644531e+00,-2.432079017162322998e-01,-5.276182293891906738e-01,-2.241705179214477539e+00,-2.059963941574096680e+00,-8.604991436004638672e-01,-2.340532541275024414e+00,-7.787506580352783203e-01,-9.806937575340270996e-01,-2.273060381412506104e-01,4.213893786072731018e-02,1.599299013614654541e-01,9.135885834693908691e-01,4.985212087631225586e-01,-1.764091998338699341e-01,-6.061559915542602539e-02,6.290938258171081543e-01,5.936161279678344727e-01,-2.253904193639755249e-01,-1.014562100172042847e-01,-6.163067221641540527e-01,8.915224075317382812e-01,-2.008010745048522949e-01,3.287578523159027100e-01], 
[8.783959746360778809e-01,-1.297753691673278809e+00,3.054320439696311951e-02,7.292236089706420898e-01,4.427608251571655273e-01,-1.076445698738098145e+00,-7.148210406303405762e-01,-1.229741692543029785e+00,4.926630556583404541e-01,8.072228431701660156e-01,8.114566206932067871e-01,-2.129449844360351562e+00,-1.627279639244079590e+00,-2.572104930877685547e-01,3.989839553833007812e-02,-1.228928327560424805e+00,1.074609756469726562e+00,3.118561804294586182e-01,1.715852975845336914e+00,1.245982289314270020e+00,8.606585264205932617e-01,6.371878385543823242e-01,-3.403256833553314209e-01,-1.424303412437438965e+00,4.960089325904846191e-01,6.095127463340759277e-01,-7.817386388778686523e-01,9.388458728790283203e-01,-7.818238735198974609e-01,1.229679957032203674e-02,-3.640756607055664062e-01,-1.116449713706970215e+00,2.288946866989135742e+00,-3.291352987289428711e-01,8.774346709251403809e-01,-1.334406971931457520e+00,-2.979497015476226807e-01,-3.157010078430175781e-01,5.269924998283386230e-01,3.730039000511169434e-01,7.249338030815124512e-01,1.226357936859130859e+00,-2.079573571681976318e-01,-2.090241014957427979e-01,-2.858183979988098145e-01,9.884177893400192261e-02,-1.969695240259170532e-01,-1.951646357774734497e-01] ]
                            

tdnn1Bias = [3.471738100051879883e-01, 
-2.152900934219360352e+00, 
-4.984462261199951172e+00, 
-1.693641662597656250e+00, 
-2.693674087524414062e+00, 
-1.592693328857421875e+00, 
-2.226783037185668945e+00, 
1.703680276870727539e+00]

tdnn2Weight = [[-2.518510103225708008e+00,-1.269004106521606445e+00,-1.167806684970855713e-01,-1.489641427993774414e+00,-1.420427680015563965e+00,-3.349974870681762695e+00,2.439129590988159180e+00,1.024893879890441895e+00,2.882792651653289795e-01,-3.333858251571655273e+00,-1.480436801910400391e+00,1.614195823669433594e+00,3.468437194824218750e+00,2.306246995925903320e+00,9.902876019477844238e-01,-1.065966010093688965e+00,-6.285744905471801758e-01,-6.177406311035156250e-01,4.445912316441535950e-02,-1.643864393234252930e+00,-2.417411804199218750e+00,1.040309071540832520e+00,-8.770182728767395020e-02,1.745612621307373047e+00], 
[1.940568923950195312e+00,-1.239314317703247070e+00,-3.366604089736938477e+00,-2.307534933090209961e+00,6.595717668533325195e-01,1.692628502845764160e+00,-6.560643672943115234e+00,-4.485341548919677734e+00,-1.156704664230346680e+00,-3.251230239868164062e+00,6.559439748525619507e-02,2.007281780242919922e+00,-2.764745712280273438e+00,-1.771603822708129883e+00,1.028169512748718262e+00,3.288066387176513672e+00,2.977946519851684570e+00,1.475748419761657715e+00,-4.241210460662841797e+00,1.205540895462036133e+00,1.451887965202331543e+00,4.149601459503173828e+00,2.312139272689819336e+00,1.711303949356079102e+00], 
[2.447964906692504883e+00,1.387485861778259277e+00,1.902276754379272461e+00,4.142662525177001953e+00,1.628898978233337402e-01,7.513161301612854004e-01,3.268991112709045410e-01,1.119639515876770020e+00,-5.802182555198669434e-01,3.942100048065185547e+00,3.285771846771240234e+00,1.234983801841735840e+00,1.288206338882446289e+00,-2.241858720779418945e+00,-7.950381040573120117e-01,-3.425814151763916016e+00,1.409118056297302246e+00,1.542488932609558105e-01,1.337586164474487305e+00,-5.267814993858337402e-01,6.840823292732238770e-01,-4.622075557708740234e+00,-4.883073568344116211e-01,6.890448927879333496e-01] ]

tdnn2Bias = [3.478255748748779297e+00, 
-1.725674420595169067e-01, 
-1.151237130165100098e+00]

linearWeight = [[1.074077248573303223e+00,1.392075181007385254e+00,6.201183795928955078e-01,3.511539697647094727e-01,8.132041692733764648e-01,1.455018818378448486e-01,-4.315594434738159180e-01,1.898505724966526031e-02,-1.159036636352539062e+00,-7.952469348907470703e+00,-4.975874423980712891e+00,-1.979544043540954590e+00,-5.364618897438049316e-01,-5.045258998870849609e-01,2.258702218532562256e-01,7.087447047233581543e-01,5.463308095932006836e-01,-1.529615044593811035e+00,5.161601066589355469e+00,3.021363973617553711e+00,1.213116168975830078e+00,-5.129529833793640137e-01,-5.966424345970153809e-01,2.872000336647033691e-01,1.059065461158752441e+00,7.294355630874633789e-01,1.975052654743194580e-01], 
[-2.407886505126953125e+00,-3.074234724044799805e+00,-2.810477256774902344e+00,-2.466505765914916992e+00,-2.528295755386352539e+00,-2.996613502502441406e+00,-2.599432706832885742e+00,-1.974507093429565430e+00,-3.053683280944824219e+00,6.499930381774902344e+00,5.578430652618408203e+00,1.673927903175354004e+00,-2.943126857280731201e-01,4.108847379684448242e-01,-4.167074263095855713e-01,-4.942923188209533691e-01,-7.886945009231567383e-01,-1.282857418060302734e+00,3.299722909927368164e+00,4.194572925567626953e+00,3.387089490890502930e+00,1.126910448074340820e+00,-6.184095889329910278e-02,-1.159652948379516602e+00,-1.714333057403564453e+00,-1.835815787315368652e+00,-2.081422328948974609e+00], 
[1.902660727500915527e+00,2.162961006164550781e+00,1.690208792686462402e+00,1.197375178337097168e+00,2.086381196975708008e+00,2.414008140563964844e+00,3.180685520172119141e+00,1.917525887489318848e+00,2.655402660369873047e+00,4.337633609771728516e+00,2.501040697097778320e+00,-7.814369797706604004e-01,-2.639186382293701172e-01,-3.859965801239013672e-01,-3.858404606580734253e-02,4.153030812740325928e-01,-1.529323756694793701e-01,5.243257284164428711e-01,-5.695884704589843750e+00,-5.660903930664062500e+00,-4.387213706970214844e+00,-1.625827908515930176e+00,-8.942835927009582520e-01,4.885446131229400635e-01,8.465791940689086914e-01,1.167461872100830078e+00,1.000181511044502258e-01] ]

linearBias = [-9.676914662122726440e-02, 
-3.758729934692382812e+00, 
-3.513656854629516602e+00]

mean = torch.Tensor([[-9.834073638916015625e+01,-9.467967224121093750e+01,-9.097612762451171875e+01,-8.872144317626953125e+01,-8.832453918457031250e+01,-8.857080841064453125e+01,-8.886937713623046875e+01,-8.908764648437500000e+01,-8.916129302978515625e+01,-8.910598754882812500e+01,-8.906036376953125000e+01,-8.907570648193359375e+01,-8.924309539794921875e+01,-8.953280639648437500e+01,-8.995555877685546875e+01], 
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
[-9.914311981201171875e+01,-9.031877136230468750e+01,-8.546905517578125000e+01,-8.406415557861328125e+01,-8.470528411865234375e+01,-8.566583251953125000e+01,-8.648978424072265625e+01,-8.729183197021484375e+01,-8.822299194335937500e+01,-8.958621978759765625e+01,-9.113597106933593750e+01,-9.279380798339843750e+01,-9.429965209960937500e+01,-9.563889312744140625e+01,-9.687388610839843750e+01] ])

std = torch.Tensor([[7.479398250579833984e+00,7.401870727539062500e+00,6.203837394714355469e+00,5.033008575439453125e+00,4.868764877319335938e+00,4.963184356689453125e+00,5.008862018585205078e+00,5.070852756500244141e+00,5.045858383178710938e+00,5.051112174987792969e+00,5.022284507751464844e+00,5.057617664337158203e+00,5.129976749420166016e+00,5.394038677215576172e+00,5.629917621612548828e+00], 
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
[1.723992729187011719e+01,1.430276298522949219e+01,1.201265716552734375e+01,1.193276596069335938e+01,1.247105216979980469e+01,1.259671306610107422e+01,1.256118011474609375e+01,1.242243576049804688e+01,1.220871925354003906e+01,1.215003299713134766e+01,1.234600353240966797e+01,1.275378799438476562e+01,1.332350444793701172e+01,1.369323158264160156e+01,1.399093723297119141e+01] ])

tdnn1Weight = torch.reshape(torch.Tensor(tdnn1Weight), (8, 16, 3))
tdnn2Weight = torch.reshape(torch.Tensor(tdnn2Weight), (3, 8, 3))

model_params = [tdnn1Weight,
                torch.Tensor(tdnn1Bias),
                tdnn2Weight,
                torch.Tensor(tdnn2Bias),
                torch.Tensor(linearWeight),
                torch.Tensor(linearBias)]


with torch.no_grad():
  for i, param in enumerate(model.parameters()):
    # print(param.shape, model_params[i].shape)
    param.data = model_params[i]

model.eval()

sample = [[2.8186209, 4.1880436, 1.7028786 , 0.9356455 , 1.7333127, 0.7655976 , 1.3677535 , 1.2597543 , 0.7152895 , 1.3696305, 0.7893806 , 0.7393811 , 1.1656697, 1.4181595, 1.370113 , 1.021428 ],
      [1.6247277, 3.313374 , 0.4845198 , -0.4744147, 1.093118 , 0.2478916 , 0.8028367 , 0.8271071 , 0.2943383 , 0.8851447, 0.4211222 , 0.0268637 , 0.7792557, 1.3986003, 1.4949967, 1.0192689],
      [1.3415051, 3.2128215, -0.1574803, -1.8991555, 1.0550702, -0.3287358, 0.3520168 , 0.4353173 , -0.1670815, 0.595939 , 0.0473881 , -0.3636043, 0.5171883, 1.2442881, 1.3776209, 0.8098662],
      [2.1077118, 3.3393581, -0.8952735, -1.8257405, 1.1155943, -0.0262802, -0.1632867, 0.0825851 , -0.4921537, 0.474443 , -0.1802165, -0.0084288, 0.3077805, 1.188006 , 1.392886 , 0.9522743],
      [2.1901228, 3.6353228, -0.5601382, -1.3870234, 1.6850165, 0.1939644 , -0.3557574, -0.0870388, -0.6408781, 0.4662603, -0.1828301, 0.0979994 , 0.1764059, 1.2135147, 1.4141926, 1.0122197],
      [1.7095711, 3.418889 , 0.6770742 , -1.1560045, 2.1770935, 0.5312408 , -0.3411216, -0.1969456, -0.6241139, 0.62739  , -0.0363381, 0.326984  , 0.5363609, 1.2943401, 1.4505746, 1.0183992],
      [2.2991278, 3.784822 , 1.8154761 , -0.5514582, 2.5509107, 0.488824  , -0.1188279, -0.254968 , -0.4058081, 0.7065035, 0.1377438 , 0.3413518 , 0.8703545, 1.2893503, 1.3996693, 0.8750871],
      [3.084312 , 5.0286999, 3.1321683 , 0.2703333 , 2.7562022, 0.2459742 , 0.0154269 , -0.1244354, -0.0957654, 0.7384533, 0.1284038 , 0.1048495 , 0.7424732, 1.0178401, 1.2933316, 0.6637709],
      [3.1141853, 5.001224 , 3.0978396 , 0.2058826 , 3.1637087, 0.2743991 , 0.0628785 , -0.1254594, -0.0898553, 0.8153346, 0.2439162 , 0.175741  , 0.8087493, 1.1025084, 1.3862386, 0.7516604],
      [3.1698682, 5.3061771, 3.2864885 , 0.1556477 , 3.1990709, 0.291018  , 0.0446974 , 0.0820465 , 0.0461309 , 0.977899 , 0.4165784 , 0.2648374 , 0.6848099, 0.8820407, 1.3988328, 0.815344 ],
      [3.4236906, 5.5108657, 3.2279699 , -0.1844518, 3.5191383, 0.417226  , 0.1353864 , 0.3781174 , 0.1722927 , 1.1786819, 0.5507333 , 0.3562169 , 0.6947966, 0.9063212, 1.4111433, 1.0703896],
      [3.2205691, 5.5490594, 3.376159  , -0.221753 , 3.5503144, 0.5608096 , 0.1745947 , 0.4341953 , 0.1763622 , 1.2029225, 0.6073037 , 0.5094419 , 0.9527916, 1.195676 , 1.480659 , 1.2164176],
      [2.5308855, 5.0642424, 3.2052758 , -0.4882325, 3.3590097, 0.7825704 , 0.4055093 , 0.4885808 , -0.0503858, 1.06092  , 0.5711997 , 0.9509562 , 1.2397072, 1.337636 , 1.4064087, 1.293214 ],
      [2.3210294, 4.5264215, 2.6788831 , 0.1505123 , 2.9231596, 0.8964877 , 0.4968213 , 0.508998  , 0.2166245 , 0.9496124, 0.745156  , 1.1559353 , 1.2294915, 1.1523483, 0.9623786, 1.3309622],
      [2.2988746, 4.4212556, 2.5919409 , 0.2634238 , 2.9127238, 1.0325845 , 0.6227809 , 0.633864  , 0.3620543 , 1.0305443, 0.818477  , 1.2262609 , 1.2850026, 1.2019671, 1.0247476, 1.3909124]]


data_tensor = torch.Tensor(sample)
# logged_data = torch.transpose(data_tensor, 0, 1)
logged_data = data_tensor
if (not logged):
		logged_data = torch.log(data_tensor).max(torch.tensor(-25))
normed_data = logged_data
if (not normed):
  normed_data = (logged_data - mean)/std
inputs = normed_data

inputs = torch.transpose(inputs, 0, 1)
curr_input = inputs.unsqueeze(0)
print(curr_input.shape)
outputs = model(curr_input)
_, predicted = torch.max(outputs.data, 1)

# print('outputs', outputs)
# print('predicted', predicted)

# inputs = torch.expand

tdnn1out = model.tdnn1(curr_input)
sigmoid1out = model.sigmoid1(tdnn1out)
tdnn2out = model.tdnn2(sigmoid1out)
sigmoid2out = model.sigmoid2(tdnn2out)
flattened = model.flatten(sigmoid2out)
densed = model.linear(flattened)

print('=========== debugging output results, layer by layer ===========')
print()
print('input shape', curr_input.shape)
print('input', curr_input)
print()
print('tdnn1out shape', tdnn1out.shape)
print('tdnn1out', tdnn1out)
print()
print()
print('sigmoid1out shape', sigmoid1out.shape)
print('sigmoid1out', sigmoid1out)
print()
print()
print('tdnn2out shape', tdnn2out.shape)
print('tdnn2out', tdnn2out)
print()
print()
print('sigmoid2out shape', sigmoid2out.shape)
print('sigmoid2out', sigmoid2out)
print()
print()
print('flattened shape', flattened.shape)
print('flattened', flattened)
print()
print()
print('densed shape', densed.shape)
print('densed (final output)', densed)
print()
print('model actual output:')
print(model(curr_input))
