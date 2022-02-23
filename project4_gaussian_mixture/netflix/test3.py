import numpy as np
import common
import em

X2=np.array([[0.85794562, 0.84725174], [0.623563, 0.38438171], [0.29753461, 0.05671298], [0.27265629, 0.47766512], [0.81216873, 0.47997717], [0.392784, 0.83607876], [0.33739616, 0.64817187], [0.36824154, 0.95715516], [0.14035078, 0.87008726], [0.47360805, 0.80091075], [0.52047748, 0.67887953], [0.72063265, 0.58201979], [0.53737323, 0.75861562], [0.10590761, 0.47360042], [0.18633234, 0.73691818]])

K2= 6

post2=np.array([[0.15765074,0.20544344,0.17314824,0.15652173,0.12169798,0.18553787], [0.1094766,  0.22310587,0.24109142,0.0959303,0.19807563,0.13232018],[0.22679645,0.36955206,0.02836173,0.03478709,0.00807236,0.33243031],[0.16670188,0.18637975,0.20964608,0.17120102,0.09886116,0.16721011],[0.04250305,0.22996176,0.05151538,0.33947585,0.18753121,0.14901275],[0.09799086,0.28677458,0.16895715,0.21054678,0.0069597,0.22877093],[0.16764519,0.16897033,0.25848053,0.18674186,0.09846462,0.11969746],[0.28655211,0.02473762,0.27387452,0.27546459,0.08641467,0.05295649],[0.11353057,0.13090863,0.20522811,0.15786368,0.35574052,0.03672849],[0.10510461,0.08116927,0.3286373,0.12745369,0.23464272,0.12299241],[0.09757735,0.06774952,0.40286261,0.08481828,0.1206645,0.22632773],[0.24899344,0.02944918,0.25413459,0.02914503,0.29614373,0.14213403],[0.35350682,0.21890411,0.26755234,0.01418274,0.10235276,0.04350123],[0.15555757,0.06236572,0.16703133,0.21760554,0.03369562,0.36374421],[0.1917808 ,0.08982788,0.17710673,0.03179658,0.19494387,0.31454414]])

m2_text='Output:\nMu: [[0.43216722 0.64675402]\n[0.46139681 0.57129172]\n[0.44658753 0.68978041]\n[0.44913747 0.66937822]\n[0.47080526 0.68008664]\n[0.40532311 0.57364425]]\nVar: [0.25 0.25 0.25 0.25 0.25 0.25]\nP: [0.1680912  0.15835331 0.21384187 0.14223565 0.14295074 0.17452722]\n'

X=np.array([[0.,         0.,         0.77342887, 0.        ],[0.,         0.,         0.,         0.       ],[0.,         0.,         0.,         0.        ],[0.,         0.,         0.,         0.        ],[0.,         0.17052019, 0.,         0.        ],[0.,         0.,         0.,         0.21405342],[0.,         0.,         0.,         0.        ],[0.80403009, 0.,         0.,         0.        ],[0.,         0.,         0.71304285, 0.        ],[0.,         0.72977067, 0.,         0.        ],[0.,         0.,         0.57494742, 0.0625674 ],[0.,         0.,         0.,         0.41426901],[0.,         0.,         0.,         0.07654668],[0.,         0.,         0.,         0.        ],[0.72218521, 0.,         0.,         0.        ],[0.,         0.,         0.,         0.70442173],[0.67079789, 0.,         0.11930711, 0.        ],[0.,         0.92018479, 0.,         0.56768024]])

K=5

post=np.array([[2.37321245e-01, 6.24768398e-02, 2.85000448e-01, 4.12790207e-01, 2.41126116e-03], [5.37343893e-02, 2.65318315e-01, 2.60129116e-01, 1.50279280e-01, 2.70538900e-01], [2.46752143e-01, 5.15381346e-01, 1.57848905e-01, 2.23889248e-02, 5.76286813e-02], [1.24579216e-01, 2.60106708e-01, 1.82963648e-01, 2.28589046e-01, 2.03761381e-01], [1.38475667e-01, 1.39385103e-01, 2.64472724e-01, 2.06148311e-01, 2.51518195e-01], [2.49384450e-02, 7.62343913e-02, 4.03628896e-01, 2.69128526e-04, 4.94929139e-01], [3.79081147e-01, 4.00488767e-02, 2.07631328e-01, 2.43246772e-01, 1.29991876e-01], [2.61325929e-01, 2.48802781e-01, 1.80996106e-01, 1.66269307e-01 , 1.42605877e-01], [1.57861365e-02, 3.24012824e-01, 4.19878416e-02, 3.26498495e-01, 2.91714704e-01], [1.37834462e-01, 3.27729784e-01, 5.63873128e-03, 2.37326561e-01, 2.91470462e-01], [2.34930507e-01, 4.96737073e-02, 5.60061599e-01, 6.04827802e-02, 9.48514057e-02], [1.65783970e-01, 3.05865626e-01, 2.67219157e-01, 2.30400904e-01, 3.07303435e-02], [2.17090655e-01, 2.57520117e-01, 1.37311793e-01, 3.56653377e-01, 3.14240576e-02], [4.14097231e-01, 1.95296570e-01, 2.35773572e-01, 4.70655768e-02, 1.07767051e-01], [2.20063714e-01, 2.40225601e-02, 2.09854231e-01, 2.42958675e-01, 3.03100820e-01], [4.88620563e-02, 1.04956493e-01, 2.74412170e-01, 3.77039823e-01, 1.94729459e-01], [3.07937657e-03, 4.18925441e-01, 1.18837458e-01, 3.43201180e-01, 1.15956544e-01], [2.89138542e-01, 3.28628534e-01, 8.03187600e-02, 5.47409960e-02, 2.47173169e-01]])
m_text='Output:\nMu: [[ 0.35217825 -0.32505856 -0.38330192 -0.02396785]\n[-0.23746422 -0.34201138 -0.08168301  0.37968361]\n[-0.46135517  0.61020124  0.58311781  0.27949025]\n[-0.35479128 -0.83597853  0.54926251  0.39205814]\n[-0.92122461 -0.12003181 -0.06234407  0.36979796]]\nVar: [0.56978562 0.55308421 0.25814221 0.57561002 0.78315595]\nP: [0.17849305 0.21913256 0.21522703 0.2059083  0.18123907]'

mix,_ = common.init(X, K)
print(mix)
print(em.mstep(X, post, mix))
print(m_text)
