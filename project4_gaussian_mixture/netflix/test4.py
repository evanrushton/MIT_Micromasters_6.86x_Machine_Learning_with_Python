import numpy as np
import em
from common import GaussianMixture

X= np.array([[0.85794562, 0.84725174], [0.6235637 , 0.38438171], [0.29753461, 0.05671298], [0.        , 0.47766512], [0.        , 0.        ], [0.3927848 , 0.        ], [0.        , 0.64817187], [0.36824154, 0.        ], [0.        , 0.87008726], [0.47360805, 0.        ], [0.        , 0.        ], [0.        , 0.        ], [0.53737323, 0.75861562], [0.10590761, 0.        ], [0.18633234, 0.        ]])

K=6

Mu=np.array([[0.6235637,  0.38438171],[0.3927848 , 0.        ],[0.        , 0.        ],[0.        , 0.87008726],[0.36824154, 0.        ],[0.10590761, 0.        ]])

Var=np.array([0.16865269, 0.14023295, 0.1637321 , 0.3077471 , 0.13718238, 0.14220473])

P=np.array([0.1680912 , 0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])


text='Output:\nX_pred:\n[[0.85794562 0.84725174]\n[0.6235637  0.38438171]\n[0.29753461 0.05671298]\n[0.28510109 0.47766512]\n[0.23813851 0.18836861]\n[0.3927848  0.15145129]\n[0.28675542 0.64817187]\n[0.36824154 0.14957025]\n[0.2518095  0.87008726]\n[0.47360805 0.15911262]\n[0.23813851 0.18836861]\n[0.23813851 0.18836861]\n[0.53737323 0.75861562]\n[0.10590761 0.14322076]\n[0.18633234 0.14239418]]'


mix = GaussianMixture(mu=Mu, var=Var, p=P)

print(em.fill_matrix(X, mix))
print(text)
