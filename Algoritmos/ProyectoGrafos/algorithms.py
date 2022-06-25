from collections import OrderedDict
import q as q
from matplotlib.testing.jpl_units import m



############################################################################################################
#          QUEYRANNE V1 - CUTFUN
############################################################################################################

#Algoritmo de Queyranne para minimizar funciones submodulares simétricas

# F: función submodular.
# V: índice de lista o conjunto.
# Devuelve una solución óptima a min F(A) s.t. 0<|A|<n

c=[]
def queyranne(F, V):
    def Fnew(a):
        r = []
        for x in a:
            r += S[x - 1]
        return F(r)

    n = len(V)
    S = [[x] for x in V]
    s = []
    A = []
    inew = OrderedDict()
    for x in range(1, n + 1):
        inew[x] = x
    minimum = float("inf")
    position_of_min = 0
    for h in range(n - 1):
        # Encuentra un pendant pair
        [t, u] = pendentpair(Fnew, inew)

        # Da una solución candidata
        A.append(S[u - 1].copy())
        c.append(S[u - 1].copy())
        s.append(Fnew({u}))
        if s[-1] < minimum:
            minimum = s[-1]
            position_of_min = len(s) - 1
        S[t - 1] += S[u - 1]
        del inew[u]
        for x in range(len(S[u - 1])):
            S[u - 1][x] *= -1

    return A[position_of_min]


# Implementa la subrutina de búsqueda de pendentpair del algoritmo de Queyranne
# # (Queyranne '95)
# # F es la función submodular
# # V es una matriz de índices; (típicamente, 1:n)
def pendentpair(F, V):
    vstart = V.popitem(last=False)[0]
    vnew = vstart
    n = len(V)
    Wi = []
    used = [0] * n
    for i in range(n):
        vold = vnew
        Wi += [vold]
        # Now update the keys
        keys = [1e99] * n
        minimum = float("inf")
        counter = -1
        for j in V:
            counter += 1
            if used[counter]:
                continue
            Wi += [V[j]]
            keys[counter] = F(Wi) - F({V[j]})
            del Wi[-1]
            if keys[counter] < minimum:
                minimum = keys[counter]
                argmin_key = j
                argmin_position = counter
            vnew = argmin_key
            used[argmin_position] = 1
    V[vstart] = vstart
    V.move_to_end(vstart, last=False)

    return [vold, vnew]

#Función de corte que devuelve la suma de los pesos de las rutas entre dos componentes en
#un gráfico representado por una matriz. función submodular
def cutfun(graph):
    def f(component):
        if type(component) != type(set()):
            component = set(component)
        outside_component = set(range(1, len(graph) + 1)) - component
        return sum([sum([graph[x - 1][y - 1] for y in outside_component]) for x in component])

    return f




############################################################################################################
#          SPECTRAL CLUSTERING
############################################################################################################

#SPECTRAL CLUSTERING Information integration in large brain networks

#Esta función implementa la solución basada en agrupamiento espectral
#que encuentra el MIB se aplica el agrupamiento espectral a cada
#matriz de correlación transformada, y se calcula la información integrada
#en cada bipartición resultante. La conjetura de la función para el MIB de
#sus datos es la bipartición (entre estos) que minimiza normalizado
#información integrada.


import numpy
from numpy import linalg as LA
import numpy as np
import time
import sys

#graph= matriz de adyacencias de grafo
#k= número de cluster a calcular

def get_clusters(graph, k):
    num_nodes = graph.shape[0]
    # labels = np.random.randint(1, size=num_nodes)

    clusters = [list(range(graph.shape[0]))]

    for i in range(k - 1):

        # obtener el clúster más grande
        # tupla = max(enumerar(clústeres), clave=lambda x: len(x[1]))
        # max_len_cluster = tupla[1]
        # max_len_cluster_index = tupla[0]

        # para almacenar el índice del clúster que se dividirá en esta iteración
        old_cluster_index = 0
        # para almacenar el segundo valor propio más pequeño del grupo antiguo que se dividirá en esta iteración
        old_cluster_ss_eval = sys.maxsize
        new_clusters = [clusters[old_cluster_index]]
        # newclusterfound = False
        # crear nuevos clústeres a partir de un clúster grande anterior
        for cluster_index, cluster in enumerate(clusters):
            # matriz de adyacencia para nodos de clúster
            if len(cluster) <= 1:
                continue
            cluster_adj_mat = graph[:, cluster]
            cluster_adj_mat = cluster_adj_mat[cluster, :]
            possible_new_clusters, ss_eval = spectral_partition(cluster_adj_mat, cluster)
            # if ss_eval <= old_cluster_ss_eval and len(possible_new_clusters[0])>0 and len(possible_new_clusters[1])>0:
            if ss_eval <= old_cluster_ss_eval:
                old_cluster_ss_eval = ss_eval
                old_cluster_index = cluster_index
                new_clusters = possible_new_clusters

        del clusters[old_cluster_index]
        clusters += new_clusters

        # print("newclusters:",clusters)
        # print("creating cluster ", (i + 1))
        # print()

    # for i,v in enumerate(clusters):
    #    labels[v] = i
    clusters_dict = {}
    for i, v in enumerate(clusters):
        cluster_label = "cluster_" + str(i + 1)
        clusters_dict[cluster_label] = v
    return clusters_dict


def spectral_partition(adj_mat, nodeIds):
    # grado de todos los nodos del grafo
    # print("Calculando grado")
    degrees = np.sum(adj_mat, axis=1)

    # matriz laplaciana del grafo
    #print("matriz laplacianamatriz laplaciana")
    lap_mat = np.diag(degrees) - adj_mat

    #print("Calculating eval and evecs")
    st = time.time()
    e_vals, e_vecs = LA.eigh(lap_mat)
    print("matriz laplaciana:",lap_mat)
    print("evals:", e_vals)
    print("evec:", e_vecs)
    tt = time.time() - st
    #print("Time to calculate eval of shape:", adj_mat.shape[0], " is:", tt)

    # Como todos los valores propios de la matriz laplaciana son mayores o iguales a 0, se recorta cualquier valor negativo pequeño
    e_vals = e_vals.clip(0)

    #print("sorting eval")
    sorted_evals_index = np.argsort(e_vals)

    # ss_index para almacenar el índice del segundo valor propio más pequeño
    ss_index = 0
    for index in sorted_evals_index:
        if (e_vals[index] > 0):
            ss_index = index
            break

    # vector propio del segundo valor propio más pequeño
    ss_evec = e_vecs[:, ss_index]

    # second smallest eigen value
    ss_eval = e_vals[ss_index]

    clusters = getClusterFromEvec(ss_evec, nodeIds)
    print(clusters)
    if (len(clusters[0]) == 0 or len(clusters[1]) == 0):
        clusters = getClusterFromEvec(e_vecs[:, 1], nodeIds)
        ss_eval = e_vals[1]
        #print("inside:",clusters)
        #print("evec:",e_vecs)
    print("clusters recibidos:",nodeIds)
    print("clusters retornados:",clusters)
    return clusters, ss_eval


def getClusterFromEvec(ss_evec, nodeIds):
    # print("creating 2 clus")
    clusters = [[], []]
    for i, v in enumerate(ss_evec):
        if v >= 0:
            clusters[0].append(nodeIds[i])
        else:
            clusters[1].append(nodeIds[i])
    return clusters


def getComponentsFromEvec(ss_evec, nodeIds):
    # print("creating 2 clus")
    clusters = [[], []]
    for i, v in enumerate(ss_evec):
        if v == 0:
            clusters[0].append(nodeIds[i])
        else:
            clusters[1].append(nodeIds[i])
    return clusters


###########EJEMPLO

'''grafo = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]]
grafo = numpy.array(grafo)
print(get_clusters(grafo,3))'''


############################################################################################################
#          SSP
############################################################################################################

#Procedimiento submodular-supermodular de Narasimhan & Bilmes para minimizar la diferencia de dos funciones submodulares
#Este algoritmo está garantizado para converger a un óptimo local

#opt: estructura opcional de parámetros, que puede contener la tolerancia deseada para esta función.

# F: función submodular
# G: función submodular
# V: conjunto de índices, no una lista o matriz.


# Devuelve una solución óptima localmente al problema A = argmin_A F(A) - G(A)


from sympy import *

import random
import numpy
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from sympy import *


def ssp(F, G, V, opt={}):
    TOL = opt["ssp_tolerance"] if "ssp_tolerance" in opt else 1e-6

    N = len(V)
    pi = random.sample(V, len(V))
    bestVal = float("inf")
    A = set()
    Varray = numpy.array([[x] for x in V])
    while True:
        Hw = ssp_modular_approx(G, pi)

        FM = lambda x: F(x) - sum([Hw[y - 1] for y in x])
        A = min_norm_point(FM, Varray, {"minnorm_init": A}, 0, set)[0]
        curVal = FM(A)
        D = V - A
        D = random.sample(D, len(D))
        pi = list(A) + D
        if curVal < bestVal - TOL:
            bestVal = curVal
        else:
            break
    return A


def ssp_modular_approx(G, pi):
    H = [0] * max(pi)
    W = []
    oldVal = G(W)
    for i in range(len(pi)):
        W += [pi[i]]
        newVal = G(W)
        H[i] = newVal - oldVal
        oldVal = newVal
    return H


#Función de prueba de Iwata para probar algoritmos de minimización submodular
#f(X)=|X||Xc|−∑k∈X(5k−2n)

def Iwata(n):
    return lambda x: len(x) * (n - len(x)) - sum(x) * 5 + 2 * n * len(x)


# opt: estructura de opción de parámetros. referenciando:

# minnorm_init: conjetura inicial para la solución óptima
# minnorm_stopping_thresh: umbral de detención para la búsqueda
# minnorm_tolerance: tolerancia numérica
# minnorm_callback: rutina de devolución de llamada para visualizar soluciones intermedias

# Devuelve una solución óptima A, ligada a la suboptimalidad

# La siguiente variable almacena la función de tipo de clase que se usará más adelante
f = type(lambda x:x)

def min_norm_point(F, V, opt={}, display=0, return_type=list):
    n = len(V)
    Ainit = opt["minnorm_init"] if "minnorm_init" in opt else set()
    eps = opt["minnorm_stopping_thresh"] if "minnorm_stopping_thresh" in opt else 1e-10
    TOL = opt["minnorm_tolerance"] if "minnorm_tolerance" in opt else 1e-10

    # Paso 1: inicialice eligiendo un punto en el politopo.
    wA = charvector(V, Ainit)
    xw = polyhedrongreedy(F, V, wA)
    xhat = xw

    # Por ahora creo que heredar xhat está bien como cambiar xhat completamente a
    # otra matriz no debería afectar a S.
    S = xhat

    Abest = -1
    Fbest = numpy.inf
    c = 1
    while True:
        # Paso 2: Encuentra un vértice en el politopo base que minimice su
        # producto interno con xhat.
        squarednorm = sum(numpy.multiply(xhat, xhat))
        norm = numpy.sqrt(squarednorm)
        if norm < TOL:  # Snap to zero
            xhat = numpy.zeros((n, 1))
        # Obtener phat yendo desde xhat hacia el origen hasta que lleguemos al
        # límite del politopo base
        phat = polyhedrongreedy(F, V, -xhat)
        S = numpy.append(S, phat, axis=1)
        # Check current function value
        lessthanzero = xhat < 0
        s = V[lessthanzero]
        Fcur = F(s)
        if Fcur < Fbest:
            Fbest = Fcur
            Abest = s

        # Obtener límite de subóptima
        subopt = Fbest - sum(xhat[lessthanzero])
        if display:
            print("suboptimality bound: " + str(float(Fbest - subopt)) +
                  "<=min_A F(A)<=F(A_best)=" + str(float(Fbest)) +
                  "; delta<=" + str(float(subopt)))

        absolute = abs(sum(numpy.multiply(xhat, xhat - phat))[0])
        if absolute < TOL or subopt < eps:
            # Hemos terminado: xhat ya es el punto de norma más cercano
            if absolute < TOL:
                subopt = 0
            A = Abest
            break

        # Aquí hay un código solo para mostrar el estado actual
        if "minnorm_callback" in opt and type(opt["minnorm"]) == f:  # Do something
            # with current state
            opt["minnorm_callback"](Abest)

        [xhat, S] = min_norm_point_update(xhat, S, TOL)
    if display:
        print("suboptimality bound: " + str(float(Fbest - subopt)) +
              "<=min_A F(A)<=F(A_best)=" + str(float(Fbest)) +
              "; delta<=" + str(float(subopt)))
    # Convierte el valor devuelto en el tipo que queremos
    if return_type:
        A = return_type(A)
    return [A, subopt]


# Función de ayuda
def min_norm_point_update(xhat, S, TOL):
    while True:
        # Paso 3: Encuentre el punto de norma mínimo en el casco afín atravesado por S

        S0 = S[:, 1:] - S[:, [0] * (len(S[0]) - 1)]
        firstcolumn = S[:, [0]]
        y = firstcolumn - numpy.matmul(numpy.matmul(S0, numpy.linalg.pinv(S0)),
                                       firstcolumn)  # Now y in min norm

        # Obtener la representación de y en términos de S. Hacer cumplir la combinación afín
        # (es decir, suma (mu) == 1)
        pseudoinverse = numpy.linalg.pinv(numpy.append(S,
                                                       [[1] * len(S[0])], axis=0))
        mu = numpy.matmul(pseudoinverse, numpy.append(y, [[1]], axis=0))

        # y is written as positive convex combination of S <==> y in conv(S)
        if not numpy.size(mu[mu < -TOL]) and abs(sum(mu) - 1) < TOL:
            return [y, S]

        # Paso 4: Proyecte y de vuelta al politopo.

        # Obtener la representación de xhat en términos de S; hacer cumplir que nos hacemos afines
        # combinación (es decir, sum(L)==1)
        L = numpy.matmul(pseudoinverse, numpy.append(xhat, numpy.array([[1]]),
                                                     axis=0))

        # Ahora encuentre z en conv(S) que esté más cerca de y
        bounds = numpy.divide(L, L - mu)
        bounds = bounds[bounds > TOL]
        beta = min(bounds)
        z = (1 - beta) * xhat + beta * y
        gamma = (1 - beta) * L + beta * mu
        S = S[:, numpy.where(gamma > TOL)[0]]
        xhat = z


# F: función submodular que también toma una lista como entrada válida
# V: lista de índice que es una matriz numpy en la forma [[número],[número],...]
# w: vector de peso que es una matriz numpy en la misma forma que V
def polyhedrongreedy(F, V, w):
    n = len(V)
    v = [[w[x][0], V[x][0], x] for x in range(n)]
    v.sort(reverse=True)
    r = numpy.zeros((n, 1))
    f = set()
    s = F(f)
    for x in v:
        f.add(x[1])
        g = F(f)
        r[x[2]] = [g - s]
        s = g
    return r


# charvector indica si un elemento en la lista V está en el conjunto A, y True es
# convertido a 1, falso a 0
def charvector(V, A):
    r = []
    for x in V:
        r.append([1 if x[0] in A else 0])
    return numpy.array(r)



############################################################################################################
#          S_T MIN CUT
############################################################################################################


#Algoritmo de punto de norma mínima de Fujishige para minimizar funciones submodulares generales
#Encontrar el mínimo A de una función submodular tal que s en A y t no en A

#Encuentra la ganancia marginal en el valor de la función cuando la lista B integra la lista x.
#[número],...]
def residual(F,B):
    def f(x):
        if type(x) != type(set):
            x = set(x)
        return F(x.union(B)) - F(B)
    return f

#Encontrar el mínimo A de una función submodular tal que s en A y t no en A

#Devuelve A en mincut(F,V,s,t)
#F: función submodular
#V: conjunto de índices
#s: elemento en V a incluir
#t: elemento en V a excluir
#opt(opcional): opciones, dependiendo de
#minnorm_stopping_thresh: umbral para detener la minimización (1e-5)


def s_t_mincut(F,V,s,t,opt={}):
    if "minnorm_stopping_thresh" not in opt:
        opt["minnorm_stopping_thresh"] = 1e-5
    F2 = residual(F,{s})
    V2 = numpy.array([[x] for x in V - {s,t}])
    return  min_norm_point(F2,V2,opt,return_type=list)[0]+[s]


############################################################################################################
#          QUEYRANNE V2 - MUTUAL INFORMATION
############################################################################################################

from re import M

def mutualInformation (m1, m2):
    sum_mi = 0.0
    x_value_list = np.unique(m1)
    y_value_list = np.unique(m2)
    Px = np.array([ len(m1[m1==xval])/float(len(m1)) for xval in x_value_list ])
    Py = np.array([ len(m2[m2==yval])/float(len(m2)) for yval in y_value_list ])
    for i in range(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = m2[m1 == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(m2))  for yval in y_value_list])
        t = pxy[Py>0.]/Py[Py>0.] /Px[i]
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) )
    return sum_mi

def pendent_pair(Vprime, V, S, f, params=None):
    x = 0
    vnew = Vprime[x]
    n = len(Vprime)
    Wi = []
    used = np.zeros((n, 1))
    used[x] = 1
    for i in range(n - 1):
        vold = vnew
        Wi = Wi + S[vold]
        ## update keys
        keys = np.ones((n, 1)) * np.inf
        for j in range(n):
            if used[j]:
                continue
            keys[j] = f(Wi + S[Vprime[j]], V, params) - f(S[Vprime[j]], V, params)
        ## extract min
        argmin = np.argmin(keys)
        vnew = Vprime[argmin]
        used[argmin] = 1
        fval = np.min(keys)
    s = vold
    t = vnew
    return s, t, fval


def diff(A, B):
    m = np.amax(np.array([np.amax(A), np.amax(B)]))
    vals = np.zeros((m + 1, 1))
    vals[A] = 1
    vals[B] = 0
    idx = np.nonzero(vals)
    return idx[0]


def optimal_set(V, f, params=None):
    n = len(V)
    S = [[] for _ in range(n)]
    for i in range(n):
        S[i] = [V[i]]
    p = np.zeros((n - 1, 1))
    A = []
    idxs = range(n)
    for i in range(n - 1):
        ## find a pendant pair
        t, u, fval = pendent_pair(idxs, V, S, f, params)
        ## candidate solution
        A.append(S[u])
        p[i] = f(S[u], V, params)
        S[t] = [*S[t], *S[u]]
        idxs = diff(idxs, u)
        S[u] = []

    ## return minimum solution
    i = np.argmin(p)
    R = A[i]
    fval = p[i]
    ## make R look pretty
    notR = diff(V, R)
    ## R = sorted(R)
    ## notR = sorted(notR)
    if R[0] < notR[0]:
        R = (tuple(R), tuple(notR))
    else:
        R = (tuple(notR), tuple(R))
    return R, fval


def k_subset(s, k):
    if k == len(s):
        return (tuple([(x,) for x in s]),)
    k_subs = []
    for i in range(len(s)):
        partials = k_subset(s[0:i] + s[i + 1:len(s)], k)
        for partial in partials:
            for p in range(len(partial)):
                k_subs.append(partial[:p] + (partial[p] + (s[i],),) + partial[p + 1:])
    return k_subs

def uniq_subsets(s):
    u = set()
    for x in s:
        t = []
        for y in x:
            y = list(y)
            y.sort()
            t.append(tuple(y))
        t.sort()
        u.add(tuple(t))
    return u

def find_partitions(V, k):
    k_subs = k_subset(V, k)
    k_subs = uniq_subsets(k_subs)
    return k_subs

def intersection(a, b):
    return list(set(a) & set(b))

def union(a, b):
    return list(set(a) | set(b))


###EJEMPLO
'''m1 = np.array(
[[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1 ], [1, 0, 1, 1]]
)
m2 = np.array(
[[0,0,1,0],[1,0,1,1],[0,1,0,1],[1,0,1,1]]
)

V = [x for x in range(1, len(m1) + 1)]
f = lambda V,S, params: mutualInformation(m1,m2)

R, fval = optimal_set(V, f)

print(R)
print(fval)'''




#################################### RMCMC ##############################################################
def log_prob(x):
  return -0.5 * np.sum(x ** 2)


def proposal(x, stepsize):
  return np.random.uniform(low=x - 0.5 * stepsize,high=x + 0.5 * stepsize,size=x.shape)


def p_acc_MH(x_new, x_old, log_prob):
  return min(1, np.exp(log_prob(x_new) - log_prob(x_old)))


def sample_MH(x_old, log_prob, stepsize):
  x_new = proposal(x_old, stepsize)
  # here we determine whether we accept the new state or not:
  # we draw a random number uniformly from [0,1] and compare
  # it with the acceptance probability
  accept = np.random.random() < p_acc_MH(x_new, x_old, log_prob)
  if accept:
      return accept, x_new
  else:
      return accept, x_old


def build_MH_chain(init, stepsize, n_total, log_prob):

  n_accepted = 0
  chain = [init]

  for _ in range(n_total):
    accept, state = sample_MH(chain[-1], log_prob, stepsize)
    chain.append(state)
    n_accepted += accept

  acceptance_rate = n_accepted / float(n_total)

  return chain, acceptance_rate
