import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def shoot2(x_vect, x, K, epsilon):
    return [x_vect[1], (K*x**2-epsilon) * x_vect[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
eps_start = 0.5
eps_list = [] 
eig_func_list = []
A = 1
K = 1 
L = 4
xp = [-L, L] 
xspan = np.linspace(-L, L, int((2 * L) / 0.1) + 1)
#x0 = [A, A*np.sqrt(K*L**2)] #initial conditions
for modes in range(1, 6):  # begin mode loop
    eps = eps_start  # initial value of eigenvalue beta
    deps = eps_start / 100  # default step size in beta
    for _ in range(1000):  # begin convergence loop for beta
        x0 = [A, A*np.sqrt(K*L**2-eps)] #initial conditions
        y = odeint(shoot2, x0, xspan, args=(K,eps)) 
        # y = RK45(shoot2, xp[0], x0, xp[1], args=(n0,beta)) 

        if abs(y[-1, 1] + np.sqrt(K*L**2-eps)*y[-1,0]) < tol:  # final condition
            #print(eps)  # write out eigenvalue 
            eps_list.append(eps)
            #print(_)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(K*L**2-eps)*y[-1,0]) > 0:
            eps += deps
        else:
            eps -= deps / 2
            deps /= 2

    eps_start = eps + 0.01  # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xspan)  # calculate the normalization
    #plt.figure(modes)
    #plt.plot(xspan, y[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes
    eig_func_list.append(abs(y[:, 0] / np.sqrt(norm)))

plt.show()
A1 = eig_func_list
A2 = eps_list
other_A1 = np.array([[0.0002560239, 0.0014526119, 0.0056578387, 0.0174247034, 0.0449716541],
                   [0.0003767047, 0.0020809, 0.0078740636, 0.0234942195, 0.0585268011],
                   [0.000551333, 0.0029647739, 0.0108973736, 0.0314957085, 0.0757119111],
                   [0.0008007364, 0.0041902727, 0.014954987, 0.041849882, 0.0970292819],
                   [0.0011523093, 0.0058654573, 0.020315603, 0.0550069971, 0.1228922931],
                   [0.0016424715, 0.0081265477, 0.0272920161, 0.0714286436, 0.153600245],
                   [0.0023190618, 0.0111444492, 0.036257165, 0.0916072532, 0.1892805368],
                   [0.003241625, 0.0151206884, 0.0476086454, 0.1159759062, 0.2298115288],
                   [0.0044852282, 0.0202874785, 0.0617600352, 0.1448402041, 0.274657391],
                   [0.0061460174, 0.0269273187, 0.0791314209, 0.1783735465, 0.3228458627],
                   [0.0083394849, 0.0353568208, 0.1001656283, 0.2165513091, 0.3729690276],
                   [0.0112001712, 0.0459112671, 0.1252139442, 0.2590555282, 0.4231878857],
                   [0.0148905387, 0.0589394455, 0.1545129308, 0.3051758982, 0.4708509415],
                   [0.0196051616, 0.0748155115, 0.1881798151, 0.3537980343, 0.5129286947],
                   [0.0255584824, 0.0939162573, 0.2261501245, 0.4034034724, 0.5462236828],
                   [0.0329837099, 0.1165507769, 0.2680936611, 0.4521034223, 0.567370481],
                   [0.0421321895, 0.1429574887, 0.3133538837, 0.4974561209, 0.5730716947],
                   [0.0532923187, 0.1732933981, 0.3609420811, 0.5365166271, 0.5602875603],
                   [0.066748047, 0.2075908014, 0.4095373724, 0.5663400061, 0.5266558951],
                   [0.0827669847, 0.2457071139, 0.457505297, 0.5840024481, 0.4712000967],
                   [0.1015998787, 0.287267838, 0.5028690571, 0.5866453676, 0.3943415422],
                   [0.1234644509, 0.3316642838, 0.5431138343, 0.571963188, 0.2980336869],
                   [0.1485539688, 0.3780535688, 0.5757109006, 0.5378720531, 0.1857960833],
                   [0.1769771853, 0.4253793063, 0.5981905133, 0.4834504694, 0.0627977424],
                   [0.2087383963, 0.4723089071, 0.6081419144, 0.4091755718, 0.0640997474],
                   [0.2437374405, 0.5170883871, 0.6033806625, 0.316923123, 0.1866595041],
                   [0.2817705107, 0.5578973787, 0.5822108816, 0.2098166609, 0.2966157104],
                   [0.3225753332, 0.592882894, 0.543281281, 0.0922681442, 0.3856492837],
                   [0.3656209099, 0.6201593246, 0.4862815384, 0.0301531876, 0.4466741136],
                   [0.4102127635, 0.6378088598, 0.4119765735, 0.1509404245, 0.4743214461],
                   [0.4555881252, 0.6441712053, 0.322206547, 0.2629954717, 0.4651991069],
                   [0.5009159349, 0.6377232792, 0.2197083292, 0.3593433387, 0.4188620387],
                   [0.545296841, 0.6172502656, 0.1081064218, 0.4333757844, 0.3385586099],
                   [0.587763201, 0.5821506241, 0.0081806275, 0.4799641685, 0.2303447957],
                   [0.6272790805, 0.5324360892, 0.1241665027, 0.4955738094, 0.1026270557],
                   [0.6627637351, 0.4687316713, 0.2343657015, 0.4785967409, 0.0339809566],
                   [0.6932991105, 0.3922513898, 0.3335347674, 0.4293166725, 0.1675731578],
                   [0.7180082325, 0.3047225312, 0.4165170616, 0.3506611897, 0.286376159],
                   [0.7361790581, 0.2084109234, 0.4790478806, 0.2481046626, 0.3798131261],
                   [0.7473005501, 0.1059049342, 0.5179425031, 0.1284758914, 0.4395463198],
                   [0.7510626773, 0.0000949786, 0.5311993627, 0.0001314671, 0.460149189],
                   [0.7473564147, 0.1058264817, 0.5180875331, 0.1288328848, 0.4394283187],
                   [0.7362737429, 0.2084442501, 0.4791048382, 0.2485367632, 0.3795051473],
                   [0.718106556, 0.3048982497, 0.4162939891, 0.351024686, 0.2860008156],
                   [0.6933841234, 0.3924862898, 0.3332144463, 0.429328219, 0.1671240127],
                   [0.6628996913, 0.4689464134, 0.2341722836, 0.4784803287, 0.0335066598],
                   [0.6275166014, 0.5325695105, 0.1240598543, 0.4956714839, 0.1030038584],
                   [0.5881395753, 0.5821993174, 0.0080991056, 0.4800672462, 0.2306503023],
                   [0.5457147152, 0.6172435784, 0.1082916198, 0.4331942057, 0.3388823025],
                   [0.501229503, 0.6377930221, 0.2199463794, 0.3589980489, 0.4190633024],
                   [0.4556914311, 0.6441572097, 0.322524115, 0.2636408303, 0.4656276668],
                   [0.4101089916, 0.6378121003, 0.4119976414, 0.1510411471, 0.4744076488],
                   [0.3654810512, 0.6201678166, 0.4863234306, 0.0301727827, 0.446781248],
                   [0.322782242, 0.592890349, 0.5433196447, 0.0923195558, 0.385761151],
                   [0.2819467037, 0.5578998157, 0.582245445, 0.2098555162, 0.2966920622],
                   [0.243855701, 0.5170917527, 0.6034423441, 0.3169533646, 0.1866982432],
                   [0.2084450732, 0.4723150456, 0.6082105885, 0.4092150841, 0.0641364246],
                   [0.1766987435, 0.4253897377, 0.5982620194, 0.4834712651, 0.0628350026],
                   [0.1486858382, 0.3780578714, 0.575783501, 0.5378823736, 0.1858329954],
                   [0.1235946817, 0.3316685687, 0.5431925579, 0.5719688977, 0.2980781259],
                   [0.1017101281, 0.2872691536, 0.5029394047, 0.586648617, 0.394375582],
                   [0.0828836478, 0.2457066872, 0.4575595773, 0.584019994, 0.4712488646],
                   [0.0668677244, 0.207591405, 0.4096193965, 0.5664161718, 0.5266896875],
                   [0.0534145861, 0.1732941951, 0.3609654055, 0.5365575929, 0.5603167116],
                   [0.0422574284, 0.1429583967, 0.3133903301, 0.4974712775, 0.573115394],
                   [0.0329892296, 0.1165518966, 0.2681511275, 0.4521397564, 0.567431538],
                   [0.0255584824, 0.0939162573, 0.2261501245, 0.4034034724, 0.5462236828],
                   [0.0196051616, 0.0748155115, 0.1881798151, 0.3537980343, 0.5129286947],
                   [0.0148905387, 0.0589394455, 0.1545129308, 0.3051758982, 0.4708509415],
                   [0.0112001712, 0.0459112671, 0.1252139442, 0.2590555282, 0.4231878857],
                   [0.0083394849, 0.0353568208, 0.1001656283, 0.2165513091, 0.3729690276],
                   [0.0061460174, 0.0269273187, 0.0791314209, 0.1783735465, 0.3228458627],
                   [0.0044852282, 0.0202874785, 0.0617600352, 0.1448402041, 0.274657391],
                   [0.003241625, 0.0151206884, 0.0476086454, 0.1159759062, 0.2298115288],
                   [0.0023190618, 0.0111444492, 0.036257165, 0.0916072532, 0.1892805368],
                   [0.0016424715, 0.0081265477, 0.0272920161, 0.0714286436, 0.153600245],
                   [0.0011523093, 0.0058654573, 0.020315603, 0.0550069971, 0.1228922931],
                   [0.0008007364, 0.0041902727, 0.014954987, 0.041849882, 0.0970292819],
                   [0.000551333, 0.0029647739, 0.0108973736, 0.0314957085, 0.0757119111],
                   [0.0003767047, 0.0020809, 0.0078740636, 0.0234942195, 0.0585268011],
                   [0.0002560239, 0.0014526119, 0.0056578387, 0.0174247034, 0.0449716541]]) 

error = A1-np.transpose(other_A1)
y_min = -0.0007
y_max = 0.0007
y_ticks = np.linspace(y_min, y_max, 3)
fig, axs = plt.subplots(5, 1, figsize=(10, 15))
for i in range(6,11):
    axs[i-6].plot(xspan, error[i-6])
    axs[i-6].set_ylim(y_min, y_max)  # Set the same y-axis limits for each subplot
    axs[i-6].set_yticks(y_ticks)
    axs[i-6].set_title(f'Plot {i-6+1}')  # Optionally set a title for each subplot
plt.subplots_adjust(hspace=0.5) 
plt.show()    
#print(error)
print('A2 = ' + str(A2))