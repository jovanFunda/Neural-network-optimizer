import random
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from functools import reduce

mut_rate = 0.2
pop_size = 10
iterations_over_one_program = 3
max_iter = 150
bytes_in_hromozom = 14
output_path = ""
parametar_selekcije = 3
random.seed(99)

x_r = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90,
    2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90,
    3.00, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90,
    4.00, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90]

y_k = [
    7.0000, 7.4942, 8.1770, 9.0482, 10.1080, 11.3562, 12.7929, 14.4181, 16.2319, 18.2341,
    20.4248, 22.8040, 25.3717, 28.1279, 31.0726, 34.2058, 37.5274, 41.0376, 44.7363, 48.6234,
    52.6991, 56.9633, 61.4159, 66.0571, 70.8867, 75.9049, 81.1115, 86.5066, 92.0903, 97.8624,
    103.8230, 109.9721, 116.3097, 122.8358, 129.5504, 136.4535, 143.5451, 150.8252, 158.2938, 165.9509,
    173.7964, 181.8305, 190.0531, 198.4641, 207.0637, 215.8518, 224.8283, 233.9933, 243.3469, 252.8889]

def calc_neuron(neurons, input_weights, input_values, bias):
    sum = bias
    for i in range(neurons):
        sum += input_weights[i] * input_values[i]
    return sum

L1N = 4
L2N = 4

layer1 = np.zeros((L1N, 1))
layer2 = np.zeros((L2N, L1N))
layer3 = np.zeros((1, L2N))

bias1 = np.zeros((L1N))
bias2 = np.zeros((L2N))
bias3 = np.zeros((1))

output0 = np.zeros((1))
output1 = np.zeros((L1N))
output2 = np.zeros((L2N))

def trosak(argumenti):
    y_r = y_k

    total_expected_args = 1 + L1N + L1N * L2N + L2N + L1N + L2N
    if len(argumenti) != total_expected_args: 
        print("Broj argumenata nije odgovarajuci, ocekivano", total_expected_args - 1, "realnih vrednosti. Trenutno ima", len(argumenti), "vrednosti")
        exit()

    ai = 1
    for i in range(L1N):
        layer1[i][0] = argumenti[ai]
        ai += 1

    for i in range(L2N):
      for j in range(L1N):
          layer2[i][j] = argumenti[ai]
          ai += 1

    for i in range(L2N): 
        layer3[0][i] = argumenti[ai]

    for i in range(L1N): 
        bias1[i] = argumenti[ai]
        ai += 1

    for i in range(L2N):
        bias2[i] = argumenti[ai]
        ai += 1

    bias3[0] = argumenti[ai]

    mse = 0

    for k in range(50):
        output0[0] = x_r[k]

        for i in range(L1N):
            output1[i] = calc_neuron(1, layer1[i], output0, bias1[i])
        
        for i in range(L2N):
            output2[i] = calc_neuron(L1N, layer2[i], output1, bias2[i])

        val = calc_neuron(L2N, layer3[0], output2, bias3[0])

        err = pow(y_r[k] - val, 2)
        mse += err

    return (mse / 50)

def ukrsti(h1, h2):
    r = random.randrange(1, len(h1)-1)
    h3 = np.append(h1[:r], h2[r:])
    h4 = np.append(h2[:r], h1[r:])
    return h3, h4

def turnir(pop):
    najbolji = None
    najbolji_f = None
    z = []
    for i in range(parametar_selekcije):
        z.append(pop[math.floor(random.random()*len(pop)-1)])
    for e in z:
        ff = trosak(e)
        if najbolji is None:
            najbolji_f = ff
            najbolji = e
        if ff < najbolji_f:
            najbolji_f = ff
            najbolji = e
    return najbolji

def mutiraj(hromozom, verovatnoca):
    h1 = np.array([])
    i = 0
    while i < len(hromozom):
        if i % (int(1 / verovatnoca)) == 0 and i < (len(hromozom) - int((1 / verovatnoca))):
            if random.random() <= verovatnoca:
                temp = i + int(1/verovatnoca)
                while i < temp:
                    if hromozom[i] == 1:
                        h1 = np.append(h1, 0)
                    else:
                        h1 = np.append(h1, 1)
                    i += 1
            else:
                h1 = np.append(h1, hromozom[i])      
                i += 1
        else:
            h1 = np.append(h1, hromozom[i])
            i += 1
    return h1

def p_norm(number):
    return round((number + 3) / 6, 2)

def kodiraj_broj(broj):
    bits = np.zeros((bytes_in_hromozom))
    for m in range(bytes_in_hromozom):
        ukupna_suma = 0
        j = 0
        while j < m: 
            ukupna_suma += bits[j] * 2 ** (-j)
            j += 1
        bits[m] = math.ceil(p_norm(broj) - 2**(-m) - ukupna_suma)
    return bits

def dekodiraj_broj(broj_u_bitovima):
    ukupna_suma = 0
    for m in range(bytes_in_hromozom):
        ukupna_suma += broj_u_bitovima[m] * 2 ** (-m) + 2 ** (-bytes_in_hromozom+1)
    return round(ukupna_suma * 6 - 3, 2)

def read_file():
    global mut_rate
    global pop_size
    global iterations_over_one_program
    global max_iter
    global bytes_in_hromozom
    global output_path
    seed = 0
    params_file = open("Params.txt",  "r", encoding="utf8")
    lines = params_file.readlines()
    params_file.close()

    for line in lines:
        if line.startswith("mut_rate = "):
            mut_rate = line.split(" = ")[1]
            mut_rate = float(mut_rate[:len(mut_rate) - 1])
        if line.startswith("pop_size = "):
            pop_size = line.split(" = ")[1]
            pop_size = int(pop_size[:len(pop_size) - 1])
        if line.startswith("output_path = "):
            output_path = line.split(" = ")[1]
            output_path = output_path[:len(output_path) - 1]
        if line.startswith("max_iter = "):
            max_iter = line.split(" = ")[1]
            max_iter = int(max_iter[:len(max_iter) - 1])
        if line.startswith("bytes_in_hromozom = "):
            bytes_in_hromozom = line.split(" = ")[1]
            bytes_in_hromozom = int(bytes_in_hromozom[:len(bytes_in_hromozom) - 1])
        if line.startswith("iterations_over_one_program = "):
            iterations_over_one_program = line.split(" = ")[1]
            iterations_over_one_program = int(iterations_over_one_program[:len(iterations_over_one_program) - 1])
        if line.startswith("selection_parameter = "):
            parametar_selekcije = line.split(" = ")[1]
            parametar_selekcije = int(parametar_selekcije[:len(parametar_selekcije) - 1])
        if line.startswith("random seed = "):
            seed = random.seed(int(line.split(" = ")[1][:len(line.split(" = ")[1]) - 1]))
    return seed

def random_population(pop_size):
    ret_list = []
    for i in range(pop_size):
        ret_list.append([round(random.uniform(-3, 3), 2) for i in range(33)])
    return ret_list

def hromozom_to_bytes(hromozom):
    ret_list = np.array([[]])
    for i in range(len(hromozom)):
        ret_list = np.append(ret_list, kodiraj_broj(hromozom[i]))
    return ret_list

def bytes_to_hromozom(byte_list):
    ret_list = np.array([[]])
    while byte_list.size != 0:
        ret_list = np.append(ret_list, dekodiraj_broj(byte_list[0:bytes_in_hromozom]))
        for i in range(bytes_in_hromozom):
            byte_list = byte_list[1:]
    return ret_list.flatten()

def test_kvantizacija(pop_size):
    pop = random_population(pop_size)
    for i in range(pop_size):
        sum = 0
        for j in range(bytes_in_hromozom):
            sum += abs(pop[i][j]-bytes_to_hromozom(hromozom_to_bytes(pop[i]))[j])
    return sum/pop_size

def plot_prvog_tipa(najbolje_resenje_po_generacijama):
    plt.subplot(2, 1, 1)
    plt.plot(najbolje_resenje_po_generacijama[0])
    plt.plot(najbolje_resenje_po_generacijama[1])
    plt.plot(najbolje_resenje_po_generacijama[2])
    plt.ylabel("Trosak funkcije")
    maxTrosak = max(najbolje_resenje_po_generacijama[0][0], najbolje_resenje_po_generacijama[1][0], najbolje_resenje_po_generacijama[2][0])
    plt.axis([0, max_iter, 0, maxTrosak])

def plot_drugog_tipa(prosecni_trosak_po_programu):
    plt.subplot(2, 1, 2)
    plt.plot(prosecni_trosak_po_programu[0])
    plt.plot(prosecni_trosak_po_programu[1])
    plt.plot(prosecni_trosak_po_programu[2])
    plt.xlabel("Generacija")
    plt.ylabel("Prosecni trosak funkcije")
    maxTrosak = max(prosecni_trosak_po_programu[0][0], prosecni_trosak_po_programu[1][0], prosecni_trosak_po_programu[2][0])
    plt.axis([0, max_iter, 0, maxTrosak])

def genetski():
    npop_vel = 10

    ret_seed = read_file()
    random.seed(ret_seed)

    if output_path != "" and os.path.exists(output_path):
        output_file = open(output_path + "/output_file.txt", "w")
    else:  
        output_file = open("output_file.txt", "w")
    
    srednji_trosak = 0
    best = None
    best_f = None
    resenje_po_generacijama = []
    najbolje_resenje_po_generacijama = []
    prosecni_trosak = []
    prosecni_trosak_po_programu = []

    for k in range(iterations_over_one_program):
        najbolji_hromozom = None
        najbolji_trosak = None
        iter = 0

        pop = random_population(pop_size)

        while najbolji_trosak != 0 and iter < max_iter:
            n_pop = pop[:]
            while len(n_pop) < pop_size + npop_vel:
                h1 = turnir(pop)
                h2 = turnir(pop)
                h3, h4 = ukrsti(hromozom_to_bytes(h1), hromozom_to_bytes(h2))
                h3 = mutiraj(h3, mut_rate)
                h4 = mutiraj(h4, mut_rate)
                n_pop.append(bytes_to_hromozom(h3))
                n_pop.append(bytes_to_hromozom(h4))
            pop = sorted(n_pop, key=lambda x : trosak(x))[:pop_size]
            suma_troskova = 0
            for l in range(len(pop)):
                suma_troskova += trosak(pop[l])
            prosecni_trosak.append(suma_troskova / len(pop))
            f = round(trosak(pop[0]), 2)
            resenje_po_generacijama.append(f)
            if najbolji_trosak is None or najbolji_trosak > f:
                najbolji_trosak = f
                najbolji_hromozom = pop[0]
            iter += 1

        srednji_trosak += najbolji_trosak
        output_str = "Najbolji trosak u " + str(k+1) + ". generaciji je: " + str(round(najbolji_trosak, 2)) + "\n"
        output_file.write(output_str)
        if best_f is None or best_f > najbolji_trosak:
            best = najbolji_hromozom
            best_f = najbolji_trosak
        najbolje_resenje_po_generacijama.append(resenje_po_generacijama)
        prosecni_trosak_po_programu.append(prosecni_trosak)
        resenje_po_generacijama = []
        prosecni_trosak = []

    srednji_trosak /= 3

    output_file.write('Srednji trosak: %.2f \n' % srednji_trosak)
    output_file.write('Najbolji trosak: %.2f \n\n' % best_f)
    output_file.write('Najbolje resenje: %s \n\n' % best)
    output_file.write('Najbolje resenje u binarnom obliku: %s \n' % hromozom_to_bytes(best))
    output_file.close()

    plot_prvog_tipa(najbolje_resenje_po_generacijama)
    plot_drugog_tipa(prosecni_trosak_po_programu)
    plt.show()

genetski()
