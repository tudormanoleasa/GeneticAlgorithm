import math
import random
import copy
from statistics import mean
class Genetic:
    def __init__(self, dim, dimPop, pm, pc, func):
        self.dim = dim
        self.dimPop = dimPop
        self.pm = pm
        self.pc = pc
        self.func = func
        if func == 1 or func ==2:
            #Dejong, Rastrigin
            self.a = [-5.12 for i in range(0, self.dim)]
            self.b = [5.12 for i in range(0, self.dim)]
        elif func == 3:
            #Rosenbrock
            self.a = [-2.048 for i in range(0, self.dim)]
            self.b = [2.048 for i in range(0, self.dim)]
        else:
            #Sptring Design
            self.a = [0.25, 0.05, 2.0]
            self.b = [1.3, 2.0, 15.0]
            self.constraints = [lambda x1, x2, x3: 1 - ((x1 ** 3 * x3)/(71785 * x2 ** 4)) <= 0,
                                lambda x1, x2, x3: (((4 * x1 ** 2) - (x1 * x2)) / (12566 * (x1 * x2 ** 3 - x2 ** 4))) + 1 / (5108 * x2 ** 2) -1 <=0,
                                lambda x1, x2, x3: 1 - ((140.45 * x2) / (x1 ** 2 * x3)) <= 0,
                                lambda x1, x2, x3: ((x1 + x2) / 1.5) - 1 <= 0]

    def generatePopulation(self):
        self.population = []
        for i in range(0, self.dimPop):
            if self.func <= 3:
                candidate_solution = []
                for j in range(0, self.dim):
                    candidate_solution.append(random.uniform(self.a[j], self.b[j]))
            else:
                while 1:
                    candidate_solution = []
                    candidate_solution.append(random.uniform(self.a[0], self.b[0]))
                    candidate_solution.append(random.uniform(self.a[1], self.b[1]))
                    candidate_solution.append(random.uniform(self.a[2], self.b[2]))
                    ok = 0
                    for constraint in self.constraints:
                        if constraint(candidate_solution[0], candidate_solution[1], candidate_solution[2]) == False:
                            ok = 1
                    if ok == 0:
                        break
            self.population.append(candidate_solution)

    def f(self, vec):
        if self.func == 1:
            return 1 / self.dejong(vec)
        elif self.func == 2:
            return 1 / self.rastrigin(vec)
        elif self.func == 3:
            return 1 / self.rosenbrock(vec)
        else:
            return 1 / self.springdesign(vec)

    def dejong(self, vec):
        return sum([elem ** 2 for elem in vec])

    def rastrigin(self, vec):
        return 10 * self.dim + sum([elem ** 2 - 10 * math.cos(2 * math.pi * elem) for elem in vec])

    def rosenbrock(self, vec):
        return sum([100 * (vec[i+1] - vec[i] ** 2) ** 2 + (1 - vec[i]) ** 2 for i in range(0, len(vec) - 1)])

    def springdesign(self, vec):
        return (vec[2] + 2) * vec[0] * (vec[1] ** 2)

    def random_mutation(self):
        for i in range(0, self.dimPop):
            for j in range(0, len(self.population[i])):
                if random.uniform(0, 100) > self.pm:
                    self.population[i][j] = random.uniform(self.a[j], self.b[j])

    def nonuniform_mutation(self, t, tmax):
        for i in range(0, self.dimPop):
            for j in range(0, len(self.population[i])):
                if random.uniform(0, 100) > self.pm:
                    ok = 0
                    val = 0
                    while not ok:
                        ok = 1
                        tau = random.randint(0, 1)
                        r = random.uniform(0, 1)
                        b = -2
                        val = self.population[i][j]
                        if tau:
                            self.population[i][j] -= (self.population[i][j] - self.a[j]) * (1 - r ** ((1 - t / tmax) ** b))
                        else:
                            self.population[i][j] += (self.b[j] - self.population[i][j]) * (1 - r * ((1 - t / tmax) ** b))
                        if val < self.a[j] or val > self.b[j]:
                            ok = 0
                    self.population[i][j] = val

    def muhlenbein_mutation(self):
        for i in range(0, self.dimPop):
            for j in range(0, len(self.population[i])):
                if random.uniform(0, 100) > 99:
                    ok = 0
                    val = 0
                    while not ok:
                        ok = 1
                        rang = 0.1 * (self.b[j] - self.a[j])
                        v = []
                        val = self.population[i][j]
                        for i in range(15):
                            p_alfa = random.randint(1, 16)
                            alfa = 0
                            if p_alfa == 1:
                                alfa = 1
                            v.append(alfa * (2 ** (-i)))
                        gamma = sum(v)
                        if random.uniform(0, 1) > 0.5:
                            val += rang * gamma
                        else:
                            val -= rang * gamma
                        if val < self.a[j] or val > self.b[j]:
                                ok = 0
                    self.population[i][j] = val

    def real_number_creep_mutation(self):
        maximum = 0
        maximum_chromosome = 0
        for i in range(self.dimPop):
            f = self.f(self.population[i])
            if f > maximum:
                maximum = f
                maximum_chromosome = self.population[i]
        for i in range(0, self.dimPop):
            for j in range(0, len(self.population[i])):
                if random.uniform(0, 100) > self.pm:
                    ok = 0
                    val = 0
                    while not ok:
                        ok = 1
                        modify_value = random.uniform(0, math.fabs(self.population[i][j] - maximum_chromosome[j])) * 0.5
                        val = self.population[i][j]
                        if random.uniform(0, 1) > 0.5:
                            val -= modify_value
                        else:
                            val += modify_value
                        if val < self.a[j] or val > self.b[j]:
                            ok = 0
                    self.population[i][j] = val

    def get_chromosomes_for_crossover(self):
        chromosome1 = random.randint(0, self.dimPop - 1)
        chromosome2 = random.randint(0, self.dimPop - 1)
        while chromosome1 == chromosome2:
            chromosome1 = random.randint(0, self.dimPop - 1)
            chromosome2 = random.randint(0, self.dimPop - 1)
        return chromosome1, chromosome2

    def simple_crossover(self):
        """
        Verificam mai intai daca ne este respectata probabilitatea de 25% de incrucisare
        """
        if not random.randint(1, 100) <= self.pc:
            return
        """
        Daca da, interschimb prima jumatate de dimensiuni a cromozomului 1 cu prima jumatate de dimensiuni a cromozomului 2 
        """
        chromosome1, chromosome2 = self.get_chromosomes_for_crossover()
        aux1 = self.population[chromosome1][0:self.dim // 2]
        aux2 = self.population[chromosome2][0:self.dim // 2]
        for j in range(0, self.dim // 2):
            self.population[chromosome1][j] = aux2[j]
            self.population[chromosome1][j] = aux1[j]

    def flat_crossover(self):
        if not random.randint(1, 100) <= self.pc:
            return
        chromosome1, chromosome2 = self.get_chromosomes_for_crossover()
        """
        Pentru fiecare dimensiune a fiecarui cromozom voi genera un nou numar cuprins intre
        dimensiunea i a cromozomului 1 si dimensiunea i a cromozomului 2
        """
        for i in range(0, self.dim):
            self.population[chromosome1][i] = random.uniform(self.population[chromosome1][i],
                                                             self.population[chromosome2][i])
            self.population[chromosome2][i] = random.uniform(self.population[chromosome1][i],
                                                             self.population[chromosome2][i])

    def linear_crossover(self):
        if not random.randint(1, 100) <= self.pc:
            return
        chromosome1 = 0
        chromosome2 = 0
        ok = 0
        aux1 = []
        aux2 = []
        aux3 = []
        while not ok:
            chromosome1, chromosome2 = self.get_chromosomes_for_crossover()
            ok = 1
            """
            Se construiesc 3 noi cromozomi (aux1, aux2, aux3) conform formulei din pdf
            """
            aux1 = []
            aux2 = []
            aux3 = []
            for i in range(0, self.dim):
                aux1.append(1 / 2 * self.population[chromosome1][i] + 1 / 2 * self.population[chromosome2][i])
                aux2.append(3 / 2 * self.population[chromosome1][i] - 1 / 2 * self.population[chromosome2][i])
                aux3.append(-1 / 2 * self.population[chromosome1][i] + 3 / 2 * self.population[chromosome2][i])
            """
            Daca vreun cromozom din cei 3 formati are in una din dimensiuni un numar care nu e in intervalul [a, b]
            atunci se reia procedeul  
            """
            for j in range(0, self.dim):
                if aux1[j] < self.a[j] or aux1[j] > self.b[j]:
                    ok = 0
                if aux2[j] < self.a[j] or aux2[j] > self.b[j]:
                    ok = 0
                if aux3[j] < self.a[j] or aux3[j] > self.b[j]:
                    ok = 0

        """
        Din cei 3 cromozomi creati mai sus, se aleg primii 2 dpdv al calitatii 
        """
        if self.f(aux1) > self.f(aux2):
            if self.f(aux2) > self.f(aux3):
                self.population[chromosome1] = copy.deepcopy(aux1)
                self.population[chromosome2] = copy.deepcopy(aux2)
            else:
                self.population[chromosome1] = copy.deepcopy(aux1)
                self.population[chromosome2] = copy.deepcopy(aux3)
        else:
            if self.f(aux1) > self.f(aux3):
                self.population[chromosome1] = copy.deepcopy(aux1)
                self.population[chromosome2] = copy.deepcopy(aux2)
            else:
                self.population[chromosome1] = copy.deepcopy(aux2)
                self.population[chromosome2] = copy.deepcopy(aux3)

    def extended_line_crossover(self):
        if not random.randint(1, 100) <= self.pc:
            return
        chromosome1, chromosome2 = self.get_chromosomes_for_crossover()
        ok = 0
        aux1 = []
        aux2 = []
        while not ok:
            ok = 1
            alfa1 = random.uniform(-0.25, 1.25)
            alfa2 = random.uniform(-0.25, 1.25)
            aux1 = []
            aux2 = []
            """
            Se construiesc 2 noi cromozomi (aux1, aux2) conform formulei din pdf
            """
            for i in range(0, self.dim):
                aux1.append(self.population[chromosome1][i] + alfa1 * (self.population[chromosome2][i] -
                                                                       self.population[chromosome1][i]))
                aux2.append(self.population[chromosome1][i] + alfa2 * (self.population[chromosome2][i] -
                                                                       self.population[chromosome1][i]))
            """
            Daca vreun cromozom din cei 2 formati are in una din dimensiuni un numar care nu e in intervalul [a, b]
            atunci se reia procedeul  
            """
            for j in range(0, self.dim):
                if aux1[j] < self.a[j] or aux1[j] > self.b[j]:
                    ok = 0
                if aux2[j] < self.a[j] or aux2[j] > self.b[j]:
                    ok = 0

        """
        Cei 2 cromozomi vechi sunt inlocuiti de cei noi obtinuti in while-ul de mai sus
        """
        self.population[chromosome1] = copy.deepcopy(aux1)
        self.population[chromosome2] = copy.deepcopy(aux2)

    def extended_intermediate_crossover(self):
        if not random.randint(1, 100) <= self.pc:
            return
        """
        Procedeul este aproximativ acelasi ca la extended_line_crossover.
        Diferenta fata de extended_line_crossover e aceea ca aici alfa se genereaza diferit pt fiecare dimensiune in parte.
        """
        chromosome1, chromosome2 = self.get_chromosomes_for_crossover()
        aux1 = []
        aux2 = []
        for i in range(0, self.dim):
            ok = 0
            while not ok:
                ok = 1
                alfa1 = random.uniform(-0.25, 1.25)
                alfa2 = random.uniform(-0.25, 1.25)
                aux1.append(self.population[chromosome1][i] + alfa1 * (self.population[chromosome2][i] -
                                                                       self.population[chromosome1][i]))
                aux2.append(self.population[chromosome1][i] + alfa2 * (self.population[chromosome2][i] -
                                                                       self.population[chromosome1][i]))

                if aux1[i] < self.a[i] or aux1[i] > self.b[i] or aux2[i] < self.a[i] or aux2[i] > self.b[i]:
                    ok = 0
                    del aux2[-1]
                    del aux1[-1]

        self.population[chromosome1] = copy.deepcopy(aux1)
        self.population[chromosome2] = copy.deepcopy(aux2)

    def wright_heuristic_crossover(self):
        if not random.randint(1, 100) <= self.pc:
            return
        chromosome1, chromosome2 = self.get_chromosomes_for_crossover()
        if self.f(self.population[chromosome2]) > self.f(self.population[chromosome2]):
            chromosome1, chromosome2 = chromosome2, chromosome1
        ok = 0
        aux1 = []
        aux2 = []
        while not ok:
            ok = 1
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            aux1 = []
            aux2 = []
            for i in range(0, self.dim):
                aux1.append(r1 * (self.population[chromosome1][i] - self.population[chromosome2][i]) +
                            self.population[chromosome1][i])
                aux2.append(r2 * (self.population[chromosome1][i] - self.population[chromosome2][i]) +
                            self.population[chromosome1][i])
            for j in range(0, self.dim):
                if aux1[j] < self.a[j] or aux1[j] > self.b[j]:
                    ok = 0
                if aux2[j] < self.a[j] or aux2[j] > self.b[j]:
                    ok = 0
        self.population[chromosome1] = copy.deepcopy(aux1)
        self.population[chromosome2] = copy.deepcopy(aux2)

    def selectionWheel(self):
        eval = []
        total = 0
        for i in range(0, self.dimPop):
            valF = self.f(self.population[i])
            eval.append(valF)
            total = total + valF
        p = []
        for i in range(0, self.dimPop):
            p.append(eval[i]/total)
        q = [0]
        for i in range(0, self.dimPop):
            q.append(q[i] + p[i])
        q[-1] = 1
        newPop = []
        for i in range(0, self.dimPop):
            r = random.uniform(0.000001,1)
            ok = 0
            maxi = 99999999
            cr = -1
            for j in range(0, self.dimPop - 1):
                if q[j] < r <= q[j+1]:
                    if self.springdesign(self.population[j + 1]) < maxi:
                        maxi = self.springdesign(self.population[j + 1])
                        cr = j + 1
            if cr == -1:
                newPop.append(self.population[i])
            else:
                newPop.append(self.population[cr])
        return newPop


    def selectionElitism(self):
        # retin cele mai bune k solutii candidat
        k = 50
        self.population = sorted(self.population, key=lambda elem: self.springdesign(elem))
        best_candidate_solutions = [self.population[i] for i in range(0, k)]
        """
        apalez roata dupa care sortez oleaca descrescator solutiile candidat ca sa pot inlocui pe alea mai mari(alea proaste adica)
        cu alea mai 100 pastrate deasupra
        """
        self.population = self.selectionWheel()
        self.population.sort(key=lambda elem: self.springdesign(elem), reverse=True)
        for i in range(0, k):
            self.population[i] = copy.deepcopy(best_candidate_solutions[i])
        return self.population


    def selectionRank(self):
        self.population.sort(key=lambda elem: self.springdesign(elem))
        ranks = [i for i in range(0, self.dimPop)]
        """
        1) q se alege in asa fel incat suma probabilitatilor sa fie aproximativ 1.
        2) Daca daca dau alta dimensiune la populatie, q trebuie  schimbat
        """
        q = 0.005
        p = [q]
        for i in range(1, self.dimPop):
            p.append(q * ((1-q)**(i-1)))
        """
        In continuare se face ruleta, cu probabilitatile calculate in functie de q
        """
        Q = [0]
        for i in range(0, self.dimPop):
            Q.append(Q[i] + p[i])
        newPop = []
        for i in range(0, self.dimPop):
            r = random.uniform(0.000001,1)
            ok = 0
            for j in range(0, self.dimPop - 1):
                if Q[j] < r <= Q[j+1]:
                    newPop.append(self.population[j])
                    ok = 1
                    break
            if ok == 0:
                newPop.append(self.population[i])
        return newPop


    def geneticAlgorithm(self):
        self.generatePopulation()
        t = 0
        minim =100000
        while t<500:
            print(t)
            self.population = copy.deepcopy(self.selectionRank())
            self.muhlenbein_mutation()
            self.linear_crossover()
            mini = min([self.springdesign(elem) for elem in self.population])
            if minim > mini:
                minim = mini
            t = t + 1
        print(t)
        return minim



if __name__ == "__main__":
    l = list()
    repetitions = 1
    while repetitions <= 5:
        elem = Genetic(dim = 3, dimPop = 1000, pm = 1, pc = 25, func = 4)
        l.append(elem.geneticAlgorithm())
        repetitions += 1
    print(str(int(min(l)*100000)/100000)+" & "+str(int(mean(l)*100000)/100000)+" & "+str(int(max(l)*100000)/100000)+ " & 0.0025")
