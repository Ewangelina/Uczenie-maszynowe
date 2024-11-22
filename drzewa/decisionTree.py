import numpy as np
from collections import Counter
import random
import string

categorical_value_options = [0] #Zmienione z [0, 1] bo jeśli pole ma 2 opcje do dla 0 będzie T albo F a dla 1 odwrotnie więc nie trzeba sprawdzać obu
numerical_value_options = np.linspace(-0.1, 0,9, 50).tolist() #30 wartości od -0,1 do 0,9

#Używane do określenia czy dane kolumna powinna być traktowana jako liczba czy kategoria
#0 to kategoria 1 to liczba
#data_categories = [1,1,1,1,1,1,1,0,...] nie chce mi się wypisywać 420 cech więc wszystkie liczbowe są na początku

class decisionTree:
    def __init__(self, tree_destination=".\\trees\\196.csv"):
        self.tree_destination = tree_destination
        w = open(self.tree_destination, "w")
        w.close()

    def isUniform(self, ratings):
        most_common = Counter(ratings).most_common(1)[0]
        current_uniformity = most_common[1] / len(ratings)
        if current_uniformity >= self.uniformity:
            return True
        else:
            return False
        
    def addNode(self, string):
        o = open(self.tree_destination, "a")
        string = string + "\n"
        o.write(string)
        o.close()

    def generateRandomId(self):
        return ''.join([random.choice(string.ascii_letters + string.digits) for n in range(40)])

    #generuje liczbę z zakresu 0 do (length-1) (mniejszą niż 100000)
    def generateRowNumber(self, length):
        num = random.random() * 100000
        num = int(num % length)
        return num
        
    # condition_string to np. "4 < 0,5" gdzie:
    # 4 to indeks ewaluowanej kolumny
    # oddzielone spacjami, "<" to kolumna mniejsza od, "=" to kolumna równa
    # wartość używana do porównywania
    def useCondition(self, condition_string, data, ratings):
        split_str = condition_string.split(" ")
        column_id = int(split_str[0])
        action_str = split_str[1]
        value = float(split_str[2])

        result_true = []
        result_false = []
        result_rating_true = []
        result_rating_false = []

        if action_str == "=":
            for i in range(len(data)):
                if data[i][column_id] == value:
                    result_true.append(data[i])
                    result_rating_true.append(ratings[i])
                else:
                    result_false.append(data[i])
                    result_rating_false.append(ratings[i])
        elif action_str == "<":
            for i in range(len(data)):
                if data[i][column_id] == value:
                    result_true.append(data[i])
                    result_rating_true.append(ratings[i])
                else:
                    result_false.append(data[i])
                    result_rating_false.append(ratings[i])

        return result_false, result_rating_false, result_true, result_rating_true
        
    #uniformity -> jak czyste muszą być zbiory zwierające tylko 1 wynik
    #min_size -> ile minimalnie może być przypadków w zbiorze
    def createTree(self, data, ratings, uniformity=1, min_size=3):
        self.uniformity = uniformity
        self.min_size = min_size
        if len(ratings) > 2*min_size: #can the set be divided into two
            self.createNode("0", data, ratings)
        else:
            answer = str(Counter(ratings).most_common(1)[0][0])
            node_string = node_id + ";0 = 0;!" + str(answer) + ";!" + str(answer)
            self.addNode(node_string)
        

    def createNode(self, node_id, data, ratings):
        considered_rows = []
        breaking_loop = False
        split_in_half = False
        node_false_result = None
        node_correct_result = None
        no_loops = 0
        half_tolerance = 0.1

        comparison_value = len(data[0])
        while len(considered_rows) < comparison_value: #póki każda z kolumn nie została rozważona
            row = self.generateRowNumber(len(data[0]))
            if row not in considered_rows:
                considered_rows.append(row)
                used_value_options = None
                action_str = None

                if row <= 6: #Pierwsze 6 indeksów to numeryczne, reszta to kategoryczne
                    used_value_options = numerical_value_options
                    action_str = "<"
                else:
                    used_value_options = categorical_value_options
                    action_str = "="

                for el in used_value_options:
                    condition_string = str(row) + " " + action_str + " " + str(el)
                    result_false, result_rating_false, result_true, result_rating_true = self.useCondition(condition_string, data, ratings)

                    if split_in_half: #splits set in half
                        no_loops = no_loops + 1
                        proportion_of_false_set = len(result_rating_false) / len(ratings)
                        #print(proportion_of_false_set)
                        if proportion_of_false_set > 0.5 - half_tolerance and proportion_of_false_set < 0.5 + half_tolerance: #tolerances for half
                            if len(result_rating_false) >= self.min_size * 2 and len(result_rating_true) >= self.min_size * 2:
                                node_correct_result = self.generateRandomId()
                                node_false_result = self.generateRandomId()
                                print("SPLIT")
                                node_string = node_id + ";" + condition_string + ";" + node_false_result + ";" + node_correct_result
                                self.addNode(node_string)
                                comparison_value = -1  #end loop   
                                self.createNode(node_correct_result, result_true, result_rating_true)
                                self.createNode(node_false_result, result_false, result_rating_false)
                                break;

                        if len(considered_rows) == comparison_value: #has exhausted all columns
                            half_tolerance = half_tolerance + 0.1
                            if half_tolerance < 0.5:
                                considered_rows = [] #Lower half tolerance and start over
                            else:
                                print("WARNING!!! imperfect tree created for " + self.tree_destination + " with min_size: " + str(self.min_size) + " and uniformity: " + str(self.uniformity))
                                print("Set not separated: " + str(len(ratings)) + "->" + str(ratings))
                                answer = str(Counter(ratings).most_common(1)[0][0])
                                node_string = node_id + ";0 = 0;!" + str(answer) + ";!" + str(answer)
                                self.addNode(node_string)
                                                     
                    else: # splits creating an uniform set 
                        no_loops = no_loops + 1
                        if len(result_rating_false) >= self.min_size and len(result_rating_true) >= self.min_size:
                            if self.isUniform(result_rating_false):
                                print(result_rating_false)
                                node_false_result = '!' + str(Counter(result_rating_false).most_common(1)[0][0])
                                breaking_loop = True

                            if self.isUniform(result_rating_true):
                                node_correct_result = '!' + str(Counter(result_rating_true).most_common(1)[0][0])
                                breaking_loop = True

                        if breaking_loop: #creates node WITH FIRST FOUND CONDITION                    
                            if node_correct_result is None:
                                node_correct_result = self.generateRandomId()
                                if len(result_rating_true) > 2*self.min_size: #can the remaining set be divided into two
                                    print("From correct " + node_id)
                                    self.createNode(node_correct_result, result_true, result_rating_true)
                                else:
                                    node_correct_result = '!' + str(Counter(result_rating_true).most_common(1)[0][0]) #Jeśli rezulat warunku zaczyna się od ! to jest to zwracany wynik

                            if node_false_result is None:
                                node_false_result = self.generateRandomId()
                                if len(result_rating_true) > 2*self.min_size: #can the remaining set be divided into two
                                    print("From false " + node_id)
                                    self.createNode(node_false_result, result_false, result_rating_false)
                                else:
                                    node_false_result = '!' + str(Counter(result_rating_true).most_common(1)[0][0]) #Jeśli rezulat warunku zaczyna się od ! to jest to zwracany wynik

                            node_string = node_id + ";" + condition_string + ";" + node_false_result + ";" + node_correct_result
                            self.addNode(node_string)
                            comparison_value = -1  #end loop
                            break;

                        if len(considered_rows) == comparison_value: #has exhausted all columns
                            #switch to splitting in half
                            considered_rows = []
                            split_in_half = True
                                
                        

    def getNode(self, node_id, filepath):
        o = open(filepath, "r")
        for line in o:
            this_id = str(line.split(";")[0])
            if str(this_id) == str(node_id):
                o.close()
                return line

        print("Node " + str(node_id) + " not found in " + str(filepath))
        exit(0)

    def useNode(self, row, node):
        split_str = node.split(";")
        condition_string = split_str[1]
        result_false = split_str[2]
        result_true = split_str[3]

        split_str = condition_string.split(" ")
        column_id = int(split_str[0])
        action_str = split_str[1]
        value = float(split_str[2])

        if action_str == "=":
            if row[column_id] == value:
                return result_true
            else:
                return result_false
        elif action_str == "<":
            if row[column_id] < value:
                return result_true
            else:
                return result_false
        else:
            print("useNode error")
            exit(0)

        
    def useTree(self, row, filepath=None):
        if filepath is None:
            filepath = self.tree_destination
        current_node = 0
        
        while True:
            node = self.getNode(current_node, filepath)
            result = self.useNode(row, node)
            if result[0] == "!":
                return result[1:]
                break
            else:
               current_node = result.replace("\n", "")
