from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import factorial
from math import ceil
from copy import deepcopy
from sympy import latex
from sympy import sympify
from scipy.special import factorial2

class Xi:
	def __init__(self, index):
		self.index = index

class Metric:
	def __init__(self, index1, index2, dirac_deriv_lst):
		self.index1 = index1
		self.index2 = index2
		self.dirac_deriv_lst = dirac_deriv_lst

class B:
	def __init__(self, index, dirac_deriv_lst):
		self.index = index
		self.dirac_deriv_lst = dirac_deriv_lst

class Dirac_deriv:
	def __init__(self, index):
		self.index = index

class Term:
	def __init__(self, coefficient, r_count, metric_lst, b_lst, xi_lst, eq_indices, noneq_indices, constant_lst):
		self.coefficient = coefficient
		self.r_count = r_count
		self.metric_lst = metric_lst
		self.b_lst = b_lst
		self.xi_lst = xi_lst
		self.eq_indices = eq_indices
		self.noneq_indices = noneq_indices
		self.constant_lst = constant_lst

class Riemann:
	def __init__(self, index1, index2, index3, index4, deriv_lst, important_indices, dummy_indices, tensor_type):
		self.index1 = index1
		self.index2 = index2
		self.index3 = index3
		self.index4 = index4
		self.deriv_lst = deriv_lst
		self.important_indices = important_indices
		self.dummy_indices = dummy_indices
		self.tensor_type = "Riemann"

class Constant:
	def __init__(self, index, dirac_deriv_lst):
		self.index = index
		self.dirac_deriv_lst = dirac_deriv_lst

def accel_asc_even(n): #Calculates integer partitions for an integer n with each entry an even integer
    a = [0 for i in range(0, n+1, 2)]
    k = 1
    y = n - 2
    while k != 0:
        x = a[k - 1] + 2
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 2
            y -= 2
        a[k] = x + y
        y = x + y - 2
        yield a[:k + 1]

def term_characteristic(term, index, j): #Used as an accouting method when comparing different Term objects
	term_characteristic_lst = [term.coefficient, term.r_count] #[coefficient, r_count, [g, Dg,..., D^j g], [b, Db,..., D^j b], xi_count, non_equal_index list, index]
	#This list counts the number of derivatives and ignores combinations of them
	metric_count = [0 for i in range(0, j+1)]
	b_count = [0 for i in range(0, j+1)]
	for metric in term.metric_lst:
		if len(metric.dirac_deriv_lst) != 0:
			metric_count[len(metric.dirac_deriv_lst)] += 1
		if len(metric.dirac_deriv_lst) == 0:
			metric_count[0] += 1
	for b in term.b_lst:
		if len(b.dirac_deriv_lst) != 0:
			b_count[len(b.dirac_deriv_lst)] += 1
		if len(b.dirac_deriv_lst) == 0:
			b_count[0] += 1
	term_characteristic_lst.append(metric_count)
	term_characteristic_lst.append(b_count)
	term_characteristic_lst.append(len(term.xi_lst))
	term_characteristic_lst.append(len(term.eq_indices))
	term_characteristic_lst.append(len(term.noneq_indices))
	term_characteristic_lst.append(index)
	return term_characteristic_lst

def recursion(n):
	term = Term(1,1,[],[],[],[],[],[])
	dirac_array = [[[term]]]
	index_counter = 1
	normal_coord_output = []
	taylor_expansion = []
	taylor_coefficients = []
	inner_product_lst = [[1,"JJ"]]
	max_d_count = 0 #Used and is updated when needed to help combine like terms after differentiating the inner product(s)
	taylor_length = 0

	if n == 0: #Base case for the recursion
		return dirac_array[0][0]
	else:
		deriv_index_lst = []
		for j in range(1,n+1):
			deriv_index_lst.append(index_counter)#Keeps the numbering of the derivative indices from clashing with the indices of Metric objects and B objects
			index_counter += 1
			new_iteration = []
			for row in range(0, j):
				dirac_array[row].append("") #Prepares room for calculating a new D^m r_{-2-j} to terms from previous steps
			dirac_array.append([""]) #Adds a new row to store the new term in the sequence
			for l in range(0, j):
				for k in range(0, 2):
					m = j-k-l
					deriv_indices = deriv_index_lst[:]
					if k == 0: #Handles when taking m partial derivatives of Metric objects
						if m == 0:
							metric = Metric(index_counter, index_counter+1, [])
							xi1 = Xi(index_counter)
							xi2 = Xi(index_counter+1)
							index_counter += 2
							partial_term = [1, metric, [xi1, xi2]] #[coefficient, metric or b, list of xi's]
						if m == 1:
							metric = Metric(deriv_indices[-1], index_counter, [])
							xi = Xi(index_counter)
							index_counter += 1
							partial_term = [2, metric, [xi]]
						if m == 2:
							metric = Metric(deriv_indices[-1], deriv_indices[-1], [])
							partial_term = [2, metric, []]
						if m > 2: #As Metric has no xi dependence 
							partial_term = [0, [], []]
					else: #Handles when taking m partial derivatives of B objects
						if m == 0:							
							b = B(index_counter, [])
							xi = Xi(index_counter)
							index_counter += 1
							partial_term = [1, b, [xi]]
						if m == 1:
							b = B(deriv_indices[0], [])
							partial_term = [1, b, []]
						if m > 1: # As B has no xi dependence
							partial_term = [0, []]
					if partial_term[0] != 0:
						if dirac_array[l][m] == "": #Checks if the value is not already calculated
							new_term_lst = []
							old_dirac = dirac_array[l][m-1]
							for term in old_dirac: #Takes the Dirac derivative of each summation in the sequence term
								new_term = deepcopy(term)
								new_term.coefficient *= -new_term.r_count
								new_term.r_count += 1
								new_term.metric_lst.append(Metric(index_counter, index_counter+1, [Dirac_deriv(deriv_indices[-1])]))
								new_term.xi_lst.append(Xi(index_counter))
								new_term.xi_lst.append(Xi(index_counter+1))
								index_counter += 2
								new_term_lst.append(new_term)
								if len(term.metric_lst) != 0:
									for i in range(0, len(term.metric_lst)):
										new_term = deepcopy(term) #Performs product rule on the summation
										new_term.metric_lst[i].dirac_deriv_lst.append(Dirac_deriv(deriv_indices[-1]))
										if m == 2 and len(new_term.metric_lst[i].dirac_deriv_lst) >= 2: #Makes it so that the two derivatives are taken with respect to the same variable as they would otherwise vanish when converting to normal coordinates
											new_term.eq_indices.append([new_term.metric_lst[i].dirac_deriv_lst[-2].index, new_term.metric_lst[i].dirac_deriv_lst[-1].index])
											new_term.metric_lst[i].dirac_deriv_lst[-2].index = new_term.metric_lst[i].dirac_deriv_lst[-1].index
										new_term_lst.append(new_term)
								if len(term.b_lst) != 0:
									for i in range(0, len(term.b_lst)):
										new_term = deepcopy(term) #Performs product rule on the summation
										new_term.b_lst[i].dirac_deriv_lst.append(Dirac_deriv(deriv_indices[-1]))
										if m == 2 and len(new_term.b_lst[i].dirac_deriv_lst) >= 2: #Makes it so that the two derivatives are taken with respect to the same variable
											new_term.eq_indices.append([new_term.b_lst[i].dirac_deriv_lst[-2].index, new_term.b_lst[i].dirac_deriv_lst[-1].index])
											new_term.b_lst[i].dirac_deriv_lst[-2].index = new_term.b_lst[i].dirac_deriv_lst[-1].index
										new_term_lst.append(new_term)
							dirac_array[l][m] = new_term_lst
						dirac_term = dirac_array[l][m]
					factorial_multiplier = factorial(m)
					for term in dirac_term:	#Changes indices depending on the partition and updates the coefficients of terms					
						new_term = deepcopy(term)
						new_term.coefficient *= -Fraction(1, factorial_multiplier) * partial_term[0] #Multiplies by -r_{-2}
						new_term.r_count += 1
						#This next block multiplies the term by \partial^\mu a_{2-k}
						if isinstance(partial_term[1], Metric):
							new_term.metric_lst.append(deepcopy(partial_term[1]))
						elif isinstance(partial_term[1], B):
							new_term.b_lst.append(deepcopy(partial_term[1]))
						for xi in partial_term[-1]:
							new_term.xi_lst.append(deepcopy(xi))
						
						#This block converts to normal coordinates (i.e. keeps terms derivatives of metrics of order \mu or derivatives of b's of order \mu-1 or a product of b's with no derviatives)
						metric_boolean = True
						b_boolean = True
						for metric in new_term.metric_lst:
							if len(metric.dirac_deriv_lst) < j and len(metric.dirac_deriv_lst) > 0:
								metric_boolean = False
								break
						for b in new_term.b_lst:
							if len(b.dirac_deriv_lst) < j-1 and len(b.dirac_deriv_lst) > 0:
								b_boolean = False
								break
						if metric_boolean == True and b_boolean == True:
							new_iteration.append(new_term)
			term_counts = []
			temp = []
			for i in range(0, len(new_iteration)): #Generates list to help compare Term objects and test if they are equal up to their coefficients and index labeling
				term_counts.append(term_characteristic(new_iteration[i], i, j))
			for i in range(0, len(new_iteration)):
				boolean = False #Used to denote if this term is to be added to another
				for item in temp:
					if term_counts[i][1:7] == item[1:7]: #Set at fixed numbers due to the format of the output from the term_characteristic function
						item[0] += term_counts[i][0]
						boolean = True
				if boolean == False:
					temp.append(term_counts[i])
			final_version = []
			for item in temp: #Updates Term object coefficients and deletes all but one of the equal Terms
				index = item[-1]
				new_term = deepcopy(new_iteration[index])
				new_term.coefficient = item[0]
				if new_term.coefficient != 0:
					final_version.append(new_term)
			dirac_array[j][0] = deepcopy(final_version)
			while taylor_length < j:
				new_inner_product_lst = []
				for inner_product in inner_product_lst:
					for i in range(0, len(inner_product[1])):
						temp = inner_product[:]
						if inner_product[1][i] == "J":
							if inner_product[1][i-1] == "D": #Uses the Jacobi Field equation DDJ = RJ to make a substitution
								temp[1] = temp[1][:i-1] + "R" + temp[1][i:]
								new_inner_product_lst.append(temp)
								if temp[1].count("D") > max_d_count:
									max_d_count = temp[1].count("D")
							else: #Applies a derivative to a J (if the J does not already have a derivative)
								temp[1] = temp[1][:i] + "D" + temp[1][i:]
								new_inner_product_lst.append(temp)
								if temp[1].count("D") > max_d_count:
									max_d_count = temp[1].count("D")
						elif inner_product[1][i] == "R": #Applies a derivative to an R
							temp[1] = temp[1][:i] + "D" + temp[1][i:]
							new_inner_product_lst.append(temp)
							if temp[1].count("D") > max_d_count:
								max_d_count = temp[1].count("D")
				symbol_counts = [] #[coefficient of the inner product,J,R,DJ,DR,DDR,etc.,index in symbol_counts]
				for k in range(0, len(new_inner_product_lst)):
					count = [new_inner_product_lst[k][0]]
					count.append(new_inner_product_lst[k][1].count("J"))
					count.append(new_inner_product_lst[k][1].count("R"))
					if max_d_count > 0:
						count.append(new_inner_product_lst[k][1].count("DJ"))
						for i in range(0, j-2):
							count.append(new_inner_product_lst[k][1].count("D"*i + "R"))
					count.append(k)
					symbol_counts.append(count)
				temp = []
				final_version = []
				for i in range(0,len(symbol_counts)): #Decides which terms to combine (and which ones to keep after)
					boolean = False
					for item in temp:
						if item[1:-1] == symbol_counts[i][1:-1]:
							item[0] += symbol_counts[i][0]
							boolean = True
					if boolean == False:
						temp.append(symbol_counts[i])
				for item in temp:
					to_append = new_inner_product_lst[item[-1]][:]
					to_append[0] = item[0]
					final_version.append(to_append)

				inner_product_lst = final_version[:]
				new_order_term = []
				for inner_product in inner_product_lst:
					boolean = True #Keep this term when converting to normal coordinates
					for i in range(0, len(inner_product[1])):
						if inner_product[1][i] == "J" and inner_product[1][i-1] != "D":
							boolean = False
							break
					if boolean == True:	
						new_order_term.append(inner_product)
				if len(new_order_term) > 0:
					taylor_length += 1
					taylor_expansion = new_order_term[:]
			if j % 2 == 0:
				riemann_products = []
				taylor_copy = deepcopy(taylor_expansion)
				for term in taylor_copy:
					term[0] = Fraction(term[0], factorial(j+2))
					riemann_lst = []
					R_count = [term[1].count("D"*k+ "R") for k in range(0, j+1)] #[R,DR,DDR,...]
					for k in range(len(R_count)-1,0,-1):
						for l in range(0,k):
							R_count[l] -= R_count[k]
					indices_to_use = [k for k in range(1, j+3)] #Number of indices equal to order of the expansion plus 2 (to account for two indices from DJDJ)
					for k in range(len(R_count)-1,-1,-1):
						if R_count[k] != 0:
							for l in range(0, R_count[k]):
								riemann_lst.append([0 for m in range(0,4+k)])
					indices_to_use_copy = indices_to_use[:]
					if len(riemann_lst) == 1: #Allocated indices for when we only have one Riemann tensor
						riemann_lst[0][0] = indices_to_use_copy[0]
						riemann_lst[0][2] = indices_to_use_copy[1]
						indices_to_use_copy = indices_to_use_copy[2:]
						for k in range(0, len(riemann_lst[0])):
							if riemann_lst[0][k] == 0:
								riemann_lst[0][k] = indices_to_use_copy[0]
								indices_to_use_copy = indices_to_use_copy[1:]
						riemann_products.append(riemann_lst)
					else: #Allocated indices for when we have a product of Riemann tensors
						for riemann in riemann_lst: #Fills in the first index of each Riemann tensor with an index
							riemann[0] = indices_to_use_copy[0]
							indices_to_use_copy = indices_to_use_copy[1:]
						for riemann in riemann_lst: #Fills in the second (and third if room) indices of each Riemann tensor
							if len(indices_to_use_copy) >= 2:
								riemann[1] = indices_to_use_copy[0]
								riemann[2] = indices_to_use_copy[1]
								indices_to_use_copy = indices_to_use_copy[2:]
							elif len(indices_to_use_copy) == 1:
								riemann[1] = indices_to_use_copy[0]
								indices_to_use_copy = indices_to_use_copy[1:]
						counter = 0
						for index in indices_to_use_copy: #Allocates the remainder of indices that can be differentiated (correspond to one of the "x"'s in the Taylor expansion)
							for k in range(0, len(riemann_lst)):
								if len(riemann_lst[counter]) > 4:
									if 0 in riemann_lst[counter][4:]:
										zero_index = riemann_lst[counter][4:].index(0) + 4
										riemann_lst[counter][zero_index] = index
										counter = (counter + 1) % len(riemann_lst)
										break
								counter = (counter + 1) % len(riemann_lst)
						index = max(indices_to_use)+1
						zero_count = 0
						for k in range(0, len(riemann_lst)):
							zero_count += riemann_lst[k].count(0)
						zero_count = int(zero_count/2)
						index_first_pass = max(indices_to_use) + 1
						index_second_pass = max(indices_to_use) + 1
						for riemann in riemann_lst: #We allocate dummy indices once we have run out of indices corresponding to the "x"'s in the Taylor expansion
							if zero_count == 0:
								break
							else:
								if 0 in riemann:
									zero_index = riemann.index(0)
									riemann[zero_index] = index_first_pass
									index_first_pass += 1
									zero_count -= 1
						for riemann in riemann_lst:
							riemann_zero_count = riemann.count(0)
							for k in range(0, riemann_zero_count):
								zero_index = riemann.index(0)
								riemann[zero_index] = index_second_pass
								index_second_pass += 1
						riemann_products.append(riemann_lst)
				for riemann_lst in riemann_products: #Accounts for which indices can be changed by differentiating (important) and which cannot (dummy)
					for k in range(0, len(riemann_lst)):
						riemann_object = Riemann(riemann_lst[k][0], riemann_lst[k][1], riemann_lst[k][2], riemann_lst[k][3], [], [], [], "Riemann")
						for l in range(0, 4):
							if riemann_lst[k][l] > j+2 and riemann_lst[k][l] not in riemann_object.dummy_indices:
								riemann_object.dummy_indices.append(riemann_lst[k][l])
							elif riemann_lst[k][l] <= j+2 and riemann_lst[k][l] > 2:
								riemann_object.important_indices.append(riemann_lst[k][l])
						for l in range(4, len(riemann_lst[k])):
							riemann_object.deriv_lst.append(riemann_lst[k][l])
							if riemann_lst[k][l] > j+2 and riemann_lst[k][l] not in riemann_object.dummy_indices:
								riemann_object.dummy_indices.append(riemann_lst[k][l])
							elif riemann_lst[k][l] <= j+2:
								riemann_object.important_indices.append(riemann_lst[k][l])
						riemann_lst[k] = riemann_object
				taylor_coefficients.append([])
				for taylor in taylor_copy:
					taylor_coefficients[-1].append(taylor[0])
				taylor_copy = riemann_products[:]
				normal_coord_terms = []
				for term in dirac_array[j][0]: #This block fills a list with terms that do not vanish in normal coordinates (those that have a certain number of derivatives equal to the order of the taylor expansion of the metric)
					deriv_indices = []
					normal_term = deepcopy(term)
					normal_term.coefficient = normal_term.coefficient/factorial(term.r_count - 1) #Evaluates the heat contour integral
					temp_metric = []
					temp_b = []
					for metric in normal_term.metric_lst:
						if len(metric.dirac_deriv_lst) == j:
							temp_metric.append(metric)
							for deriv in metric.dirac_deriv_lst:
								deriv_indices.append(deriv.index)	
					for b in normal_term.b_lst:
						if len(b.dirac_deriv_lst) == j-1 or len(b.dirac_deriv_lst) == 0:
							temp_b.append(b)
							for deriv in b.dirac_deriv_lst:
								deriv_indices.append(deriv.index)
					for metric in normal_term.metric_lst: #If a metric is removed (set to I in normal coordinates) we set its indices equal to each other elsewhere in the term 
						if metric.index1 != metric.index2:
							if metric.index1 in deriv_indices:
								if len(temp_metric) != 0:
									temp_metric[0].dirac_deriv_lst[deriv_indices.index(metric.index1)].index = metric.index2
								if len(temp_b) != 0:
									temp_b[0].dirac_deriv_lst[deriv_indices.index(metric.index1)].index = metric.index2
							if metric.index2 in deriv_indices:
								if len(temp_metric) != 0:
									temp_metric[0].dirac_deriv_lst[deriv_indices.index(metric.index2)].index = metric.index1
								if len(temp_b) != 0:
									temp_b[0].dirac_deriv_lst[deriv_indices.index(metric.index2)].index = metric.index1
					normal_term.metric_lst = temp_metric[:]			
					normal_term.b_lst = temp_b[:]
					normal_coord_terms.append(normal_term)
				new_term_lst = []
				for term in normal_coord_terms:
					xi_indices = [xi.index for xi in term.xi_lst]
					term.noneq_indices = []
					even_partition_lst = list(accel_asc_even(len(xi_indices))) #Integer partitions are generated such that each element of the partition is a multiple of 2 so the Gaussians do not vanish
					for partition in even_partition_lst:#Evaluation of the Gaussians
						new_term = deepcopy(term)	
						for number in partition: 
							new_term.coefficient *= Fraction(int(factorial2(number - 1))/2**(number/2))
						if len(new_term.metric_lst) != 0: #These next few blocks of code account for different index labeling among metric terms with the same partition
							if len(new_term.metric_lst[0].dirac_deriv_lst) % 4 != 0:
								new_term.coefficient *= -1
							deriv_indices_to_change = []
							other_deriv_indices = []
							for dirac in new_term.metric_lst[0].dirac_deriv_lst:
								if dirac.index in xi_indices:
									deriv_indices_to_change.append(dirac.index) #Tracks which derivative indices will be changed when evaluating Gaussians
								else:
									other_deriv_indices.append(dirac.index) #Tracks which derivative indices will not be changed when evaluating Gaussians
							deriv_indices_to_change_unique = list(set(deriv_indices_to_change))
							other_deriv_indices_unique = list(set(other_deriv_indices))
							if len(partition) > 1:
								for k in range(len(other_deriv_indices_unique)+1, len(partition)+len(other_deriv_indices_unique)+1):
									new_term.noneq_indices.append(k)
							for k in range(0, len(other_deriv_indices)):
								other_deriv_indices[k] = other_deriv_indices_unique.index(other_deriv_indices[k])+1
							#Sets the metric derivatives to be the maximum out of all indices (to help when we evaluate the gaussian integrals since we will count down to do this)
							new_term.metric_lst[0].index1 = len(partition) + len(other_deriv_indices_unique)
							new_term.metric_lst[0].index2 = len(partition) + len(other_deriv_indices_unique)
							partition_copy = deepcopy(partition)
							partition_copy = list(reversed(partition_copy))
							partition_copy[-1] -= 2 #Since setting the metric indices equal to each other removes two indices from consideration
							for k in range(0, len(other_deriv_indices)): #Handles the indices for the derivatives that will not change when evaluating the gaussians
								new_term.metric_lst[0].dirac_deriv_lst[k].index = other_deriv_indices[k]
							counter = len(other_deriv_indices) #Starts changing derivative indices just after the ones that are not suppose to be changed
							if len(partition) == 1: #A partition length of 1 means that we set all indices (other than stationary derivative indices) to be equal to each other
								for k in range(len(other_deriv_indices), len(new_term.metric_lst[0].dirac_deriv_lst)):
									new_term.metric_lst[0].dirac_deriv_lst[k].index = new_term.metric_lst[0].index1
								new_term_lst.append(deepcopy(new_term))
							else:
								for k in range(0, len(partition)): #This for loop makes sure all the derivative indices correspond to which indices are allowed to equal or not
									if partition_copy[k] != 0:
										for l in range(0, partition_copy[k]):
											new_term.metric_lst[0].dirac_deriv_lst[counter].index = new_term.noneq_indices[k]
											counter += 1
								new_term_lst.append(deepcopy(new_term))
								dirac_lst = []
								for k in range(0, len(new_term.metric_lst[0].dirac_deriv_lst)):
									dirac_lst.append(new_term.metric_lst[0].dirac_deriv_lst[k].index)
								for k in range(len(new_term.noneq_indices)-1,0,-1): #Handles the case when the metric indices are not equal 
									new_term_copy = deepcopy(new_term)
									new_term_copy.noneq_indices = sorted(new_term_copy.noneq_indices)
									new_term_copy.coefficient *= 2 #Since we have two of the terms but by symmetry of the metric they are equal
									#We now iterate over all possible combinations of indices. This start the process in each iteration by making the first metric index one greater than the second index
									new_term_copy.metric_lst[0].index1 = new_term_copy.noneq_indices[k]
									new_term_copy.metric_lst[0].index2 = new_term_copy.noneq_indices[k-1]									#Changes derivative indices appropriately
									if new_term_copy.metric_lst[0].index1 != len(partition) + len(other_deriv_indices_unique):
										new_term_copy.metric_lst[0].dirac_deriv_lst[dirac_lst.index(new_term_copy.metric_lst[0].index1)].index = len(partition) + len(other_deriv_indices_unique)
									if new_term_copy.metric_lst[0].index2 != len(partition) + len(other_deriv_indices_unique):
										new_term_copy.metric_lst[0].dirac_deriv_lst[dirac_lst.index(new_term_copy.metric_lst[0].index2)].index = len(partition) + len(other_deriv_indices_unique)
									new_term_lst.append(deepcopy(new_term_copy))
									new_term_copy.metric_lst[0].dirac_deriv_lst[dirac_lst.index(new_term_copy.metric_lst[0].index2)].index = new_term_copy.metric_lst[0].dirac_deriv_lst[dirac_lst.index(new_term_copy.metric_lst[0].index2)+1].index 
									
									for l in range(k-2,-1,-1): #Decreases the second metric index appropriately
										new_term_copy.metric_lst[0].index2 = new_term_copy.noneq_indices[l]
										if new_term_copy.metric_lst[0].index2 != len(partition) + len(other_deriv_indices_unique):
											new_term_copy.metric_lst[0].dirac_deriv_lst[dirac_lst.index(new_term_copy.metric_lst[0].index2)].index = len(partition) + len(other_deriv_indices_unique)
										new_term_lst.append(deepcopy(new_term_copy))
										new_term_copy.metric_lst[0].dirac_deriv_lst[dirac_lst.index(new_term_copy.metric_lst[0].index2)].index = new_term_copy.metric_lst[0].dirac_deriv_lst[dirac_lst.index(new_term_copy.metric_lst[0].index2)+1].index 
						if len(new_term.b_lst) != 0:
							if len(new_term.b_lst) > 1: #Handles the case when the term has a product of b's
								index = 1
								new_term.b_lst = []
								for number in partition:
									for k in range(0, number): #Partitions the product of constants according to the gaussian integral
										new_term.constant_lst.append(Constant(index,[]))
									new_term.noneq_indices.append(index)
									index += 1
									if number % 4 != 0: #Since each constant is multiplied by i (the imaginary unit)
										new_term.coefficient *= -1
								new_term_lst.append(new_term)
							else: #Does the same process as the metrics with some slight changes (see below)
								deriv_indices_to_change = []
								other_deriv_indices = []
								for dirac in new_term.b_lst[0].dirac_deriv_lst:
									if dirac.index in xi_indices:
										deriv_indices_to_change.append(dirac.index)
									else:
										other_deriv_indices.append(dirac.index)
								deriv_indices_to_change_unique = list(set(deriv_indices_to_change))
								other_deriv_indices_unique = list(set(other_deriv_indices))
								if len(partition) > 1:
									for k in range(len(other_deriv_indices_unique)+1, len(partition)+len(other_deriv_indices_unique)+1):
										new_term.noneq_indices.append(k)
								for k in range(0, len(other_deriv_indices)):
									other_deriv_indices[k] = other_deriv_indices_unique.index(other_deriv_indices[k])+1
								new_term.b_lst[0].index = len(partition) + len(other_deriv_indices_unique)
								partition_copy = partition[:]
								partition_copy = list(reversed(partition_copy))
								partition_copy[-1] -= 1 #Since the b object only has one index that changes
								for k in range(0, len(other_deriv_indices)):
									new_term.b_lst[0].dirac_deriv_lst[k].index = other_deriv_indices[k]
								counter = len(other_deriv_indices)
								if len(partition) == 1: #We split into 3 terms by definition of b in normal coordinates
									for k in range(len(other_deriv_indices), len(new_term.b_lst[0].dirac_deriv_lst)):
										new_term.b_lst[0].dirac_deriv_lst[k].index = new_term.b_lst[0].index							
									metric1_term = Term(-new_term.coefficient/2, 0, [Metric(len(partition) + len(other_deriv_indices_unique)+1, len(partition) + len(other_deriv_indices_unique)+1, new_term.b_lst[0].dirac_deriv_lst[:] + [Dirac_deriv(new_term.b_lst[0].index)])], [], [], [], new_term.noneq_indices, [])
									if len(metric1_term.metric_lst[0].dirac_deriv_lst) % 4 != 0:
										metric1_term.coefficient *= -1
									new_term_lst.append(deepcopy(metric1_term))
									metric2_term = Term(new_term.coefficient, 0, [Metric(new_term.b_lst[0].index, len(partition) + len(other_deriv_indices_unique)+1, new_term.b_lst[0].dirac_deriv_lst[:] + [Dirac_deriv(len(partition) + len(other_deriv_indices_unique)+1)])], [], [], [], new_term.noneq_indices, [])
									if len(metric2_term.metric_lst[0].dirac_deriv_lst) % 4 != 0:
										metric2_term.coefficient *= -1
									new_term_lst.append(deepcopy(metric2_term))
									constant_term = Term(new_term.coefficient, 0, [], [], [], [], new_term.noneq_indices, [Constant(new_term.b_lst[0].index,new_term.b_lst[0].dirac_deriv_lst[:])])
									constant_term.coefficient *= (-1)**len(constant_term.constant_lst[0].dirac_deriv_lst)
									if (len(constant_term.constant_lst[0].dirac_deriv_lst) + 1) % 4 != 0:
										constant_term.coefficient *= -1
									new_term_lst.append(deepcopy(constant_term))
								else:
									for k in range(0, len(partition)):
										if partition_copy[k] != 0:
											for l in range(0, partition_copy[k]):
												new_term.b_lst[0].dirac_deriv_lst[counter].index = new_term.noneq_indices[k]
												counter += 1
									dirac_lst = []
									for k in range(0, len(new_term.b_lst[0].dirac_deriv_lst)):
										dirac_lst.append(new_term.b_lst[0].dirac_deriv_lst[k].index)
									for k in range(0,len(new_term.noneq_indices)):
										new_term_copy = deepcopy(new_term)
										new_term_copy.b_lst[0].index = new_term_copy.noneq_indices[k]
										if new_term_copy.b_lst[0].index != len(partition) + len(other_deriv_indices_unique):
											new_term_copy.b_lst[0].dirac_deriv_lst[dirac_lst.index(new_term_copy.b_lst[0].index)].index = len(partition) + len(other_deriv_indices_unique)
										metric1_term = Term(-new_term_copy.coefficient/2, 0, [Metric(len(partition) + len(other_deriv_indices_unique)+1, len(partition) + len(other_deriv_indices_unique)+1, new_term_copy.b_lst[0].dirac_deriv_lst[:] + [Dirac_deriv(new_term_copy.b_lst[0].index)])], [], [], [], new_term_copy.noneq_indices, [])
										if len(metric1_term.metric_lst[0].dirac_deriv_lst) % 4 != 0:
											metric1_term.coefficient *= -1
										new_term_lst.append(deepcopy(metric1_term))
										metric2_term = Term(new_term_copy.coefficient, 0, [Metric(new_term_copy.b_lst[0].index, len(partition) + len(other_deriv_indices_unique)+1, new_term_copy.b_lst[0].dirac_deriv_lst[:] + [Dirac_deriv(len(partition) + len(other_deriv_indices_unique)+1)])], [], [], [], new_term_copy.noneq_indices, [])
										if len(metric2_term.metric_lst[0].dirac_deriv_lst) % 4 != 0:
											metric2_term.coefficient *= -1
										new_term_lst.append(deepcopy(metric2_term))
										constant_term = Term(new_term_copy.coefficient, 0, [], [], [], [], new_term_copy.noneq_indices, [Constant(new_term_copy.b_lst[0].index,new_term_copy.b_lst[0].dirac_deriv_lst[:])])
										constant_term.coefficient *= (-1)**len(constant_term.constant_lst[0].dirac_deriv_lst)
										if (len(constant_term.constant_lst[0].dirac_deriv_lst) + 1) % 4 != 0:
											constant_term.coefficient *= -1
										new_term_lst.append(deepcopy(constant_term))
				expanded_sums = []
				for term in new_term_lst: #We split up the sums to get some cancellations
					term.noneq_indices = sorted(term.noneq_indices)
					reference_indices = []
					if len(term.metric_lst) != 0:
						for dirac in term.metric_lst[0].dirac_deriv_lst:
							reference_indices.append(dirac.index)
						reference_indices.append(term.metric_lst[0].index1)
						reference_indices.append(term.metric_lst[0].index2)
						reference_unique = list(set(reference_indices))
						if len(reference_unique) == 1:
							expanded_sums.append(deepcopy(term))
						elif len(reference_unique) == len(term.noneq_indices):
							expanded_sums.append(deepcopy(term))
						else:
							if len(term.noneq_indices) == 0: #Sets all indices equal to each other
								new_term = deepcopy(term)
								for dirac in new_term.metric_lst[0].dirac_deriv_lst:
									dirac.index = 1
								new_term.metric_lst[0].index1 = 1
								new_term.metric_lst[0].index2 = 1
								expanded_sums.append(deepcopy(new_term))	
							else:
								for noneq in term.noneq_indices: #Sets all indices equal to each other subject to the noneqal indices list
									new_term = deepcopy(term)
									for k in range(0, len(reference_indices)-2):
										if reference_indices[k] not in new_term.noneq_indices:
											new_term.metric_lst[0].dirac_deriv_lst[k].index = noneq
									if reference_indices[-2] not in new_term.noneq_indices:
										new_term.metric_lst[0].index1 = noneq
									if reference_indices[-1] not in new_term.noneq_indices:
										new_term.metric_lst[0].index2 = noneq
									expanded_sums.append(deepcopy(new_term))
							index_combinations = []
							combination_references = [0,0]
							for k in range(2, len(reference_unique)+1): #Generates combinations of nonequal indices to use
								combinations_to_add = list(combinations(reference_unique, k))
								combination_references.append(len(combinations_to_add))
								for combination in combinations_to_add:
									index_combinations.append(list(combination))
							start = combination_references[len(term.noneq_indices)] #Starts looking at combinations of nonequal indices of length greater than the current one the term has
							for k in range(start, len(index_combinations)):
								if set(term.noneq_indices).issubset(set(index_combinations[k])): #Represents adding another index to be not equal that was not already present in the constraint
									new_term = deepcopy(term)
									new_term.noneq_indices = index_combinations[k][:]
									if len(new_term.noneq_indices) == len(reference_unique):
										expanded_sums.append(deepcopy(new_term))
									else:
										for noneq in new_term.noneq_indices:
											new_term_copy = deepcopy(new_term)
											for l in range(0, len(reference_indices)-2):
												if reference_indices[l] not in new_term_copy.noneq_indices:
													new_term_copy.metric_lst[0].dirac_deriv_lst[l].index = noneq
											if reference_indices[-2] not in new_term_copy.noneq_indices:
												new_term_copy.metric_lst[0].index1 = noneq
											if reference_indices[-1] not in new_term_copy.noneq_indices:
												new_term_copy.metric_lst[0].index2 = noneq
											expanded_sums.append(deepcopy(new_term_copy))
					if len(term.constant_lst) == 1: #Does the same thing for splitting summations but for constant objects
						for dirac in term.constant_lst[0].dirac_deriv_lst:
							reference_indices.append(dirac.index)
						reference_indices.append(term.constant_lst[0].index)
						reference_unique = list(set(reference_indices))
						if len(reference_unique) == 1:
							expanded_sums.append(deepcopy(term))
						elif len(reference_unique) == len(term.noneq_indices):
							expanded_sums.append(deepcopy(term))
						else:
							if len(term.noneq_indices) == 0:
								new_term = deepcopy(term)
								for dirac in new_term.constant_lst[0].dirac_deriv_lst:
									dirac.index = 1
								new_term.constant_lst[0].index = 1
								expanded_sums.append(deepcopy(new_term))	
							else:
								for noneq in term.noneq_indices:
									new_term = deepcopy(term)
									for k in range(0, len(reference_indices)-2):
										if reference_indices[k] not in new_term.noneq_indices:
											new_term.constant_lst[0].dirac_deriv_lst[k].index = noneq
									if reference_indices[-1] not in new_term.noneq_indices:
										new_term.constant_lst[0].index = noneq
									expanded_sums.append(deepcopy(new_term))
							index_combinations = []
							combination_references = [0,0]
							for k in range(2, len(reference_unique)+1):
								combinations_to_add = list(combinations(reference_unique, k))
								combination_references.append(len(combinations_to_add))
								for combination in combinations_to_add:
									index_combinations.append(list(combination))
							start = combination_references[len(term.noneq_indices)]
							for k in range(start, len(index_combinations)):
								if set(term.noneq_indices).issubset(set(index_combinations[k])):
									new_term = deepcopy(term)
									new_term.noneq_indices = index_combinations[k][:]
									if len(new_term.noneq_indices) == len(reference_unique):
										expanded_sums.append(deepcopy(new_term))
									else:
										for noneq in new_term.noneq_indices:
											new_term_copy = deepcopy(new_term)
											for l in range(0, len(reference_indices)-2):
												if reference_indices[l] not in new_term_copy.noneq_indices:
													new_term_copy.constant_lst[0].dirac_deriv_lst[l].index = noneq
											if reference_indices[-1] not in new_term_copy.noneq_indices:
												new_term_copy.constant_lst[0].index = noneq
											expanded_sums.append(deepcopy(new_term_copy))
					
					if len(term.constant_lst) > 1:
						expanded_sums.append(deepcopy(term))
					term_counts = []
					temp = []					
					for k in range(0, len(expanded_sums)):
						term_characteristic_lst = [expanded_sums[k].coefficient] #[coefficient, metric present?, metric indices equal?, number of constants?, any nonequal indices?, derivative counts, how many times the metric indices are in the derivative list,index]
						if len(expanded_sums[k].metric_lst) == 1:
							term_characteristic_lst.append(True) #If the term has a metric in it
							if expanded_sums[k].metric_lst[0].index1 != expanded_sums[k].metric_lst[0].index2: 
								term_characteristic_lst.append(False) #If the metric indices are not equal
							else: 
								term_characteristic_lst.append(True) #If the metric indices are equal
							deriv_indices = []
							for dirac in expanded_sums[k].metric_lst[0].dirac_deriv_lst:
								deriv_indices.append(dirac.index)
							deriv_indices = sorted(deriv_indices)
							if expanded_sums[k].metric_lst[0].index1 in deriv_indices:
								term_characteristic_lst.append(True)
							else:
								term_characteristic_lst.append(False)
							if expanded_sums[k].metric_lst[0].index2 in deriv_indices:
								term_characteristic_lst.append(True)
							else:
								term_characteristic_lst.append(False)
							unique_deriv_indices = list(set(deriv_indices))
							deriv_index_counts = [deriv_indices.count(unique_deriv_indices[l]) for l in range(0, len(unique_deriv_indices))]
							term_characteristic_lst.append(sorted(deriv_index_counts))
							metric_deriv_index_counts = [deriv_indices.count(expanded_sums[k].metric_lst[0].index1),deriv_indices.count(expanded_sums[k].metric_lst[0].index2)]
							term_characteristic_lst.append(sorted(metric_deriv_index_counts))
						else:
							term_characteristic_lst.append(False) #If the term does not have a metric in it (and therefore the indices are not equal)
							term_characteristic_lst.append(False) #Metric indices are not equal because no metrics are present
							term_characteristic_lst.append(False) #Metric indices are not present in the derivatives for the same reason as above for this boolean and the next
							term_characteristic_lst.append(False) 
							term_characteristic_lst.append([])
							term_characteristic_lst.append([])
						term_characteristic_lst.append(len(expanded_sums[k].constant_lst))
						term_characteristic_lst.append(len(expanded_sums[k].noneq_indices))
						term_characteristic_lst.append(k)
						term_counts.append(term_characteristic_lst)
					for k in range(0, len(expanded_sums)): #Combines like terms
						boolean = False
						for item in temp:							
							if term_counts[k][1:9] == item[1:9]:
								item[0] += term_counts[k][0]
								boolean = True
						if boolean == False:
							temp.append(term_counts[k])
					split_sums = []
					for item in temp:
						index = item[-1]
						new_term = deepcopy(expanded_sums[index])
						new_term.coefficient = item[0]
						if new_term.coefficient != 0:
							split_sums.append(new_term)
				taylor_term = deepcopy(taylor_copy)
				riemann_terms = []
				product_rule_terms = []
				for term in split_sums: #Relabels indices for convenience
					if len(term.metric_lst) >0:
						indices_to_change = []
						if term.metric_lst[0].index1 not in indices_to_change:
							indices_to_change.append(term.metric_lst[0].index1)
						if term.metric_lst[0].index2 not in indices_to_change:
							indices_to_change.append(term.metric_lst[0].index2)
						for dirac in term.metric_lst[0].dirac_deriv_lst:
							if dirac.index not in indices_to_change:
								indices_to_change.append(dirac.index)
						reference_indices = [l for l in range(1, len(indices_to_change)+1)]
						for dirac in term.metric_lst[0].dirac_deriv_lst:
							dirac.index = reference_indices[indices_to_change.index(dirac.index)]
						term.metric_lst[0].index1 = reference_indices[indices_to_change.index(term.metric_lst[0].index1)]
						term.metric_lst[0].index2 = reference_indices[indices_to_change.index(term.metric_lst[0].index2)]
						for k in range(0, len(term.noneq_indices)):
							term.noneq_indices[k] = k + 1
				
				for term in split_sums: #Substitutes Riemann tensors for metric tensors and allocates indices accordingly
					if len(term.constant_lst) > 0:
						riemann_terms.append(deepcopy(term))
					else:
						for k in range(0, len(taylor_term)):
							taylor_term[k][0].index1 = str(term.metric_lst[0].index1)
							if len(taylor_term[k]) == 1: #for a term with a single Riemann tensor we have g^ij -> R_{i*j*}					
								taylor_term[k][0].index3 = str(term.metric_lst[0].index2)
							else: #For a term with more than one Riemann tensor we have g^ij -> R_{i***}R_{j***}
								taylor_term[k][1].index1 = str(term.metric_lst[0].index2)
							temp1 = [deepcopy(taylor_term[k])]
							temp2 = []
							temp_dirac_lst = []
							for dirac in term.metric_lst[0].dirac_deriv_lst:
								temp_dirac_lst.append(int(dirac.index))								
								for riemann_product in temp1:						
									for riemann in riemann_product:									
										for index in riemann.important_indices:											
											riemann_copy = deepcopy(riemann)
											if riemann_copy.index1 == index:
												riemann_copy.index1 = str(dirac.index)
											elif riemann_copy.index2 == index:
												riemann_copy.index2 = str(dirac.index)
											elif riemann_copy.index3 == index:
												riemann_copy.index3 = str(dirac.index)
											elif riemann_copy.index4 == index:
												riemann_copy.index4 = str(dirac.index)
											elif index in riemann_copy.deriv_lst:
												deriv_index = riemann_copy.deriv_lst.index(index)
												riemann_copy.deriv_lst[deriv_index] = str(dirac.index)
											riemann_copy.important_indices.remove(index)
											riemann_product_copy = deepcopy(riemann_product)
											riemann_product_copy[riemann_product.index(riemann)] = deepcopy(riemann_copy)											
											temp2.append(deepcopy(riemann_product_copy))
								temp1 = temp2[:]
								temp2 = []
							for riemanns in temp1:
								term_copy = deepcopy(term)
								term_copy.metric_lst = deepcopy(riemanns)
								term_copy.coefficient *= -taylor_coefficients[int((j-2)/2)][k]
								product_rule_terms.append(deepcopy(term_copy))
				to_delete = []
				to_add = []
				for term in product_rule_terms:
					splitting_indices = []
					for riemann in term.metric_lst:
						if int(riemann.index1) not in term.noneq_indices and int(riemann.index1) not in splitting_indices:
							splitting_indices.append(int(riemann.index1))
						if int(riemann.index2) not in term.noneq_indices and int(riemann.index2) not in splitting_indices:
							splitting_indices.append(int(riemann.index2))
						if int(riemann.index3) not in term.noneq_indices and int(riemann.index3) not in splitting_indices:
							splitting_indices.append(int(riemann.index3))
						if int(riemann.index4) not in term.noneq_indices and int(riemann.index4) not in splitting_indices:
							splitting_indices.append(int(riemann.index4))
						for deriv in riemann.deriv_lst:
							if int(deriv) not in term.noneq_indices and int(deriv) not in splitting_indices:
								splitting_indices.append(int(deriv))
					if len(splitting_indices) > 0:
						splitting_indices = sorted(splitting_indices)
						to_delete.append(product_rule_terms.index(term))
						temp1 = [deepcopy(term)]
						temp2 = []
						for split in splitting_indices:
							for item in temp1:
								for k in range(0, len(item.noneq_indices)+1):
									term_copy = deepcopy(item)
									for riemann in term_copy.metric_lst:
										if int(riemann.index1) == split:
											riemann.index1 = k+1
										if int(riemann.index2) == split:
											riemann.index2 = k+1
										if int(riemann.index3) == split:
											riemann.index3 = k+1
										if int(riemann.index4) == split:
											riemann.index4 = k+1
										for l in range(0, len(riemann.deriv_lst)):
											if int(riemann.deriv_lst[l]) == split:
												riemann.deriv_lst[l] = k+1
										if k+1 not in term_copy.noneq_indices:
											term_copy.noneq_indices.append(k+1)
									temp2.append(deepcopy(term_copy))
							temp1 = deepcopy(temp2)
							temp2 = []
						to_add = to_add + deepcopy(temp1)
				for delete in sorted(to_delete, reverse = True):
					del product_rule_terms[delete]
				product_rule_terms = product_rule_terms + to_add
				term_counts = []
				for k in range(0, len(product_rule_terms)): #[coefficient, number of nonequal indices, number of derivatives in each riemann tensor and their placement, riemann tensor indices, index in product_rule_terms]
					term_characteristic_lst = [product_rule_terms[k].coefficient, len(product_rule_terms[k].noneq_indices)]
					riemann_counts = [0 for l in range(0, j-1)] #[0 derivatives, 1 derivative, 2 derivatives, ..., j-2 derivatives]
					riemann_index_lst = [] 
					deriv_info_lst = []
					for riemann in product_rule_terms[k].metric_lst:
						riemann_counts[len(riemann.deriv_lst)] += 1
						indices = [int(riemann.index1),int(riemann.index2),int(riemann.index3),int(riemann.index4)]	
						#We impose an ordering on the indices of the Riemann tensor to make it easier to compare them when adding like terms
						if indices[0] > indices[1]:
							temp = indices[0]
							indices[0] = indices[1]
							indices[1] = temp
							term_characteristic_lst[0] *= -1
						if indices[2] > indices[3]:	
							temp = indices[2]
							indices[2] = indices[3]
							indices[3] = temp
							term_characteristic_lst[0] *= -1
						deriv_indices_lst = [int(deriv) for deriv in riemann.deriv_lst]
						indices = deepcopy(indices + sorted(deepcopy(deriv_indices_lst)))
						riemann_index_lst.append(indices[:])
					term_characteristic_lst.append(riemann_counts[:])
					term_characteristic_lst.append(sorted(riemann_index_lst[:]))
					term_characteristic_lst.append(k) #Keeps track of the terms position in the product_rule_terms list
					to_add_boolean = True #If this is true by the end of the comparison then this will be added to the term_counts list. If it is False by the end of the comparison then the term will be summed with the term in term_counts.
					if len(term_counts) > 0:
						if j == 2: #Applies the Bianchi identity
							if int(product_rule_terms[term_characteristic_lst[-1]].metric_lst[0].index1) == int(product_rule_terms[term_characteristic_lst[-1]].metric_lst[0].index2):
								if int(product_rule_terms[term_characteristic_lst[-1]].metric_lst[0].index3) == int(product_rule_terms[term_characteristic_lst[-1]].metric_lst[0].index4):
									for k in range(0, len(term_counts)):
										if int(product_rule_terms[term_counts[k][-1]].metric_lst[0].index1) == int(product_rule_terms[term_counts[k][-1]].metric_lst[0].index4):
											if int(product_rule_terms[term_counts[k][-1]].metric_lst[0].index2) == int(product_rule_terms[term_counts[k][-1]].metric_lst[0].index3):
												to_add_boolean = False
												del term_counts[k]											
												for l in range(0, len(term_counts)):
													if int(product_rule_terms[term_counts[l][-1]].metric_lst[0].index1) == int(product_rule_terms[term_counts[l][-1]].metric_lst[0].index3):
														if int(product_rule_terms[term_counts[l][-1]].metric_lst[0].index2) == int(product_rule_terms[term_counts[l][-1]].metric_lst[0].index4):
															term_counts[l][0] += term_characteristic_lst[0]
															break
												break
							if int(product_rule_terms[term_characteristic_lst[-1]].metric_lst[0].index1) == int(product_rule_terms[term_characteristic_lst[-1]].metric_lst[0].index4):
								if int(product_rule_terms[term_characteristic_lst[-1]].metric_lst[0].index2) == int(product_rule_terms[term_characteristic_lst[-1]].metric_lst[0].index3):
									for l in range(0, len(term_counts)):
										if int(product_rule_terms[term_counts[l][-1]].metric_lst[0].index1) == int(product_rule_terms[term_counts[l][-1]].metric_lst[0].index2):
											if int(product_rule_terms[term_counts[l][-1]].metric_lst[0].index3) == int(product_rule_terms[term_counts[l][-1]].metric_lst[0].index4):
												to_add_boolean = False
												del term_counts[l]
												for m in range(0, len(term_counts)):
													if int(product_rule_terms[term_counts[m][-1]].metric_lst[0].index1) == int(product_rule_terms[term_counts[m][-1]].metric_lst[0].index3):
														if int(product_rule_terms[term_counts[m][-1]].metric_lst[0].index2) == int(product_rule_terms[term_counts[m][-1]].metric_lst[0].index4):
															term_counts[m][0] += term_characteristic_lst[0]
															break
												break
						for l in range(0, len(term_counts)):
							if term_counts[l][1] == term_characteristic_lst[1] and to_add_boolean != False:
								if term_counts[l][2] == term_characteristic_lst[2]:
									if len(term_counts[l][3]) == len(term_characteristic_lst[3]):																			
										term_count_copy = deepcopy(term_counts[l][3])
										for m in range(0, len(term_characteristic_lst[3])):
											if term_characteristic_lst[3][m] in term_count_copy: #Tests if the Riemann tensor is present as is
												del term_count_copy[term_count_copy.index(term_characteristic_lst[3][m])]
												if len(term_count_copy) == 0:
													term_counts[l][0] += term_characteristic_lst[0]
													to_add_boolean = False
													break
											else:
												term_characteristic_copy = deepcopy(term_characteristic_lst[3][m])
												temp = [term_characteristic_copy[0], term_characteristic_copy[1]] #Changes R_{abcd} to R_{cdab}
												term_characteristic_copy[0] = term_characteristic_copy[2]
												term_characteristic_copy[1] = term_characteristic_copy[3]
												term_characteristic_copy[2] = temp[0]
												term_characteristic_copy[3] = temp[1]
												if term_characteristic_copy in term_count_copy: #Tests if the Riemann tensor is present after sending R_{abcd} to R_{cdab}
													del term_count_copy[term_count_copy.index(term_characteristic_copy)]
													if len(term_count_copy) == 0:
														term_counts[l][0] += term_characteristic_lst[0]
														to_add_boolean = False
														break
										if to_add_boolean == True: #This happens if we need to relabel some indices
											for n in range(0, len(product_rule_terms[k].noneq_indices)-1):
												for o in range(n+1, len(product_rule_terms[k].noneq_indices)):
													counter = 0 #Some indices will be swapped and this will keep track of the sign changes
													term_count_backup = deepcopy(term_counts[l][3])
													term_characteristic_backup = deepcopy(term_characteristic_lst[3])
													
													for count in term_characteristic_backup: #We now relabel certain indices using the nonequal indices as a reference
														for p in range(0, len(count)):
															if count[p] == product_rule_terms[k].noneq_indices[n]:
																count[p] = product_rule_terms[k].noneq_indices[o]
															elif count[p] == product_rule_terms[k].noneq_indices[o]:
																count[p] = product_rule_terms[k].noneq_indices[n]														
														if count[0] > count[1]: #This and the next if statement impose an ordering on the indices to make comparing Riemann tensors easier
															temp = count[0]
															count[0] = count[1]
															count[1] = temp
															counter += 1
														if count[2] > count[3]:																	
															temp = count[2]
															count[2] = count[3]
															count[3] = temp	
															counter += 1
														count[4:len(count)] = sorted(count[4:len(count)])
																			
													for count in term_characteristic_backup:														
														if count in term_count_backup:																																									
															del term_count_backup[term_count_backup.index(count)]
															if len(term_count_backup) == 0:																		
																if counter % 2 == 0:
																	term_counts[l][0] += term_characteristic_lst[0]
																else:
																	term_counts[l][0] -= term_characteristic_lst[0]
																to_add_boolean = False
																break
														else: #If R_{abcd} is not present we now check if R_{cdab} is																
															temp = [count[0], count[1]]
															count[0] = count[2]
															count[1] = count[3]
															count[2] = temp[0]
															count[3] = temp[1]
															if count in term_count_backup:																			
																del term_count_backup[term_count_backup.index(count)]
																if len(term_count_backup) == 0:
																	if counter % 2 == 0:
																		term_counts[l][0] += term_characteristic_lst[0]
																	else:
																		term_counts[l][0] -= term_characteristic_lst[0]
																	to_add_boolean = False
																	break
					if to_add_boolean == True:
						term_counts.append(term_characteristic_lst[:])
				for count in term_counts:
					to_append = product_rule_terms[count[-1]]
					to_append.coefficient = count[0]
					if to_append.coefficient != 0:
						to_append.coefficient *= factorial(len(to_append.noneq_indices))
						riemann_terms.append(deepcopy(to_append))				
				normal_coord_output.append(riemann_terms[:])
	return normal_coord_output

def generate_coefficient_latex(normal_coord_output):
	latex_doc = open("coefficient_output.tex","w+")
	latex_doc.write("\\documentclass{article}"+"\n")
	latex_doc.write("\\usepackage[utf8]{inputenc}" + "\n" + "\\usepackage{amsmath}"+"\n" + "\\usepackage{amsfonts}"+ "\n" + "\\title{Recursion Output}"+"\n" + "\\date{}"+"\n" + "\\begin{document}"+"\n" +"\\maketitle"+"\n" + "\\setcounter{secnumdepth}{0}"+"\n" + "\\section{}"+"\n")

	new_line = "\\begin{align*}"+"\n"+"h_{2}=Vol_g(M)"
	latex_doc.write(new_line+"\n"+"\\end{align*}"+"\n")
	for j in range(0, len(normal_coord_output)):
		new_line = "\\newpage" + "\n"
		line_counter = 0
		new_line += "\\begin{align*}" + "\n" + "\\left(4\\pi\\right)^{\\frac{n}{2}}h_{" + str(2*(j+1)+2) + "}="
		for term in normal_coord_output[j]:
			riemann_str = ""
			if len(term.metric_lst) > 0:
				for riemann in term.metric_lst:
					riemann.deriv_lst = [int(deriv) for deriv in riemann.deriv_lst]
					if int(riemann.index1) != int(riemann.index2) and int(riemann.index3) != int(riemann.index4):
						if int(riemann.index1) == int(riemann.index3) and int(riemann.index1) not in riemann.deriv_lst:
							riemann.tensor_type = "Ricci24"
							for k in range(0, len(term.metric_lst)):
								if k != term.metric_lst.index(riemann):
									if int(riemann.index1) in term.metric_lst[k].deriv_lst:
										riemann.tensor_type = "Riemann"
									elif int(riemann.index1) == int(term.metric_lst[k].index1):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index1) == int(term.metric_lst[k].index2):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index1) == int(term.metric_lst[k].index3):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index1) == int(term.metric_lst[k].index4):
										riemann.tensor_type = "Riemann"
								if riemann.tensor_type == "Riemann":
									break
							if int(riemann.index2) == int(riemann.index4) and int(riemann.index2) not in riemann.deriv_lst:
								if riemann.tensor_type == "Ricci24":
									riemann.tensor_type = "Scalar-"
									for k in range(0, len(term.metric_lst)):
										if k != term.metric_lst.index(riemann):
											if int(riemann.index2) in term.metric_lst[k].deriv_lst:
												riemann.tensor_type = "Ricci24"
											elif int(riemann.index2) == int(term.metric_lst[k].index1):
												riemann.tensor_type = "Ricci24"
											elif int(riemann.index2) == int(term.metric_lst[k].index2):
												riemann.tensor_type = "Ricci24"
											elif int(riemann.index2) == int(term.metric_lst[k].index3):
												riemann.tensor_type = "Ricci24"
											elif int(riemann.index2) == int(term.metric_lst[k].index4):
												riemann.tensor_type = "Ricci24"
										if riemann.tensor_type == "Ricci24":
											break
								else:
									riemann.tensor_type = "Ricci13"
									for k in range(0, len(term.metric_lst)):
										if k != term.metric_lst.index(riemann):
											if int(riemann.index2) in term.metric_lst[k].deriv_lst:
												riemann.tensor_type = "Riemann"
											elif int(riemann.index2) == int(term.metric_lst[k].index1):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index2) == int(term.metric_lst[k].index2):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index2) == int(term.metric_lst[k].index3):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index2) == int(term.metric_lst[k].index4):
												riemann.tensor_type = "Riemann"
										if riemann.tensor_type == "Riemann":
											break
						elif int(riemann.index1) == int(riemann.index4) and int(riemann.index1) not in riemann.deriv_lst:
							riemann.tensor_type = "Ricci23"
							for k in range(0, len(term.metric_lst)):
								if k != term.metric_lst.index(riemann):
									if int(riemann.index1) in term.metric_lst[k].deriv_lst:
										riemann.tensor_type = "Riemann"
									elif int(riemann.index1) == int(term.metric_lst[k].index1):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index1) == int(term.metric_lst[k].index2):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index1) == int(term.metric_lst[k].index3):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index1) == int(term.metric_lst[k].index4):
										riemann.tensor_type = "Riemann"
								if riemann.tensor_type == "Riemann":
									break
							if int(riemann.index2) == int(riemann.index3) and int(riemann.index2) not in riemann.deriv_lst:
								if riemann.tensor_type == "Ricci23":
									riemann.tensor_type = "Scalar+"
									for k in range(0, len(term.metric_lst)):
										if k != term.metric_lst.index(riemann):
											if int(riemann.index2) in term.metric_lst[k].deriv_lst:
												riemann.tensor_type = "Ricci23"
											elif int(riemann.index2) == int(term.metric_lst[k].index1):
												riemann.tensor_type = "Ricci23"
											elif int(riemann.index2) == int(term.metric_lst[k].index2):
												riemann.tensor_type = "Ricci23"
											elif int(riemann.index2) == int(term.metric_lst[k].index3):
												riemann.tensor_type = "Ricci23"
											elif int(riemann.index2) == int(term.metric_lst[k].index4):
												riemann.tensor_type = "Ricci23"
										if riemann.tensor_type == "Ricci23":
											break
								else:
									riemann.tensor_type = "Ricci14"
									for k in range(0, len(term.metric_lst)):
										if k != term.metric_lst.index(riemann):
											if int(riemann.index2) in term.metric_lst[k].deriv_lst:
												riemann.tensor_type = "Riemann"
											elif int(riemann.index2) == int(term.metric_lst[k].index1):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index2) == int(term.metric_lst[k].index2):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index2) == int(term.metric_lst[k].index3):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index2) == int(term.metric_lst[k].index4):
												riemann.tensor_type = "Riemann"
										if riemann.tensor_type == "Riemann":
											break
						elif int(riemann.index2) == int(riemann.index3) and int(riemann.index2) not in riemann.deriv_lst:
							riemann.tensor_type = "Ricci14"
							for k in range(0, len(term.metric_lst)):
								if k != term.metric_lst.index(riemann):
									if int(riemann.index2) in term.metric_lst[k].deriv_lst:
										riemann.tensor_type = "Riemann"
									elif int(riemann.index2) == int(term.metric_lst[k].index1):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index2) == int(term.metric_lst[k].index2):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index2) == int(term.metric_lst[k].index3):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index2) == int(term.metric_lst[k].index4):
										riemann.tensor_type = "Riemann"
								if riemann.tensor_type == "Riemann":
									break
							if int(riemann.index1) == int(riemann.index4) and int(riemann.index1) not in riemann.deriv_lst:
								if riemann.tensor_type == "Ricci14":
									riemann.tensor_type = "Scalar+"
									for k in range(0, len(term.metric_lst)):
										if k != term.metric_lst.index(riemann):
											if int(riemann.index1) in term.metric_lst[k].deriv_lst:
												riemann.tensor_type = "Ricci14"
											elif int(riemann.index1) == int(term.metric_lst[k].index1):
												riemann.tensor_type = "Ricci14"
											elif int(riemann.index1) == int(term.metric_lst[k].index2):
												riemann.tensor_type = "Ricci14"
											elif int(riemann.index1) == int(term.metric_lst[k].index3):
												riemann.tensor_type = "Ricci14"
											elif int(riemann.index1) == int(term.metric_lst[k].index4):
												riemann.tensor_type = "Ricci14"
										if riemann.tensor_type == "Ricci14":
											break
								else:
									riemann.tensor_type = "Ricci23"
									for k in range(0, len(term.metric_lst)):
										if k != term.metric_lst.index(riemann):
											if int(riemann.index1) in term.metric_lst[k].deriv_lst:
												riemann.tensor_type = "Riemann"
											elif int(riemann.index1) == int(term.metric_lst[k].index1):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index1) == int(term.metric_lst[k].index2):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index1) == int(term.metric_lst[k].index3):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index1) == int(term.metric_lst[k].index4):
												riemann.tensor_type = "Riemann"	
										if riemann.tensor_type == "Riemann":
											break						
						elif int(riemann.index2) == int(riemann.index4) and int(riemann.index2) not in riemann.deriv_lst:
							riemann.tensor_type = "Ricci13"
							for k in range(0, len(term.metric_lst)):
								if k != term.metric_lst.index(riemann):
									if int(riemann.index2) in term.metric_lst[k].deriv_lst:
										riemann.tensor_type = "Riemann"
									elif int(riemann.index2) == int(term.metric_lst[k].index1):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index2) == int(term.metric_lst[k].index2):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index2) == int(term.metric_lst[k].index3):
										riemann.tensor_type = "Riemann"
									elif int(riemann.index2) == int(term.metric_lst[k].index4):
										riemann.tensor_type = "Riemann"
								if riemann.tensor_type == "Ricci13":
									break
							if int(riemann.index1) == int(riemann.index3) and int(riemann.index1) not in riemann.deriv_lst:
								if riemann.tensor_type == "Ricci13":
									riemann.tensor_type = "Scalar-"
									for k in range(0, len(term.metric_lst)):
										if k != term.metric_lst.index(riemann):
											if int(riemann.index1) in term.metric_lst[k].deriv_lst:
												riemann.tensor_type = "Ricci13"
											elif int(riemann.index1) == int(term.metric_lst[k].index1):
												riemann.tensor_type = "Ricci13"
											elif int(riemann.index1) == int(term.metric_lst[k].index2):
												riemann.tensor_type = "Ricci13"
											elif int(riemann.index1) == int(term.metric_lst[k].index3):
												riemann.tensor_type = "Ricci13"
											elif int(riemann.index1) == int(term.metric_lst[k].index4):
												riemann.tensor_type = "Ricci13"
										if riemann.tensor_type == "Ricci13":
											break
								else:
									riemann.tensor_type = "Ricci24"
									for k in range(0, len(term.metric_lst)):
										if k != term.metric_lst.index(riemann):
											if int(riemann.index1) in term.metric_lst[k].deriv_lst:
												riemann.tensor_type = "Riemann"
											elif int(riemann.index1) == int(term.metric_lst[k].index1):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index1) == int(term.metric_lst[k].index2):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index1) == int(term.metric_lst[k].index3):
												riemann.tensor_type = "Riemann"
											elif int(riemann.index1) == int(term.metric_lst[k].index4):
												riemann.tensor_type = "Riemann"
										if riemann.tensor_type == "Riemann":
											break
					if riemann.tensor_type == "Riemann":
						riemann_str += "R_{" + "i_{" + str(riemann.index1) + "}" + "i_{" + str(riemann.index2) + "}" + "i_{" + str(riemann.index3) + "}" + "i_{" + str(riemann.index4) + "}"
						if len(riemann.deriv_lst) > 0:
							riemann_str += ";"
							for deriv in riemann.deriv_lst:
								riemann_str += "i_{" + str(deriv) + "}"
						riemann_str += "}"
					if riemann.tensor_type[0:5] == "Ricci":
						if riemann.tensor_type == "Ricci13":
							riemann_str += "Ric_{" + "i_{" + str(riemann.index1) + "}" + "i_{" + str(riemann.index3) + "}"
						if riemann.tensor_type == "Ricci14":
							term.coefficient *= -1
							riemann_str += "Ric_{" + "i_{" + str(riemann.index1) + "}" + "i_{" + str(riemann.index4) + "}"
						if riemann.tensor_type == "Ricci23":
							term.coefficient *= -1
							riemann_str += "Ric_{" + "i_{" + str(riemann.index2) + "}" + "i_{" + str(riemann.index3) + "}"
						if riemann.tensor_type == "Ricci24":
							riemann_str += "Ric_{" + "i_{" + str(riemann.index2) + "}" + "i_{" + str(riemann.index4) + "}"
						if len(riemann.deriv_lst) > 0:
							riemann_str += ";"
							for deriv in riemann.deriv_lst:
								riemann_str += "i_{" + str(deriv) + "}"
						riemann_str += "}"
					if riemann.tensor_type[0:6] == "Scalar":
						if riemann.tensor_type[-1] == "-":
							term.coefficient *= -1
						riemann_str += "K"
						if len(riemann.deriv_lst) > 0:
							riemann_str += "_{;"
							for deriv in riemann.deriv_lst:
								riemann_str += "i_{" + str(deriv) + "}"
							riemann_str += "}"
			line_counter += 1
			if line_counter == 16:
				line_counter = 1
				new_line = new_line[:-3]
				new_line += "\n" + "\\end{align*}" + "\n" + "\\newpage" + "\n" + "\\begin{align*}"
			if term.coefficient < 0:
				if term.coefficient == -1:
					new_line += "\n &- \\int_M "
				else:
					new_line += "\n &" + latex(sympify(term.coefficient)) + " \\int_M "
			else:
				if term.coefficient == 1:
					if new_line[-1] != "=":
						new_line += "\n &+\\int_M "
					else:
						new_line += "\n &\\int_M "
				else:
					if new_line[-1] != "=":
						new_line += "\n &+ " + latex(sympify(term.coefficient)) + " \\int_M "
					else:
						new_line += "\n &" + latex(sympify(term.coefficient)) + " \\int_M "
			new_line += riemann_str
			if len(term.constant_lst) > 1:
				constant_indices_lst = []
				for constant in term.constant_lst:
					constant_indices_lst.append(constant.index)
				constant_indices_unique = list(set(constant_indices_lst))
				if len(constant_indices_unique) == 1 and len(constant_indices_lst) == 2:
					new_line += "\\left| A \\right|^2"
				else:
					for k in range(0, len(constant_indices_unique)):
						new_line += "a^" + str(constant_indices_lst.count(constant_indices_unique[k])) + "_{i_" + str(k+1) + "}"
			elif len(term.constant_lst) == 1:
				constant = term.constant_lst[0]
				if len(constant.dirac_deriv_lst) > 0:
					if len(constant.dirac_deriv_lst) == 1:
						new_line += "div\\left( A \\right)"
					else:
						new_line += "\\partial^" + "{" + str(len(constant.dirac_deriv_lst)) + "}_{"
						dirac_index_lst = []
						for dirac in constant.dirac_deriv_lst:
							dirac_index_lst.append(dirac.index)
						dirac_index_lst = list(set(dirac_index_lst))
						for dirac_index in dirac_index_lst:
							new_line += "x_{" + "{i_{" + str(dirac_index) + "}" + "}" + "},"
						new_line = new_line[:-1]
						new_line += "}"
						new_line += "a_{" + "i_{" + str(constant.index) + "}" + "}"
			new_line += "\\left|dx\\right| \\\\"
		latex_doc.write(new_line+"\n"+"\\end{align*}"+"\n")
	latex_doc.write("\\end{document}")
	latex_doc.close()	
normal_coord_output = recursion(4)
generate_coefficient_latex(normal_coord_output)