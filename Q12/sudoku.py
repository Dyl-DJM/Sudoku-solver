#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import math
import sys
import os
import time

"""Sudoku"""


def var(i, j, k):
    """Return the literal Xijk.

    >>> var(2,3,4)
    (1, 2, 3, 4)
    """
    return (1, i, j, k)


def neg(l):
    """Return the negation of the literal l.

    >>> neg(var(2,3,4))
    (-1, 2, 3, 4)
    """
    (s, i, j, k) = l
    return (-s, i, j, k)


def initial_configuration():
    """Return the initial configuration of the example in td6.pdf

    >>> cnf = initial_configuration()
    >>> [(1, 1, 4, 4)] in cnf
    True
    >>> [(1, 2, 1, 2)] in cnf
    True
    >>> [(1, 2, 3, 1)] in cnf
    False
    """
    return [[var(2, 1, 2)], [var(1, 4, 4)], [var(3, 2, 1)], [var(4, 3, 1)]]


def at_least_one(L):
    """Return a cnf that represents the constraint: at least one of the
    literals in the list L is true.

    >>> lst = [var(1, 1, 1), var(2, 2, 2), var(3, 3, 3)]
    >>> cnf = at_least_one(lst)
    >>> len(cnf)
    1
    >>> clause = cnf[0]
    >>> len(clause)
    3
    >>> clause.sort()
    >>> clause == [var(1, 1, 1), var(2, 2, 2), var(3, 3, 3)]
    True
    """
    return [L]


def at_most_one(L):
    """Return a cnf that represents the constraint: at most one of the
    literals in the list L is true

    >>> lst = [var(1, 1, 1), var(2, 2, 2), var(3, 3, 3)]
    >>> cnf = at_most_one(lst)
    >>> len(cnf)
    3
    >>> cnf[0].sort()
    >>> cnf[1].sort()
    >>> cnf[2].sort()
    >>> cnf.sort()
    >>> cnf == [[neg(var(1,1,1)), neg(var(2,2,2))], \
    [neg(var(1,1,1)), neg(var(3,3,3))], \
    [neg(var(2,2,2)), neg(var(3,3,3))]]
    True
    """
    result = []
    for (x, y) in itertools.combinations(L, 2):
        result.append([neg(x), neg(y)])
    return result


def assignment_rules(N):
    """Return a list of clauses describing the rules for the assignment (i,j) -> k.

    >>> cnf = assignment_rules(4)
    >>> len(cnf)
    112
    >>> for clause in cnf[0:8]: print(clause)
    [(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 1, 3), (1, 1, 1, 4)]
    [(-1, 1, 1, 1), (-1, 1, 1, 2)]
    [(-1, 1, 1, 1), (-1, 1, 1, 3)]
    [(-1, 1, 1, 1), (-1, 1, 1, 4)]
    [(-1, 1, 1, 2), (-1, 1, 1, 3)]
    [(-1, 1, 1, 2), (-1, 1, 1, 4)]
    [(-1, 1, 1, 3), (-1, 1, 1, 4)]
    [(1, 1, 2, 1), (1, 1, 2, 2), (1, 1, 2, 3), (1, 1, 2, 4)]
    """
    cnf = []
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            # add clauses to cnf saying that (i,j) contains
            # *exactly* one of the digits k=1..N
            L = []
            for n in range(1, N + 1):
                L.append(var(i, j, n))
            cnf.append(at_least_one(L)[0])
            tmp = at_most_one(L)
            for clause in tmp:
                cnf.append(clause)
    return cnf


def row_rules(N):
    """Return a list of clauses describing the rules for the rows.

    >>> cnf = row_rules(4)
    >>> len(cnf)
    112
    """
    cnf = []
    for line in range(1, N + 1):
        for k in range(1, N + 1):
            L = []
            for case in range(1, N + 1):
                L.append((var(line, case, k)))
            cnf.append(at_least_one(L)[0])
            tmp = at_most_one(L)
            for clause in tmp:
                cnf.append(clause)
    return cnf


def column_rules(N):
    """Return a list of clauses describing the rules for the columns.

    >>> cnf = column_rules(4)
    >>> len(cnf)
    112
    """
    cnf = []
    for case in range(1, N + 1):
        for k in range(1, N + 1):
            L = []
            for line in range(1, N + 1):
                L.append((var(line, case, k)))
            cnf.append(at_least_one(L)[0])
            tmp = at_most_one(L)
            for clause in tmp:
                cnf.append(clause)
    return cnf


def subgrid(N, max_value_case, offset_line, offset_column, cnf):
    """Add into cnf the clauses for the subgrid of size N starting at (offset_line, offset_column)
    """
    for k in range(1, max_value_case + 1):
        L = []
        for line in range(1, N + 1):
            for column in range(1, N + 1):
                literal = var(line + offset_line, column + offset_column, k)
                L.append(literal)
        cnf.append(at_least_one(L)[0])
        for clause in at_most_one(L):
            cnf.append(clause)


def subgrid_rules(N):
    """Return a list of clauses describing the rules for the subgrids.

    >>> cnf = subgrid_rules(4)
    >>> len(cnf)
    112
    """
    sqrt = int(math.sqrt(N))
    cnf = []
    for i in range(0, N, sqrt):
        for j in range(0, N, sqrt):
            subgrid(sqrt, N, i, j, cnf)
    return cnf


def generate_rules(N):
    """Return a list of clauses describing the rules of the game and print the time taken to generate them.
    """
    start_time = time.time()
    cnf = []
    cnf.extend(assignment_rules(N))
    cnf.extend(row_rules(N))
    cnf.extend(column_rules(N))
    cnf.extend(subgrid_rules(N))
    print("Time taken to generate clauses: %.6f seconds" % (time.time() - start_time))
    return cnf


def literal_to_integer(l, N):
    """Return the external representation of the literal l.

    >>> literal_to_integer(var(1,2,3), 4)
    7
    >>> literal_to_integer(neg(var(3,2,1)), 4)
    -37
    """
    (s, i, j, k) = l
    return s * ((N ** 2) * (i - 1) + N * (j - 1) + k)


def translateCNF(cnf, N, filename):
    """Translate a CNF formula into a syntax for the MiniSat solver. Produces a
    new output file in which we write the problem.

    """
    dico_translation = dict()
    output = open(filename, "w")
    number_clauses = len(cnf)
    number_variables = N ** 3
    clauses = []
    for clause in cnf:
        tmp = []
        for literal in clause:
            int_value = literal_to_integer(literal, N)
            dico_translation[int_value] = literal
            tmp.append(str(int_value))
        clauses.append(" ".join(tmp))

    output.write("p cnf " + str(number_variables) + " " + str(number_clauses) + "\n")
    output.write(" 0\n".join(clauses))
    output.write(" 0")
    output.close()
    return dico_translation


def init_sudoku(list_str):
    """Return a CNF formula describing the state of the initial sudoku field thanks to
    a list of string lines.

    """
    cnf = []
    j = 0
    for line in list_str:
        line_list = line.split()
        for i in range(len(line_list)):
            if line_list[i] != "0":
                cnf.append([var(i + 1, j + 1, int(line_list[i]))])
        j += 1
    return cnf


def proper_display(filename, N, dico_translation):
    """Show properly the asnwer of the sudoku probleme solved in the filename file. In the
    answer minisat format.
    
    """
    file_obj = open(filename, "r")
    lines = file_obj.readlines()
    compteur = 1
    solution = [ [0]*N for i in range(N)]
    if lines[0] != "SAT\n":
        print("There is no answer for this sudoku configuration")
        return
    for integer_str in lines[1].split()[:-1]:
        key = int(integer_str)
        (s, i, j, k) = dico_translation[key]
        if s == 1:
            solution[j - 1][i - 1] = k
    for line in solution:
        values = " ".join([str(i) for i in line])
        sys.stdout.write(values + "\n")
        
    file_obj.close()


def main():
    if len(sys.argv) != 2:
        print("Error in the command line")
        exit(1)
    filename = sys.argv[1]
    file_obj = open(filename, "r")
    output_sat = "sudoku.out"
    lines = file_obj.readlines()
    N = int(lines[0])
    cnf = init_sudoku(lines[1:]) + generate_rules(N)
    dico_translation = translateCNF(cnf, N, "sudoku.cnf")
    start_time = time.time()
    os.system("minisat sudoku.cnf " + output_sat + " > /dev/null")
    print("Time taken to solve the problem: %.6f seconds" % (time.time() - start_time))
    proper_display(output_sat, N, dico_translation)
    file_obj.close()



if __name__ == "__main__":
    main()


