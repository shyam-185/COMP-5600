import collections
import os
import sys
from typing import List, Tuple

from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.

# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a() -> Formula:
    # Predicates to use:
    Summer = Atom('Summer')               # whether it's summer
    California = Atom('California')       # whether we're in California
    Rain = Atom('Rain')                   # whether it's raining
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Implies(And(Atom('Summer'), Atom('California')), Not(Atom('Rain')))
    # END_YOUR_CODE

# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b() -> Formula:
    # Predicates to use:
    Rain = Atom('Rain')              # whether it is raining
    Wet = Atom('Wet')                # whether it it wet
    Sprinklers = Atom('Sprinklers')  # whether the sprinklers are on
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Equiv(Atom('Wet'), Or(Atom('Rain'), Atom('Sprinklers')))
    # END_YOUR_CODE

# Sentence: "Either it's day or night (but not both)."
def formula1c() -> Formula:
    # Predicates to use:
    Day = Atom('Day')     # whether it's day
    Night = Atom('Night') # whether it's night
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return And(Or(Atom('Day'), Atom('Night')), Not(And(Atom('Day'), Atom('Night'))))
    # END_YOUR_CODE

############################################################
# Problem 2: first-order logic

# Sentence: "Every person has a mother."
def formula2a() -> Formula:
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Mother(x, y): return Atom('Mother', x, y)  # whether x's mother is y

    # Note: You do NOT have to enforce that the mother is a "person"
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Forall('$x',Exists('$y',Implies(Person('$x'),Mother('$x','$y'))))
    # END_YOUR_CODE

# Sentence: "At least one person has no children."
def formula2b() -> Formula:
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Child(x, y): return Atom('Child', x, y)    # whether x has a child y

    # Note: You do NOT have to enforce that the child is a "person"
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Exists('$x',Forall('$y',And(Person('$x'),Not(Child('$x','$y')))))
    # END_YOUR_CODE



############################################################
# Problem 3: Liar puzzle

# Facts:
# 0. Mark: "It wasn't me!"
# 1. John: "It was Nicole!"
# 2. Nicole: "No, it was Susan!"
# 3. Susan: "Nicole's a liar."
# 4. Exactly one person is telling the truth.
# 5. Exactly one person crashed the server.
# Query: Who did it?
# This function returns a list of 6 formulas corresponding to each of the
# above facts. Be sure your formulas are exactly in the order specified.
# Hint: You might want to use the Equals predicate, defined in logic.py.  This
# predicate is used to assert that two objects are the same.
# In particular, Equals(x,x) = True and Equals(x,y) = False iff x is not equal to y.
def liar() -> Tuple[List[Formula], Formula]:
    def TellTruth(x): return Atom('TellTruth', x)
    def CrashedServer(x): return Atom('CrashedServer', x)
    mark = Constant('mark')
    john = Constant('john')
    nicole = Constant('nicole')
    susan = Constant('susan')
    formulas = []
    # We provide the formula for fact 0 here.
    formulas.append(Equiv(TellTruth(mark), Not(CrashedServer(mark))))
    # You should add 5 formulas, one for each of facts 1-5.

    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    formulas.append((Equiv(TellTruth(john), CrashedServer(nicole))))
    formulas.append((Equiv(TellTruth(nicole), CrashedServer(susan))))
    formulas.append((Equiv(TellTruth(susan), Not(TellTruth(nicole)))))
    formulas.append((Equiv(TellTruth(john), CrashedServer(nicole))))
    formulas.append(Exists('$y', Forall('$x', Equiv(TellTruth('$x'), Equals('$x', '$y')))))
    formulas.append(Exists('$y', Forall('$x', Equiv(CrashedServer('$x'), Equals('$x', '$y')))))

    # END_YOUR_CODE
    query = CrashedServer('$x')
    return (formulas, query)

############################################################
# Problem 4: Odd and even integers

# Return the following 6 laws. Be sure your formulas are exactly in the order specified.
# 0. Each number $x$ has exactly one successor, which is not equal to $x$.
# 1. Each number is either even or odd, but not both.
# 2. The successor number of an even number is odd.
# 3. The successor number of an odd number is even.
# 4. For every number $x$, the successor of $x$ is larger than $x$.
# 5. Larger is a transitive property: if $x$ is larger than $y$ and $y$ is
#    larger than $z$, then $x$ is larger than $z$.
# Query: For each number, there exists an even number larger than it.
def ints() -> Tuple[List[Formula], Formula]:
    def Even(x): return Atom('Even', x)                  # whether x is even
    def Odd(x): return Atom('Odd', x)                    # whether x is odd
    def Successor(x, y): return Atom('Successor', x, y)  # whether x's successor is y
    def Larger(x, y): return Atom('Larger', x, y)        # whether x is larger than y
    # Note: all objects are numbers, so we don't need to define Number as an
    # explicit predicate.
    # Note: pay attention to the order of arguments of Successor and Larger.
    # Populate |formulas| with the 6 laws above and set |query| to be the
    # query.
    # Hint: You might want to use the Equals predicate, defined in logic.py.  This
    # predicate is used to assert that two objects are the same.
    formulas = []
    query = None
    # BEGIN_YOUR_CODE (our solution is 23 lines of code, but don't worry if you deviate from this)
    #0)
    formulas.append(Forall('$x', Exists('$y', And(Forall('$z', Equiv(Successor('$x','$z'), Equals('$z', '$y'))), Not(Equals('$x','$y'))))))
    
    #1)
    formulas.append(Forall('$x', And(Or(Even('$x'), Odd('$x')), Not(And(Even('$x'), Odd('$x'))))))

    #2)
    formulas.append(Forall('$x', Forall('$y', Implies(And(Even('$x'), Successor('$x', '$y')), Odd('$y')))))

    #3)
    formulas.append(Forall('$x', Forall('$y', Implies(And(Odd('$x'), Successor('$x', '$y')), Even('$y')))))

    #4)
    formulas.append(Forall('$x', Forall('$y', Implies(Successor('$x', '$y'), Larger('$y', '$x')))))

    #5)
    formulas.append(Forall('$x', Forall('$y', Forall('$z', Implies(And(Larger('$x', '$y'), Larger('$y', '$z')), Larger('$x', '$z'))))))

    # END_YOUR_CODE
    query = Forall('$x', Exists('$y', And(Even('$y'), Larger('$y', '$x'))))
    return (formulas, query)