# Perceptron practical
# For this practical, numpy and other numerical libraries are forbidden. You may use only Python standard libraries and code you write and submit yourself.

# Tasks:

# 1. Implement your own Scalar and Vector classes, without using any other modules:

# doesn't work without this:
from __future__ import annotations

from copy import deepcopy

from typing import Union, List
from math import sqrt
from random import shuffle, seed

import matplotlib.pyplot as plt

class Scalar:
  def __init__(self: Scalar, val: float):
    self.val = float(val)
  def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
    # hint: use isinstance to decide what `other` is
    # raise an error if `other` isn't Scalar or Vector!
    if isinstance(other, Scalar):
        return Scalar(self.val * other.val)
    elif isinstance(other, Vector):
        return Vector(*[entry*self.val for entry in other.entries])
  def __add__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val + other.val)
  def __sub__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val - other.val)
  def __truediv__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val/other.val) # implement division of scalars
  def __rtruediv__(self: Scalar, other: Vector) -> Vector:
    return Vector([entry/self.val for entry in other.entries]) # implement division of vector by scalar
  def __repr__(self: Scalar) -> str:
    return "Scalar(%r)" % self.val
  def sign(self: Scalar) -> int:
    return Scalar(-1 + int(self.val >= 0) + int(self.val > 0)) # returns -1, 0, or 1
  def __float__(self: Scalar) -> float:
    return self.val
  def __gt__(self: Scalar, other: Scalar) -> bool:
    return self.val > other.val
  def __lt__(self: Scalar, other: Scalar) -> bool:
    return self.val < other.val
  def __eq__(self: Scalar, other: Scalar) -> bool:
    return self.val == other.val
  def __ge__(self: Scalar, other: Scalar) -> bool:
    return self.__gt__(other) or self.__eq__(other)
  def __le__(self: Scalar, other: Scalar) -> bool:
    return self.__lt__(other) or self.__eq__(other)

class Vector:
  def __init__(self: Vector, *entries: List[float]):
    self.entries = list(entries)
  def zero(size: int) -> Vector:
    return Vector(*[0 for i in range(size)])
  def __add__(self: Vector, other: Vector) -> Vector:
    return Vector(*[i+j for i,j in zip(self.entries, other.entries)])
  def __sub__(self: Vector, other: Vector) -> Vector:
    return self + Scalar(-1)*other
  def __mul__(self: Vector, other: Vector) -> Scalar:
    return Scalar(sum([i*j for i,j in zip(self.entries, other.entries)]))
  def magnitude(self: Vector) -> Scalar:
    return Scalar(sqrt(sum([i**2 for i in self.entries])))
  def unit(self: Vector) -> Vector:
    return self / self.magnitude()
  def __len__(self: Vector) -> int:
    return len(self.entries)
  def __repr__(self: Vector) -> str:
    return "Vector%s" % repr(self.entries)
  def __iter__(self: Vector):
    return iter(self.entries)
  def __getitem__(self: Vector, index: int) -> Scalar:
    return Scalar(self.entries[index])
  def __setitem__(self: Vector, index: int, value: Scalar):
    self.entries[index] = value.val


# 2. Implement the PerceptronTrain and PerceptronTest functions, using your Vector and Scalar classes. Do not permute the dataset when training; run through it linearly.
# (Hint on how to use the classes: make w and x instances of Vector, y and b instances of Scalar. What should the type of D be? Where do you see the vector operation formulas?)

def PerceptronTrain(D, MaxIter):
  assert len(set(len(i) for i in D[0])) == 1

  w = Vector(*[0 for i in range(len(D[0][0]))])
  b = Scalar(0)
  for epoch in range(MaxIter):
    for x, y in zip(*D):
      a = w * x + b
      if y * a <= Scalar(0):
        for dim in range(len(x)):
          w[dim] += y * x[dim]
        b += y
  return w, b

def PerceptronTest(weights: Vector, bias: Scalar, x: Vector):
  a = weights * x + bias
  return a.sign()

# 3. Make a 90-10 test-train split and evaluate your algorithm on the following dataset:
from random import randint
v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]

seed(1138)
shuffle(xs)
x_train, x_test = xs[:int(len(xs)*0.9)], xs[int(len(xs)*0.9):]

seed(1138)
shuffle(ys)
y_train, y_test = ys[:int(len(ys)*0.9)], ys[int(len(ys)*0.9):]

w, b = PerceptronTrain((x_train, y_train), 1000)
acc = sum([PerceptronTest(w, b, x) == y.sign() for x,y in zip(x_test, y_test)])/len(x_test)
# You should get that w is some multiple of v, and the performance should be very good. (Some noise is introduced by the last factor in y.)
print(f"Task 3 (Linear dataset) Accuracy: {acc}, v: {v}, w: {w}")

# 4. Make a 90-10 test-train split and evaluate your algorithm on the xor dataset:
from random import randint
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [Scalar(1) if x.entries[0]*x.entries[1] < 0 else Scalar(-1) for x in xs]

seed(1138)
shuffle(xs)
x_train, x_test = xs[:int(len(xs)*0.9)], xs[int(len(xs)*0.9):]

seed(1138)
shuffle(ys)
y_train, y_test = ys[:int(len(ys)*0.9)], ys[int(len(ys)*0.9):]

w, b = PerceptronTrain((x_train, y_train), 1000)
acc = sum([PerceptronTest(w, b, x) == y for x,y in zip(x_test, y_test)])/len(x_test)
# You should get some relatively random w, and the performance should be terrible.
print(f"Task 4 (XOR dataset) Accuracy: {acc}, w: {w}")


# 5. Sort the training data from task 3 so that all samples with y < 0 come first, then all samples with y = 0, then all samples with y > 0. (That is, sort by y.)
from random import randint
v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]

data = sorted([(x,y) for x,y in zip(xs, ys)], key=lambda v: v[1])
xs = [Vector(x1, x2) for (x1,x2),y in data]
ys = [Scalar(y) for (x1,x2),y in data]

# Graph the performance (computed by PerceptronTest) on both train and test sets versus epochs for perceptrons trained on
def PerceptronFineTune(D, w=None, b=None, MaxIter=1):
  assert len(set(len(i) for i in D[0])) == 1

  if w is None:
    w = Vector(*[0 for i in range(len(D[0][0]))])
  
  if b is None:
    b = Scalar(0)
  
  for epoch in range(MaxIter):
    for x, y in zip(*D):
      a = w * x + b
      if y * a <= Scalar(0):
        for dim in range(len(x)):
          w[dim] += y * x[dim]
        b += y
  return w, b

def PerceptronEval(weights: Vector, bias: Scalar, x: list[Vector], y: list[Scalar]):
  return sum([PerceptronTest(w, b, x_i) == y_i.sign() for x_i, y_i in zip(x, y)])/len(x)

# no permutation
x_train, x_test = xs[:int(len(xs)*0.9)], xs[int(len(xs)*0.9):]
y_train, y_test = ys[:int(len(ys)*0.9)], ys[int(len(ys)*0.9):]

train_accs_no_perm, test_accs_no_perm = [], []
w, b = None, None

N_EPOCHS = 100

for epoch in range(N_EPOCHS):
  w, b = PerceptronFineTune((x_train, y_train), w, b)
  acc_train = PerceptronEval(w, b, x_train, y_train)
  acc_test = PerceptronEval(w, b, x_test, y_test)
  train_accs_no_perm.append(acc_train)
  test_accs_no_perm.append(acc_test)

# random permutation at the beginning
from random import shuffle

data = ([(x,y) for x,y in zip(xs, ys)])
shuffle(data)
xs1 = [Vector(x1, x2) for (x1,x2),y in data]
ys1 = [Scalar(y) for (x1,x2),y in data]

x_train, x_test = xs1[:int(len(xs1)*0.9)], xs1[int(len(xs1)*0.9):]
y_train, y_test = ys1[:int(len(ys1)*0.9)], ys1[int(len(ys1)*0.9):]

train_accs_begin, test_accs_begin = [], []
w, b = None, None

for epoch in range(N_EPOCHS):
  w, b = PerceptronFineTune((x_train, y_train), w, b)
  acc_train = PerceptronEval(w, b, x_train, y_train)
  acc_test = PerceptronEval(w, b, x_test, y_test)
  train_accs_begin.append(acc_train)
  test_accs_begin.append(acc_test)

# random permutation at each epoch
# (This replicates graph 4.4 from Daumé.)

x_train, x_test = xs[:int(len(xs)*0.9)], xs[int(len(xs)*0.9):]
y_train, y_test = ys[:int(len(ys)*0.9)], ys[int(len(ys)*0.9):]

train_accs_each, test_accs_each = [], []
w, b = None, None

for epoch in range(N_EPOCHS):
  data_train = ([(x,y) for x,y in zip(x_train, y_train)])
  shuffle(data_train)
  x_train = [Vector(x1, x2) for (x1,x2),y in data_train]
  y_train = [Scalar(y) for (x1,x2),y in data_train]

  w, b = PerceptronFineTune((x_train, y_train), w, b)
  acc_train = PerceptronEval(w, b, x_train, y_train)
  acc_test = PerceptronEval(w, b, x_test, y_test)
  train_accs_each.append(acc_train)
  test_accs_each.append(acc_test)

plt.figure()
epochs = list(range(N_EPOCHS))
plt.plot(epochs, train_accs_no_perm, label='Train no permutation')
plt.plot(epochs, test_accs_no_perm, label='Test no permutation')
plt.plot(epochs, train_accs_begin, label='Train permutation at beginning')
plt.plot(epochs, test_accs_begin, label='Test permutation at beginning')
plt.plot(epochs, train_accs_each, label='Train permutation each epoch')
plt.plot(epochs, test_accs_each, label='Test permutation each epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Implement AveragedPerceptronTrain; using random permutation at each epoch, compare its performance with PerceptronTrain using the dataset from task 3.
def AveragedPerceptronTrain(D, MaxIter):
  assert len(set(len(i) for i in D[0])) == 1

  w = Vector(*[0 for i in range(len(D[0][0]))])
  b = Scalar(0)

  u = Vector(*[0 for i in range(len(D[0][0]))])
  B = Scalar(0)

  c = Scalar(1)

  D = [(x, y) for x,y in zip(*D)]

  for epoch in range(MaxIter):
    shuffle(D)
    for x, y in D:
      a = w * x + b
      if y * a <= Scalar(0):
        for dim in range(len(x)):
          w[dim] += y * x[dim]
          u[dim] += y * c * x[dim]
        b += y
        B += y * c
      c += Scalar(1)
  return w - (Scalar(1)/c*u), b - (Scalar(1)/c*B)

def PerceptronTrainShuffle(D, MaxIter):
  assert len(set(len(i) for i in D[0])) == 1

  w = Vector(*[0 for i in range(len(D[0][0]))])
  b = Scalar(0)

  D = [(x, y) for x,y in zip(*D)]

  for epoch in range(MaxIter):
    shuffle(D)
    for x, y in D:
      a = w * x + b
      if y * a <= Scalar(0):
        for dim in range(len(x)):
          w[dim] += y * x[dim]
        b += y
  return w, b

x_train, x_test = xs[:int(len(xs)*0.9)], xs[int(len(xs)*0.9):]
y_train, y_test = ys[:int(len(ys)*0.9)], ys[int(len(ys)*0.9):]

weights, bias = PerceptronTrainShuffle((x_train, y_train), 10)
weights_avg, bias_avg = AveragedPerceptronTrain((x_train, y_train), 10)
acc_train = PerceptronEval(weights, bias, x_train, y_train)
acc_test = PerceptronEval(weights, bias, x_test, y_test)
acc_train_avg = PerceptronEval(weights_avg, bias_avg, x_train, y_train)
acc_test_avg = PerceptronEval(weights_avg, bias_avg, x_test, y_test)

print(f"Perceptron Accuracy: {acc_train} on train, {acc_test} on test")
print(f"Averaged perceptron Accuracy: {acc_train_avg} on train, {acc_test_avg} on test")  