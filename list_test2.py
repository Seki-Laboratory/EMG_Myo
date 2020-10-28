import functools, operator

list = [[0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5]]
new_list = [functools.reduce(operator.add, x) for x in zip(*list)]

print(new_list)