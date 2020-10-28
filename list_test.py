import numpy as np





def square(list):
    r = np.array(list)
    j = r
    result = r+j

    return result

def main():
    a = np.array([1,0,2,4,5,8,6,2])
    print(a)
    print(square(a))
    print(square(a))


if __name__ == '__main__':
  main()
# rms = np.sqrt(np.square(a).mean(axis=0))
# print(rms.shape)  # (252,)
# print(rms)